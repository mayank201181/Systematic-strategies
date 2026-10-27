"""Extend the pysystemtrade history to today with free Yahoo Finance data.

Runs in GitHub Actions (this needs open internet):

    python -m syslib.live --base data/pst --live data/live --out data/merged

1. Continuous front-month closes (`CL=F` etc.) for the US-listed instruments,
   spliced onto the back-adjusted series as daily price changes. Yahoo's
   continuous series is not roll-adjusted, so changes beyond 6 sigma are
   treated as roll gaps and zeroed; every zeroed day is logged.
2. Individual contracts (`CLZ26.NYM` etc.) for the next listed months, stored
   in data/live/curve.csv. Front and next contract give the carry signal for
   the tail, and the file grows by one snapshot per run.

Instruments with no Yahoo coverage (most non-US futures) stop at the end of
the pysystemtrade data; the engine drops them from the portfolio from then on.
"""
import argparse
import datetime as dt
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PANELS

# pysystemtrade code -> (Yahoo root, exchange suffix, listed month cycle)
MONTHLY = "FGHJKMNQUVXZ"
QUARTERLY = "HMUZ"
YAHOO = {
    "SP500": ("ES", "CME", QUARTERLY), "NASDAQ": ("NQ", "CME", QUARTERLY),
    "RUSSELL_mini": ("RTY", "CME", QUARTERLY), "SP400": ("EMD", "CME", QUARTERLY),
    "NIKKEI": ("NIY", "CME", QUARTERLY),
    "US2": ("ZT", "CBT", QUARTERLY), "US5": ("ZF", "CBT", QUARTERLY),
    "US10": ("ZN", "CBT", QUARTERLY), "US20": ("ZB", "CBT", QUARTERLY),
    "US30": ("UB", "CBT", QUARTERLY),
    "AUD": ("6A", "CME", QUARTERLY), "GBP": ("6B", "CME", QUARTERLY),
    "CAD": ("6C", "CME", QUARTERLY), "EUR": ("6E", "CME", QUARTERLY),
    "JPY": ("6J", "CME", QUARTERLY), "NZD": ("6N", "CME", QUARTERLY),
    "CHF": ("6S", "CME", QUARTERLY), "MXP": ("6M", "CME", QUARTERLY),
    "BRE": ("6L", "CME", MONTHLY), "DX": ("DX", "NYB", QUARTERLY),
    "CRUDE_W": ("CL", "NYM", MONTHLY), "BRENT_W": ("BZ", "NYM", MONTHLY),
    "GASOILINE": ("RB", "NYM", MONTHLY), "HEATOIL": ("HO", "NYM", MONTHLY),
    "GAS_US": ("NG", "NYM", MONTHLY),
    "GOLD": ("GC", "CMX", "GJMQVZ"), "SILVER": ("SI", "CMX", "HKNUZ"),
    "COPPER": ("HG", "CMX", "HKNUZ"), "PLAT": ("PL", "NYM", "FJNV"),
    "PALLAD": ("PA", "NYM", "HMUZ"), "ALUMINIUM": ("ALI", "CMX", MONTHLY),
    "CORN": ("ZC", "CBT", "HKNUZ"), "SOYBEAN": ("ZS", "CBT", "FHKNQUX"),
    "SOYOIL": ("ZL", "CBT", "FHKNQUVZ"), "SOYMEAL": ("ZM", "CBT", "FHKNQUVZ"),
    "WHEAT": ("ZW", "CBT", "HKNUZ"), "REDWHEAT": ("KE", "CBT", "HKNUZ"),
    "OATIES": ("ZO", "CBT", "HKNUZ"), "RICE": ("ZR", "CBT", "FHKNUX"),
    "COCOA": ("CC", "NYB", "HKNUZ"), "COFFEE": ("KC", "NYB", "HKNUZ"),
    "SUGAR11": ("SB", "NYB", "HKNV"), "COTTON2": ("CT", "NYB", "HKNVZ"),
    "OJ": ("OJ", "NYB", "FHKNUX"),
    "LIVECOW": ("LE", "CME", "GJMQVZ"), "LEANHOG": ("HE", "CME", "GJKMNQVZ"),
    "FEEDCOW": ("GF", "CME", "FHJKQUVX"),
    "BITCOIN": ("BTC", "CME", MONTHLY), "ETHEREUM": ("ETH", "CME", MONTHLY),
}
ROLL_Z = 6.0


def _yf():
    import yfinance as yf
    return yf


def fetch_continuous(start):
    tick = {f"{r}=F": i for i, (r, _, _) in YAHOO.items()}
    df = _yf().download(list(tick), start=start, auto_adjust=False, progress=False,
                        threads=True)["Close"]
    df = df.rename(columns=tick)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


FX_CCYS = ["AUD", "CAD", "CHF", "CNH", "EUR", "GBP", "HKD", "JPY", "KRW", "MXN", "SEK", "SGD"]


def fetch_fx(start):
    tick = {f"{c}USD=X": ("MXP" if c == "MXN" else c) for c in FX_CCYS}
    df = _yf().download(list(tick), start=start, auto_adjust=False, progress=False,
                        threads=True)["Close"].rename(columns=tick)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def contract_codes(cycle, today, n):
    out = []
    y, m = today.year, today.month
    while len(out) < n:
        code = MONTHLY[m - 1]
        if code in cycle:
            out.append((code, y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def fetch_curve(today, n=4, period="5d"):
    """Close of the next `n` listed contracts for each instrument."""
    yf = _yf()
    rows = []
    tickers = {}
    for inst, (root, suf, cyc) in YAHOO.items():
        for code, y, m in contract_codes(cyc, today, n):
            tickers[f"{root}{code}{str(y)[2:]}.{suf}"] = (inst, y * 100 + m)
    df = yf.download(list(tickers), period=period, auto_adjust=False, progress=False,
                     threads=True)["Close"]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    for t, (inst, ym) in tickers.items():
        if t not in df:
            continue
        s = df[t].dropna()
        for d, v in s.items():
            rows.append({"date": d, "instrument": inst, "contract": ym, "close": v})
    return pd.DataFrame(rows)


def splice(base, yahoo, log):
    """Append Yahoo daily changes after the last base date (roll gaps zeroed)."""
    adj = base.adj.copy()
    out = {k: getattr(base, k).copy() for k in PANELS}
    for inst in yahoo.columns:
        if inst not in adj.columns:
            continue
        a = adj[inst].dropna()
        if a.empty:
            continue
        last = a.index[-1]
        y = yahoo[inst].dropna()
        # unit check: base price vs Yahoo on overlapping dates
        ov = base.price[inst].dropna().loc[y.index.min():].index.intersection(y.index)
        if len(ov) >= 5:
            ratio = (base.price[inst].loc[ov] / y.loc[ov]).median()
            scale = 10 ** round(np.log10(abs(ratio))) if np.isfinite(ratio) and ratio != 0 else 1
            y = y * scale
        d = y.diff().loc[last:].iloc[1:]
        if d.empty:
            continue
        sig = y.diff().abs().ewm(span=32, min_periods=10).mean().shift(1) * 1.25
        z = (d / sig.reindex(d.index)).abs()
        gaps = z[z > ROLL_Z].index
        for g in gaps:
            log.append({"date": g, "instrument": inst, "change": d.loc[g],
                        "z": z.loc[g]})
        d.loc[gaps] = 0.0
        tail = a.iloc[-1] + d.fillna(0).cumsum()
        idx = out["adj"].index.union(tail.index)
        for k in PANELS:
            out[k] = out[k].reindex(idx)
        out["adj"].loc[tail.index, inst] = tail.values
        out["price"].loc[tail.index, inst] = y.reindex(tail.index).values
    return out


def apply_curve(out, curve):
    """Tail carry from front/next listed contracts."""
    if curve is None or curve.empty:
        return out
    curve = curve.drop_duplicates(["date", "instrument", "contract"], keep="last")
    for (d, inst), g in curve.groupby(["date", "instrument"]):
        if inst not in out["adj"].columns or d not in out["adj"].index:
            continue
        g = g.sort_values("contract")
        # skip a contract in its delivery month
        live = g[g.contract > d.year * 100 + d.month]
        if len(live) < 2:
            continue
        c1, c2 = live.iloc[0], live.iloc[1]
        base_last = out["price_ym"][inst].last_valid_index()
        if base_last is not None and d <= base_last:
            continue
        to_y = lambda c: c // 100 + (c % 100 - 1) / 12
        out["price_ym"].loc[d, inst] = to_y(c1.contract)
        out["carry_ym"].loc[d, inst] = to_y(c2.contract)
        out["price"].loc[d, inst] = c1.close
        out["carry"].loc[d, inst] = c2.close
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/pst")
    ap.add_argument("--live", default="data/live")
    ap.add_argument("--out", default="data/merged")
    ap.add_argument("--backfill", action="store_true",
                    help="pull 3y of history for currently listed contracts")
    a = ap.parse_args()
    from .data import load_panel
    base = load_panel(a.base)
    live = Path(a.live)
    live.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp(dt.date.today())

    yahoo = fetch_continuous(start="2023-10-01")
    yahoo.to_parquet(live / "continuous.parquet")
    print(f"continuous: {yahoo.shape[1]} instruments, last {yahoo.index.max().date()}")

    curve_path = live / "curve.csv"
    old = pd.read_csv(curve_path, parse_dates=["date"]) if curve_path.exists() else None
    try:
        new = fetch_curve(today, n=6 if a.backfill else 4,
                          period="3y" if a.backfill else "10d")
        print(f"curve: {new.instrument.nunique() if len(new) else 0} instruments, "
              f"{len(new)} rows")
    except Exception as e:  # keep the run alive if contract tickers fail
        print(f"curve fetch failed: {e}")
        new = pd.DataFrame()
    curve = pd.concat([x for x in (old, new) if x is not None and len(x)], ignore_index=True) \
        if (old is not None or len(new)) else pd.DataFrame()
    if len(curve):
        curve = curve.drop_duplicates(["date", "instrument", "contract"], keep="last")
        curve.sort_values(["date", "instrument", "contract"]).to_csv(curve_path, index=False)

    log = []
    out = splice(base, yahoo, log)
    out = apply_curve(out, curve)
    pd.DataFrame(log).to_csv(live / "roll_filter_log.csv", index=False)
    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_parquet(dst / f"{k}.parquet")
    shutil.copy(Path(a.base) / "meta.csv", dst / "meta.csv")
    fx = base.fx.copy() if base.fx is not None else pd.DataFrame()
    try:
        yfx = fetch_fx("2024-01-01")
        last = fx.index.max()
        tail = yfx.loc[yfx.index > last]
        fx = pd.concat([fx, tail]).sort_index()
        fx["USD"] = 1.0
        print(f"fx: extended to {fx.index.max().date()}")
    except Exception as e:
        print(f"fx fetch failed, carrying last rates forward: {e}")
    fx.to_parquet(dst / "fx.parquet")
    print(f"merged panels to {out['adj'].index.max().date()}; "
          f"{len(log)} roll gaps zeroed")


if __name__ == "__main__":
    main()
