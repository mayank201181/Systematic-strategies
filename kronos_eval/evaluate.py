"""Score Kronos zero-shot forecasts against naive and trend baselines, after costs.

    python kronos_eval/evaluate.py [library_returns.parquet]

Everything is in daily-vol units per instrument: sigma_t is the EWMA (span 32)
std of daily price changes known at t. A forecast of 10 means one unit of
risk. Daily P&L per instrument = f_t/10 * dP_{t+1}/sigma_t; the portfolio is the
equal-weight mean across instruments, scaled to 10% annual vol ex post.
Costs: KAPPA daily-vols per unit of risk traded (0.02 is conservative for
liquid futures; ags/energy sit near it, ES/ZN well under).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"
START, SPLIT = "2024-07-01", "2025-07-01"
KAPPAS = [0.0, 0.01, 0.02]


def ewmac(p, sig, f, s):
    return (p.ewm(span=f).mean() - p.ewm(span=s).mean()) / sig


def scale(f, cap=20):
    oos = f.loc[START:]
    return (f * 10 / oos.abs().stack().mean()).clip(-cap, cap)


def portfolio(f, ret, kappa):
    """f: forecasts at t (dates x inst). ret: dP_{t+1}/sigma_t aligned on t."""
    f = f.reindex_like(ret)
    pnl = (f / 10 * ret)
    cost = kappa * (f / 10).diff().abs()
    net = (pnl - cost.fillna(0)).where(ret.notna() & f.notna())
    # P&L earned over t -> t+1 is booked on t+1, like the library's returns
    port = net.mean(axis=1).shift(1).loc[START:].dropna()
    return port


def stats(r):
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return dict(SR=np.nan)
    sr = r.mean() / r.std() * np.sqrt(252)
    eq = (r / r.std() * 0.10 / np.sqrt(252)).cumsum()
    return dict(SR=round(sr, 2), tstat=round(sr * np.sqrt(len(r) / 252), 2),
                maxDD_10vol=round((eq - eq.cummax()).min() * 100, 1), days=len(r))


def main():
    px = pd.read_parquet(D / "ohlcv_adj.parquet")
    close = px.pivot(index="date", columns="instrument", values="close").sort_index()
    close = close[close.index.dayofweek < 5]              # futures calendar (crypto weekends dropped)
    # score only on each instrument's own trading days
    sig = close.apply(lambda s: s.dropna().diff().ewm(span=32, min_periods=20).std()).reindex(close.index)
    dnext = close.apply(lambda s: s.dropna().diff().shift(-1))
    ret1 = dnext / sig                                    # next-day move, vol units
    r5 = close.apply(lambda s: (s.dropna().shift(-5) - s.dropna()))
    ret5 = r5 / (sig * np.sqrt(5))

    fc = pd.read_parquet(D / "forecasts.parquet")
    piv = lambda c: fc.pivot(index="date", columns="instrument", values=c).reindex(close.index)
    c0 = piv("close")
    k1 = (piv("mean_1") - c0) / sig
    k5 = (piv("mean_5") - c0) / (sig * np.sqrt(5))
    disp5 = piv("std_5") / (sig * np.sqrt(5))
    up5 = piv("up_5") - 0.5

    sigs = {
        "kronos_h1": k1, "kronos_h5": k5,
        "kronos_h5_smooth": k5.ewm(span=5).mean(),
        "kronos_prob_up": up5,
        "ewmac_8_32": ewmac(close, sig, 8, 32), "ewmac_16_64": ewmac(close, sig, 16, 64),
        "ewmac_32_128": ewmac(close, sig, 32, 128),
        "reversal_5d": -(close - close.shift(5)) / (sig * np.sqrt(5)),
        "long_only": close * 0 + 1.0,
    }
    # Viral claim: trade only when the sampled paths agree (low dispersion)
    rk = disp5.rank(axis=1, pct=True)
    sigs["kronos_h5_tight_only"] = k5.where(rk <= 1 / 3, 0.0).where(k5.notna())
    sigs["kronos_h5_wide_only"] = k5.where(rk > 2 / 3, 0.0).where(k5.notna())
    mask = k5.notna()
    sigs = {k: (v.where(mask) if not k.startswith("ewmac") and k != "long_only" else v.where(mask))
            for k, v in sigs.items()}
    sigs = {k: scale(v) if k != "long_only" else v * 10 for k, v in sigs.items()}

    OUT.mkdir(exist_ok=True)
    # 1. forecast skill
    skill = []
    for name, f, tgt, h in [("kronos_h1", k1, ret1, 1), ("kronos_h5", k5, ret5, 5),
                            ("ewmac_16_64", sigs["ewmac_16_64"], ret5, 5),
                            ("reversal_5d", sigs["reversal_5d"], ret5, 5)]:
        a, b = f.loc[START:].stack(), tgt.loc[START:].reindex_like(f.loc[START:]).stack()
        j = pd.concat([a, b], axis=1, keys=["f", "y"]).dropna()
        ic_t = j.groupby(level=0).apply(lambda g: g.f.rank().corr(g.y.rank()) if len(g) > 5 else np.nan)
        ts_ic = j.groupby(level=1).apply(lambda g: g.f.corr(g.y, method="spearman")).mean()
        skill.append(dict(signal=name, horizon=h, n=len(j),
                          pooled_rank_IC=round(j.f.rank().corr(j.y.rank()), 4),
                          xs_IC=round(ic_t.mean(), 4),
                          xs_IC_t=round(ic_t.mean() / ic_t.std() * np.sqrt(len(ic_t) / h), 2),
                          ts_IC_avg_inst=round(ts_ic, 4),
                          hit_rate=round((np.sign(j.f) == np.sign(j.y)).mean(), 4)))
    skill = pd.DataFrame(skill)
    # price-level error vs random walk
    e_k = ((piv("mean_1") - close.shift(-1)).abs() / sig).loc[START:].stack()
    e_rw = ((c0 - close.shift(-1)).abs() / sig).loc[START:].reindex_like(piv("mean_1").loc[START:]).stack()
    jj = pd.concat([e_k, e_rw], axis=1, keys=["k", "rw"]).dropna()
    mae = dict(kronos_MAE_h1=round(jj.k.mean(), 4), random_walk_MAE_h1=round(jj.rw.mean(), 4))
    # dispersion as a vol forecast: does std_5 rank |5d move| better than trailing vol?
    a = (piv("std_5")).loc[START:].stack()
    b = (sig * np.sqrt(5)).loc[START:].reindex_like(piv("std_5").loc[START:]).stack()
    y = r5.abs().loc[START:].reindex_like(piv("std_5").loc[START:]).stack()
    v = pd.concat([a, b, y], axis=1, keys=["k", "ewma", "y"]).dropna()
    per = v.groupby(level=1).apply(lambda g: pd.Series({"kronos_disp": g.k.corr(g.y, "spearman"),
                                                         "trailing_vol": g.ewma.corr(g.y, "spearman")}))
    mae.update({f"volfc_rankcorr_{k}": round(x, 4) for k, x in per.mean().items()})

    # 2. trading
    rows, rets = [], {}
    for name, f in sigs.items():
        for kap in KAPPAS:
            p = portfolio(f, ret1, kap)
            rets[(name, kap)] = p
            st = stats(p)
            st1, st2 = stats(p.loc[:SPLIT]), stats(p.loc[SPLIT:])
            to = (f / 10).diff().abs().loc[START:].mean(axis=1).mean() * 252
            rows.append(dict(signal=name, kappa=kap, **st, SR_JulJun25=st1["SR"],
                             SR_Jul25_Oct26=st2["SR"], turnover_x_per_yr=round(to, 1)))
    trade = pd.DataFrame(rows)

    # 3. correlations (net kappa 0.02) incl. the library's live strategies
    net = pd.DataFrame({k: v for (k, kap), v in rets.items() if kap == 0.02})
    if len(sys.argv) > 1:
        lib = pd.read_parquet(sys.argv[1])[["trend_ewmac", "trend_carry", "multi_7", "mr_fast", "s183_replica"]]
        net = net.join(lib.add_prefix("lib_"), how="left")
    corr = net.corr().round(2)
    fcorr = pd.DataFrame({k: sigs[k].loc[START:].stack() for k in
                          ["kronos_h1", "kronos_h5", "ewmac_8_32", "ewmac_16_64", "reversal_5d"]}).corr().round(2)

    skill.to_csv(OUT / "skill.csv", index=False)
    trade.to_csv(OUT / "trading.csv", index=False)
    corr.to_csv(OUT / "pnl_correlation.csv")
    fcorr.to_csv(OUT / "forecast_correlation.csv")
    pd.Series(mae).to_csv(OUT / "price_error.csv")
    net.to_parquet(OUT / "returns_net.parquet")
    pd.set_option("display.width", 250)
    print(skill.to_string(), "\n", pd.Series(mae).to_string(), "\n")
    print(trade.to_string(), "\n")
    print(corr.to_string(), "\n")
    print(fcorr.to_string())


if __name__ == "__main__":
    main()
