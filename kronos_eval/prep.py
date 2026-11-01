"""Back-adjust Yahoo continuous OHLCV (roll gaps removed) for Kronos and baselines.

Yahoo `=F` series jump at each roll. For energy, ags and livestock a >6 sigma
close-to-close move in a delivery/pre-delivery month is treated as a roll gap
(same rule as syslib.live) and removed by shifting all earlier bars, so the
latest bars stay at today's price level.
"""
import sys
import numpy as np
import pandas as pd

ROLL = {"CRUDE_W", "BRENT_W", "GASOILINE", "HEATOIL", "GAS_US", "CORN", "SOYBEAN",
        "SOYOIL", "SOYMEAL", "WHEAT", "REDWHEAT", "OATIES", "RICE", "COCOA", "COFFEE",
        "SUGAR11", "COTTON2", "OJ", "LIVECOW", "LEANHOG", "FEEDCOW"}
SECTOR = {**{k: "Commodity" for k in ROLL},
          **{k: "Metals" for k in ["GOLD", "SILVER", "COPPER", "PLAT", "PALLAD", "ALUMINIUM"]},
          **{k: "FX" for k in ["AUD", "GBP", "CAD", "EUR", "JPY", "NZD", "CHF", "MXP", "BRE"]},
          **{k: "Rates" for k in ["US2", "US5", "US10", "US20", "US30"]},
          **{k: "Equity" for k in ["SP500", "NASDAQ", "RUSSELL_mini", "NIKKEI"]},
          **{k: "Crypto" for k in ["BITCOIN", "ETHEREUM"]}}


def adjust(g, inst):
    g = g.sort_values("date").drop_duplicates("date").set_index("date")
    g = g[["open", "high", "low", "close", "volume"]].astype(float)
    g = g[g.close.notna() & (g.close > 0)]
    for c in ["open", "high", "low"]:
        g[c] = g[c].where(g[c] > 0, g.close)
    g["high"] = g[["open", "high", "low", "close"]].max(axis=1)
    g["low"] = g[["open", "high", "low", "close"]].min(axis=1)
    g["volume"] = g.volume.fillna(0)
    d = g.close.diff()
    sig = d.abs().ewm(span=32, min_periods=10).mean().shift(1) * 1.25
    z = (d / sig).abs()
    gaps = pd.Series(0.0, index=g.index)
    if inst in ROLL:
        gaps[z > 6] = d[z > 6]
    # adj_t = raw_t + sum of gap moves after t
    off = gaps[::-1].cumsum()[::-1].shift(-1).fillna(0)
    for c in ["open", "high", "low", "close"]:
        g[c] = g[c] + off
    g["n_gaps"] = int((gaps != 0).sum())
    return g


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    raw = pd.read_parquet(src)
    raw = raw[raw.groupby("instrument").date.transform("count") > 1000]
    out = pd.concat({i: adjust(g, i) for i, g in raw.groupby("instrument")}, names=["instrument"])
    out = out.reset_index()
    out["sector"] = out.instrument.map(SECTOR)
    print(out.groupby("instrument").n_gaps.first().loc[lambda s: s > 0].to_string())
    out.drop(columns="n_gaps").to_parquet(dst)
    print(out.shape, out.date.max())
