"""Replica of trends-research Strategy 183 ("IG VoV Quad Sharpened").

Follows flagship_S183/ig_strategy_183.py: per-instrument rolling-256 forecast
normalisation, 50/50 TS-EWMAC / XS-momentum trend, carry, skew and VoV sleeves
at 25% each, universe-pooled FDM from the correlation of universe-mean sleeve
forecasts (EWM 512, cap 2), halflife-1 smoothing, sigmoid vol/drawdown gates,
5-year per-instrument warm-up, Carver IDM table, equal instrument weights.

Differences forced by data: pysystemtrade prices instead of Norgate, 58 of the
62 instruments (no ASX 10y), carry from pysystemtrade's carry contract rather
than the next listed expiry.
"""
import numpy as np
import pandas as pd

from . import signals as S
from .strategies import LP_SPEEDS, Strategy, register
from .universe import S183

VOL_TARGET = 0.20
STEEP = 2.0 / VOL_TARGET
OOS_START = 1280


def _norm(x):
    return S.scale_forecast(x, mode="rolling", window=256)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def pooled_fdm_ewm(sleeves, span=512, min_periods=256, cap=2.0):
    means = pd.DataFrame({k: v.mean(axis=1) for k, v in sleeves.items()})
    corr = means.ewm(span=span, min_periods=min_periods).corr()
    n = len(sleeves)
    w = np.full(n, 1.0 / n)
    out = pd.Series(1.0, index=means.index)
    arr = corr.values.reshape(len(means), n, n)
    for t in range(len(means)):
        C = arr[t]
        if np.isnan(C).any():
            continue
        v = w @ np.clip(C, -1, 1) @ w
        if v > 0.01:
            out.iloc[t] = float(np.clip(1 / np.sqrt(v), 1.0, cap))
    return out


def s183_sleeves(c):
    ts = sum(_norm(S.ewmac(c, a, b)) for a, b in LP_SPEEDS) / len(LP_SPEEDS)
    ts = ts.clip(-20, 20)
    xs = _norm(S.xs_momentum(c, 256, within_group=False).fillna(0)
               .where(c.adj.notna()))
    trend = (0.5 * ts.fillna(0) + 0.5 * xs.fillna(0)).clip(-20, 20).where(c.adj.notna())
    # trends-research carry: (F1 - F2)/F1 annualised, unsmoothed, then normalised
    dt = (c.p.carry_ym - c.p.price_ym)
    raw_carry = ((c.p.price - c.p.carry) / c.p.price.abs() / dt.where(dt.abs() > 1e-6))
    carry = _norm(raw_carry.reindex(c.adj.index).ffill(limit=5))
    skew = _norm(-c.diff.rolling(256, min_periods=128).skew())
    vov = _norm(S.vol_of_vol(c))
    return {"trend": trend, "carry": carry, "skew": skew, "vov": vov}


def build_s183(c, overlays=True):
    sl = s183_sleeves(c)
    fdm = pooled_fdm_ewm(sl)
    blend = sum(0.25 * v.fillna(0) for v in sl.values()).mul(fdm, axis=0)
    master = _norm(blend.where(c.adj.notna()))
    fc = master.ewm(halflife=1, min_periods=1).mean().clip(-20, 20)
    if overlays:
        rv = c.adj.pct_change(fill_method=None).rolling(64).std() * 16
        vol_ratio = (rv / VOL_TARGET).fillna(1.0)
        dd = c.adj.pct_change(64, fill_method=None).shift(1).fillna(0.0)
        vol_gate = 1 - 0.5 * _sig(STEEP * (vol_ratio - 1.0))
        dd_gate = 1 - 0.5 * _sig(STEEP * (-VOL_TARGET / 2 - dd))
        fc = (fc * vol_gate * dd_gate).clip(-20, 20)
    # 5-year per-instrument warm-up before any position
    age = c.adj.notna().cumsum()
    return fc.where(age > OOS_START)


S183_ENGINE = dict(weights="equal", idm="table", start="1990-01-01")

register(Strategy(
    name="s183_replica", family="composite",
    description="trends-research S183 replica (58 futures, overlays, IDM table, from 1990)",
    build=build_s183, reference="Joly (2026) trends-research", universe=list(S183),
    engine=S183_ENGINE))
register(Strategy(
    name="s183_no_overlays", family="composite",
    description="S183 replica without the vol/drawdown sigmoid gates",
    build=lambda c: build_s183(c, overlays=False), reference="Joly (2026) trends-research",
    universe=list(S183), engine=S183_ENGINE))
