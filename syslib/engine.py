"""Portfolio backtest: Carver forecast-to-position with buffering and costs.

Everything is in fractions of a fixed notional capital (no compounding), so
daily portfolio returns are excess returns over cash: futures P&L carries no
funding.

Position (in capital-fraction per price unit) for instrument i at close t:
    p_it = w_it * IDM_t * (f_it / 10) * vol_target / (sigma_it * 16)
P&L on t+1 = p_it * (adj_i,t+1 - adj_i,t). Positions trade to the edge of a
+/-10% buffer. Costs per unit traded = spread + commission / pointsize (price
units); each roll trades the position twice.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .signals import Ctx, DAYS_YEAR, FORECAST_TARGET

VOL_TARGET = 0.20
IDM_CAP = 2.5

# Carver, Systematic Trading Table 7.8 (used by trends-research)
_IDM_N = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 70, 100, 200.])
_IDM_V = np.array([1.0, 1.2, 1.48, 1.56, 1.7, 1.9, 1.98, 2.11, 2.18, 2.24, 2.53, 2.87,
                   3.01, 3.12, 3.27, 3.39, 3.47, 3.54, 3.75, 4.0])


def instrument_weights(active, groups, scheme="group"):
    """Weights over active instruments; 'group' = equal per group, then equal within."""
    a = active.astype(float)
    if scheme == "equal":
        return a.div(a.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    g = groups.reindex(active.columns)
    w = pd.DataFrame(0.0, index=active.index, columns=active.columns)
    n_groups = pd.Series(0.0, index=active.index)
    per_group = {}
    for name, cols in g.groupby(g).groups.items():
        cols = list(cols)
        n_in = a[cols].sum(axis=1)
        per_group[name] = (cols, n_in)
        n_groups += (n_in > 0)
    for cols, n_in in per_group.values():
        w[cols] = a[cols].div(n_in.replace(0, np.nan), axis=0).fillna(0)
    return w.div(n_groups.replace(0, np.nan), axis=0).fillna(0)


def idm_from_corr(unit_rets, weights, window=2 * DAYS_YEAR, step=21, cap=IDM_CAP):
    """IDM = 1/sqrt(w'Cw) from rolling correlations of instrument sub-strategy
    returns, recomputed every `step` days and applied from the next day."""
    R = unit_rets.values
    W = weights.values
    T = len(unit_rets)
    idm = np.full(T, np.nan)
    last = 1.0
    for t in range(window, T, step):
        win = R[t - window:t]
        ok = np.isfinite(win).sum(axis=0) >= window // 2
        w = W[t - 1, ok]
        if ok.sum() >= 2 and w.sum() > 0:
            x = np.nan_to_num(win[:, ok])
            C = np.corrcoef(x, rowvar=False)
            C = np.nan_to_num(C, nan=0.0)
            C = np.clip(C, 0.0, 1.0)  # Carver: floor correlations at zero
            np.fill_diagonal(C, 1.0)
            w = w / w.sum()
            last = float(min(cap, 1.0 / np.sqrt(max(w @ C @ w, 1e-6))))
        idm[t:t + step] = last
    s = pd.Series(idm, index=unit_rets.index).shift(1)
    return s.fillna(1.0)


@dataclass
class Result:
    name: str
    returns: pd.Series            # daily portfolio return, net of costs
    gross: pd.Series              # before costs
    costs: pd.Series
    inst_returns: pd.DataFrame    # per-instrument net contribution
    positions: pd.DataFrame       # capital fraction per price unit
    forecasts: pd.DataFrame
    idm: pd.Series
    info: dict = field(default_factory=dict)


def backtest(ctx, forecasts, name="strategy", vol_target=VOL_TARGET, weights="group",
             idm="corr", buffer=0.10, cost_mult=1.0, start=None, warmup=DAYS_YEAR):
    """Run a forecast panel through the position/cost engine."""
    if not isinstance(ctx, Ctx):
        ctx = Ctx(ctx)
    p = ctx.p
    f = forecasts.reindex(index=ctx.adj.index, columns=ctx.adj.columns)
    sigma = ctx.sigma
    # need enough price history for a stable sigma
    hist = ctx.adj.notna().cumsum()
    active = f.notna() & sigma.notna() & (hist >= warmup) & ctx.adj.notna()
    if start is not None:
        active &= (f.index >= pd.Timestamp(start))[:, None]
    f = f.where(active, 0.0)

    w = instrument_weights(active, ctx.groups, weights)
    fwd = ctx.diff.shift(-1)                              # adj_{t+1} - adj_t
    unit = (f / FORECAST_TARGET) * (vol_target / np.sqrt(DAYS_YEAR)) * (fwd / sigma)
    unit = unit.where(active)
    if idm == "corr":
        idm_s = idm_from_corr(unit, w)
    elif idm == "table":
        idm_s = pd.Series(np.interp(active.sum(axis=1), _IDM_N, _IDM_V), index=f.index)
        idm_s[active.sum(axis=1) <= 1] = 1.0
    else:
        idm_s = pd.Series(float(idm), index=f.index)

    scale = w.mul(idm_s, axis=0) * vol_target / (sigma * np.sqrt(DAYS_YEAR))
    target = (scale * f / FORECAST_TARGET).fillna(0).values
    width = (buffer * scale).fillna(0).values
    pos = np.zeros_like(target)
    cur = np.zeros(target.shape[1])
    for t in range(len(target)):
        lo, hi = target[t] - width[t], target[t] + width[t]
        cur = np.where(cur < lo, lo, np.where(cur > hi, hi, cur))
        cur = np.where(active.values[t], cur, 0.0)
        pos[t] = cur
    pos = pd.DataFrame(pos, index=f.index, columns=f.columns)

    meta = p.meta.reindex(f.columns)
    unit_cost = (meta["spread"].fillna(0) + meta["commission"].fillna(0) / meta["pointsize"]) * cost_mult
    trades = pos.diff().abs().fillna(pos.abs())
    rolls = ctx.roll_days.reindex(index=pos.index, columns=pos.columns).fillna(False)
    roll_trades = 2 * pos.shift(1).abs().where(rolls, 0.0).fillna(0)
    cost = (trades + roll_trades).mul(unit_cost, axis=1)

    gross_i = (pos * fwd).fillna(0).shift(1)      # P&L realised on t+1, stamped t+1
    cost_i = cost                                 # paid on the trade date
    net_i = gross_i - cost_i
    first = active.any(axis=1).idxmax() if active.values.any() else f.index[0]
    sl = slice(first, None)
    gross = gross_i.sum(axis=1)[sl]
    costs = cost_i.sum(axis=1)[sl]
    res = Result(name=name, returns=(gross - costs), gross=gross, costs=costs,
                 inst_returns=net_i.loc[sl], positions=pos.loc[sl], forecasts=f.loc[sl],
                 idm=idm_s.loc[sl])
    # turnover in round-trips/yr relative to average absolute position
    avg_pos = pos.abs().mean()
    res.info["turnover"] = float((trades.sum() / avg_pos.replace(0, np.nan) / 2
                                   / (len(pos) / DAYS_YEAR)).median())
    res.info["n_instruments"] = int(active.any().sum())
    return res
