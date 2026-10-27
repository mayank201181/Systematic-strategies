"""Forecast combination with a forecast diversification multiplier (FDM)."""
import numpy as np
import pandas as pd

from .signals import FORECAST_CAP, DAYS_YEAR

FDM_CAP = 2.5


def pooled_fdm(forecasts, weights, step=DAYS_YEAR, min_obs=DAYS_YEAR, cap=FDM_CAP):
    """FDM = 1/sqrt(w'Cw), C = correlation of the component forecasts pooled
    across instruments, on an expanding window, re-estimated every `step` days."""
    names = list(forecasts)
    w = np.array([weights[n] for n in names], dtype=float)
    w = w / w.sum()
    idx = forecasts[names[0]].index
    stacked = [forecasts[n].reindex(idx) for n in names]
    out = pd.Series(np.nan, index=idx)
    last = 1.0
    for t in range(min_obs, len(idx) + step, step):
        t = min(t, len(idx))
        cols = [s.iloc[:t].stack(future_stack=True) for s in stacked]
        X = pd.concat(cols, axis=1).dropna()
        if len(X) > min_obs and X.std().gt(0).all():
            C = np.clip(np.nan_to_num(np.corrcoef(X.values, rowvar=False)), 0, 1)
            np.fill_diagonal(C, 1.0)
            last = float(min(cap, 1 / np.sqrt(w @ C @ w)))
        out.iloc[t:t + step] = last
    return out.fillna(1.0)


def combine(forecasts, weights=None, fdm=True):
    """Weighted blend of scaled forecasts, times FDM, capped at +/-20."""
    names = list(forecasts)
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}
    tot = sum(weights.values())
    blend = sum(forecasts[n].fillna(0) * (weights[n] / tot) for n in names)
    any_valid = pd.concat([forecasts[n].notna().stack(future_stack=True) for n in names], axis=1).any(axis=1)
    any_valid = any_valid.unstack()
    blend = blend.where(any_valid.reindex_like(blend).fillna(False))
    if fdm and len(names) > 1:
        m = pooled_fdm(forecasts, weights)
        blend = blend.mul(m, axis=0)
    return blend.clip(-FORECAST_CAP, FORECAST_CAP)
