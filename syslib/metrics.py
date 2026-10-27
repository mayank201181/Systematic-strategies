"""Performance statistics on daily excess-return series (fraction of capital)."""
import numpy as np
import pandas as pd
from scipy import stats

DAYS = 256


def _clean(r):
    r = r.dropna()
    nz = r.ne(0).values
    if nz.any():
        r = r.iloc[int(np.argmax(nz)):]
    return r


def sharpe(r):
    r = _clean(r)
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(DAYS)) if sd > 0 else np.nan


def hac_tstat(r, lags=10):
    """Newey-West t-stat of the mean daily return."""
    r = _clean(r).values
    n = len(r)
    if n < 50:
        return np.nan
    e = r - r.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(r.mean() / np.sqrt(s / n))


def deflated_sharpe(r, n_trials, sr_trials_std=None):
    """Bailey & Lopez de Prado deflated Sharpe probability (daily SR inputs)."""
    r = _clean(r)
    n = len(r)
    sr = r.mean() / r.std()
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    if sr_trials_std is None:
        sr_trials_std = 0.5 / np.sqrt(DAYS)   # assume annual SR dispersion 0.5 across trials
    emc = 0.5772156649
    n_trials = max(int(n_trials), 2)
    sr0 = sr_trials_std * ((1 - emc) * stats.norm.ppf(1 - 1 / n_trials)
                           + emc * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(n - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf(z))


def drawdown(r):
    eq = _clean(r).cumsum()
    return eq - eq.cummax()


def summary(r, costs=None, periods=None):
    r = _clean(r)
    if len(r) < 60:
        return {}
    ann_ret = r.mean() * DAYS
    ann_vol = r.std() * np.sqrt(DAYS)
    dd = drawdown(r)
    m = r.resample("ME").sum()
    down = r[r < 0].std() * np.sqrt(DAYS)
    out = {
        "start": r.index[0].date().isoformat(),
        "end": r.index[-1].date().isoformat(),
        "years": round(len(r) / DAYS, 1),
        "sharpe": sharpe(r),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sortino": ann_ret / down if down > 0 else np.nan,
        "max_dd": float(dd.min()),
        "calmar": ann_ret / -dd.min() if dd.min() < 0 else np.nan,
        "skew_daily": float(stats.skew(r)),
        "skew_monthly": float(stats.skew(m)),
        "worst_month": float(m.min()),
        "pct_pos_months": float((m > 0).mean()),
        "t_hac": hac_tstat(r),
    }
    if costs is not None:
        c = costs.reindex(r.index).fillna(0)
        out["cost_sr"] = float(c.mean() * DAYS / ann_vol) if ann_vol > 0 else np.nan
    for label, (a, b) in (periods or {}).items():
        out[f"sr_{label}"] = sharpe(r.loc[a:b]) if len(r.loc[a:b]) > 120 else np.nan
    return out


def haircut_sharpe(sr, t, n_trials):
    """Bonferroni multiple-testing haircut (Harvey & Liu 2015): scale SR by the
    ratio of the adjusted to the raw t-stat."""
    if not np.isfinite(t) or t <= 0:
        return sr
    p = 2 * (1 - stats.norm.cdf(t))
    p_adj = min(1.0, p * max(n_trials, 1))
    t_adj = stats.norm.ppf(1 - p_adj / 2) if p_adj < 1 else 0.0
    return float(sr * t_adj / t)
