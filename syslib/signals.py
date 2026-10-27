"""Forecast building blocks.

Every public function takes a Ctx and returns a raw forecast panel
(dates x instruments). Raw forecasts are put on Carver's scale (mean
|forecast| = 10, capped at +/-20) by scale_forecast(), which uses only
data available at each date.
"""
from functools import cached_property

import numpy as np
import pandas as pd

FORECAST_TARGET = 10.0
FORECAST_CAP = 20.0
DAYS_YEAR = 256


class Ctx:
    """Cached derived series shared by all signals for one Panel."""

    def __init__(self, panel):
        self.p = panel

    @cached_property
    def adj(self):
        a = self.p.adj
        # forward-fill holiday gaps inside each instrument's life only
        return a.ffill(limit=10).where(a.bfill().notna())

    @cached_property
    def diff(self):
        return self.adj.diff()

    @cached_property
    def sigma(self):
        """Daily price-unit volatility: 70% EWMA(32) + 30% 10y rolling std."""
        d = self.diff
        short = d.ewm(span=32, min_periods=32).std()
        long = d.rolling(2560, min_periods=256).std()
        s = 0.7 * short + 0.3 * long.fillna(short)
        # Carver-style floor: never below the 5th percentile of the last 10y,
        # which stops STIR/pegged FX positions exploding in dead-calm regimes
        floor = s.rolling(2560, min_periods=256).quantile(0.05)
        s = np.maximum(s, floor.fillna(0))
        return s.where(s > 0)

    @cached_property
    def norm_ret(self):
        return self.diff / self.sigma.shift(1)

    @cached_property
    def norm_price(self):
        return self.norm_ret.fillna(0).cumsum().where(self.adj.notna())

    @cached_property
    def groups(self):
        return self.p.groups.reindex(self.adj.columns)

    @cached_property
    def raw_carry(self):
        """Annualised roll yield in price units per year, / annual price vol."""
        dt = self.p.carry_ym - self.p.price_ym
        dt = dt.where(dt.abs() > 1e-6)
        rc = (self.p.price - self.p.carry) / dt
        rc = rc.reindex(self.adj.index).ffill(limit=5)
        return rc / (self.sigma * np.sqrt(DAYS_YEAR))

    @cached_property
    def roll_days(self):
        ym = self.p.price_ym.reindex(self.adj.index).ffill()
        return (ym.diff().fillna(0) != 0)

    def group_demean(self, x):
        """Subtract each date's group mean (cross-sectional within group)."""
        out = x.copy()
        for g, cols in self.groups.groupby(self.groups).groups.items():
            cols = list(cols)
            if len(cols) < 2:
                out[cols] = np.nan
                continue
            out[cols] = x[cols].sub(x[cols].mean(axis=1), axis=0)
        return out

    def group_mean(self, x):
        out = x.copy()
        for g, cols in self.groups.groupby(self.groups).groups.items():
            cols = list(cols)
            out[cols] = np.repeat(x[cols].mean(axis=1).values[:, None], len(cols), axis=1)
        return out.where(x.notna())


# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------

def scale_forecast(raw, mode="pooled", min_periods=DAYS_YEAR, window=DAYS_YEAR):
    """Scale raw forecasts so mean |f| = 10, then cap at +/-20.

    pooled  : one scalar per date from the expanding cross-instrument mean of |raw|
              (Carver's default; keeps relative signal strength over time)
    rolling : per-instrument rolling mean |raw| over `window` days
              (trends-research production normaliser)
    """
    raw = raw.replace([np.inf, -np.inf], np.nan)
    a = raw.abs()
    if mode == "pooled":
        xs = a.mean(axis=1)
        cnt = a.notna().any(axis=1).cumsum()
        avg = xs.expanding().mean().where(cnt >= min_periods)
        f = raw.mul(FORECAST_TARGET / avg, axis=0)
    elif mode == "rolling":
        avg = a.rolling(window, min_periods=window).mean()
        f = raw * (FORECAST_TARGET / avg)
    else:
        raise ValueError(mode)
    return f.clip(-FORECAST_CAP, FORECAST_CAP)


# --------------------------------------------------------------------------
# Trend / momentum
# --------------------------------------------------------------------------

def ewmac(c, fast, slow):
    a = c.adj
    return (a.ewm(span=fast, min_periods=slow).mean()
            - a.ewm(span=slow, min_periods=slow).mean()) / c.sigma


def breakout(c, n):
    a = c.adj
    hi = a.rolling(n, min_periods=n // 2).max()
    lo = a.rolling(n, min_periods=n // 2).min()
    x = (a - (hi + lo) / 2) / (hi - lo).where(hi > lo)
    return x.ewm(span=max(n // 4, 2), min_periods=1).mean()


def tsmom(c, n=DAYS_YEAR, skip=0):
    """Vol-normalised n-day return, optionally skipping the last `skip` days."""
    npx = c.norm_price
    return (npx.shift(skip) - npx.shift(n)) / np.sqrt(n - skip)


def accel(c, n):
    e = ewmac(c, n, 4 * n)
    return e - e.shift(n)


def xs_momentum(c, n=DAYS_YEAR, within_group=False):
    x = c.norm_price.diff(n)
    if within_group:
        return c.group_demean(x)
    mu, sd = x.mean(axis=1), x.std(axis=1)
    z = x.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)
    return z.where(x.notna().sum(axis=1).ge(3), axis=0)


def relative_momentum(c, n=40):
    rel = c.norm_price - c.group_mean(c.norm_price)
    return (rel.diff(n) / n).ewm(span=max(n // 4, 2), min_periods=1).mean()


def asset_trend(c, fast=16, slow=64):
    gp = c.group_mean(c.norm_price)
    return gp.ewm(span=fast, min_periods=slow).mean() - gp.ewm(span=slow, min_periods=slow).mean()


# --------------------------------------------------------------------------
# Carry
# --------------------------------------------------------------------------

def carry(c, span=90):
    return c.raw_carry.ewm(span=span, min_periods=1).mean()


def relative_carry(c, span=90):
    return c.group_demean(carry(c, span))


# --------------------------------------------------------------------------
# Higher moments
# --------------------------------------------------------------------------

def skew(c, window=DAYS_YEAR):
    return -c.norm_ret.rolling(window, min_periods=window // 2).skew()


def relative_skew(c, window=DAYS_YEAR):
    return c.group_demean(skew(c, window))


def vol_of_vol(c, inner=21, window=64, avg=256, direction_lb=64, use_direction=True):
    """trends-research VoV: low vol-of-vol -> follow 64d direction (Baltussen et al.)."""
    d = c.diff
    dv = d.rolling(inner, min_periods=10).std()
    vov = dv.rolling(window, min_periods=min(window, max(inner, window // 3))).std()
    vavg = vov.rolling(avg, min_periods=window).mean()
    sig = -(vov / vavg - 1.0)
    if use_direction:
        sig = sig * np.sign(c.adj.diff(direction_lb)).replace(0, 1)
    return sig.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------
# Value / mean reversion (expected weak or negative; kept as controls)
# --------------------------------------------------------------------------

def value(c, n=5 * DAYS_YEAR, skip=DAYS_YEAR):
    """AMP-2013-style futures value: minus the 5y vol-normalised return, vs group."""
    npx = c.norm_price
    x = -(npx.shift(skip) - npx.shift(n)) / np.sqrt(n - skip)
    return c.group_demean(x)


def mean_reversion_fast(c, span=5):
    return -(c.adj - c.adj.ewm(span=span, min_periods=span).mean()) / c.sigma


def mr_in_asset(c, span=1000):
    rel = c.norm_price - c.group_mean(c.norm_price)
    return -(rel - rel.ewm(span=span, min_periods=DAYS_YEAR).mean())
