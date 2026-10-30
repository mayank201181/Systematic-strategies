"""Strategy registry.

Each Strategy turns a Ctx into a capped forecast panel; the engine turns the
panel into positions and P&L. Add a strategy by writing a builder and calling
register(). `added` is the date the strategy entered the library: P&L after
that date is out-of-sample and the scorecard reports it separately.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import signals as S
from .combine import combine
from .universe import BROAD, S183

LP_SPEEDS = [(16, 64), (32, 128), (64, 256)]   # Levine & Pedersen (2016)
BREAKOUTS = [20, 40, 80, 160, 320]


@dataclass
class Strategy:
    name: str
    family: str
    description: str
    build: Callable
    reference: str = ""
    universe: list = field(default_factory=lambda: list(BROAD))
    engine: dict = field(default_factory=dict)
    added: str = "2026-10-27"
    prior: str = "positive"         # positive | control (expected <= 0)


REGISTRY: dict[str, Strategy] = {}


def register(s):
    REGISTRY[s.name] = s
    return s


def sc(raw):
    return S.scale_forecast(raw)


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def b_ewmac(speeds):
    def f(c):
        return combine({f"{a}_{b}": sc(S.ewmac(c, a, b)) for a, b in speeds})
    return f


def b_breakout(c):
    return combine({f"bo{n}": sc(S.breakout(c, n)) for n in BREAKOUTS})


def b_tsmom(c):
    return sc(S.tsmom(c, 256))


def b_accel(c):
    return combine({f"acc{n}": sc(S.accel(c, n)) for n in (16, 32, 64)})


def b_xs_global(c):
    return sc(S.xs_momentum(c, 256, within_group=False))


def b_xs_group(c):
    return sc(S.xs_momentum(c, 256, within_group=True))


def b_relmom(c):
    return combine({f"rm{n}": sc(S.relative_momentum(c, n)) for n in (40, 80, 160)})


def b_assettrend(c):
    return combine({f"at{a}": sc(S.asset_trend(c, a, b)) for a, b in LP_SPEEDS})


def b_carry(c):
    return sc(S.carry(c, 90))


def b_relcarry(c):
    return sc(S.relative_carry(c, 90))


def b_skew(c):
    return sc(S.skew(c, 256))


def b_relskew(c):
    return sc(S.relative_skew(c, 256))


def b_vov(c):
    return sc(S.vol_of_vol(c))


def b_value(c):
    return sc(S.value(c))


def b_mr_fast(c):
    return sc(S.mean_reversion_fast(c, 5))


def b_mr_asset(c):
    return sc(S.mr_in_asset(c, 1000))


def b_trend_carry(c):
    return combine({"ewmac": b_ewmac(LP_SPEEDS)(c), "breakout": b_breakout(c),
                    "carry": b_carry(c)},
                   {"ewmac": 0.3, "breakout": 0.3, "carry": 0.4})


def b_multi(c):
    parts = {"ewmac": b_ewmac(LP_SPEEDS)(c), "breakout": b_breakout(c),
             "xs_group": b_xs_group(c), "carry": b_carry(c),
             "relcarry": b_relcarry(c), "skew": b_skew(c), "relmom": b_relmom(c)}
    w = {"ewmac": 0.2, "breakout": 0.15, "xs_group": 0.1, "carry": 0.25,
         "relcarry": 0.1, "skew": 0.1, "relmom": 0.1}
    return combine(parts, w)


def b_quad_broad(c):
    """S183's four sleeves, built the library way (pooled scalar, FDM, no overlays)."""
    trend = combine({"ts": b_ewmac(LP_SPEEDS)(c), "xs": b_xs_global(c)}, fdm=False)
    return combine({"trend": trend, "carry": b_carry(c), "skew": b_skew(c),
                    "vov": b_vov(c)})


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

for spec in [
    ("trend_ewmac", "trend", "EWMAC 16/64, 32/128, 64/256 equal-weight", b_ewmac(LP_SPEEDS),
     "Levine & Pedersen (2016); Carver (2015)"),
    ("trend_ewmac_fast", "trend", "EWMAC 4/16, 8/32 (fast trend)", b_ewmac([(4, 16), (8, 32)]),
     "Carver (2015)"),
    ("trend_breakout", "trend", "Carver breakout 20-320d", b_breakout, "Carver (2015)"),
    ("tsmom_12m", "trend", "12m vol-normalised time-series momentum", b_tsmom,
     "Moskowitz, Ooi & Pedersen (2012)"),
    ("trend_accel", "trend", "Trend acceleration (change in EWMAC)", b_accel, "Carver (2023)"),
    ("asset_trend", "trend", "Asset-class index trend applied to members", b_assettrend,
     "Carver (2023)"),
    ("xsmom_global", "momentum", "12m XS momentum z-scored across all futures", b_xs_global,
     "trends-research S183; Asness, Moskowitz & Pedersen (2013)"),
    ("xsmom_group", "momentum", "12m XS momentum within asset class", b_xs_group,
     "Asness, Moskowitz & Pedersen (2013)"),
    ("relmom_group", "momentum", "Relative momentum vs asset class, 40-160d", b_relmom,
     "Carver (2023)"),
    ("carry", "carry", "Annualised roll yield / vol, 90d smooth", b_carry,
     "Koijen, Moskowitz, Pedersen & Vrugt (2018)"),
    ("relcarry_group", "carry", "Carry relative to asset class", b_relcarry, "Carver (2023)"),
    ("skew", "skew", "Negative 256d return skew", b_skew, "Fernandez-Perez et al. (2018)"),
    ("relskew_group", "skew", "Skew relative to asset class", b_relskew, "Carver (2023)"),
    ("vov", "vol", "Low vol-of-vol x 64d direction", b_vov,
     "Baltussen, van Bekkum & van der Grient (2018); trends-research"),
    ("trend_carry", "composite", "30% EWMAC, 30% breakout, 40% carry", b_trend_carry,
     "Carver (2015)"),
    ("multi_7", "composite", "7 positive-prior sleeves, handcrafted weights", b_multi, ""),
    ("quad_broad", "composite", "S183's 4 sleeves on the broad universe, library engine",
     b_quad_broad, "trends-research S183"),
]:
    register(Strategy(name=spec[0], family=spec[1], description=spec[2], build=spec[3],
                      reference=spec[4]))

for spec in [
    ("value_5y", "value", "5y reversal vs asset class", b_value,
     "Asness, Moskowitz & Pedersen (2013)"),
    ("mr_fast", "mean_reversion", "5d mean reversion", b_mr_fast, ""),
    ("mr_in_asset", "mean_reversion", "Mean reversion of price vs asset class, 1000d",
     b_mr_asset, "Carver (2023)"),
]:
    register(Strategy(name=spec[0], family=spec[1], description=spec[2], build=spec[3],
                      reference=spec[4], prior="control"))


# --------------------------------------------------------------------------
# Variance risk premium: short front vol futures (VIX, V2X)
# --------------------------------------------------------------------------

VOL = ["VIX", "V2X"]


def b_vrp_short(c):
    """Always short, vol-targeted: the raw VRP / roll-down harvest."""
    return (c.adj * 0 - 10.0).where(c.adj.notna())


def b_vrp_contango(c):
    """Short scaled by roll-down; flat (never long) in backwardation."""
    return sc(S.carry(c, 20)).clip(upper=0)


def b_vrp_gated(c):
    """Short only in contango and while fast vol trend is not rising."""
    contango = S.carry(c, 5) < 0
    calm = S.ewmac(c, 8, 32) <= 0
    return (-10.0 * (contango & calm)).where(c.adj.notna())


for spec in [
    ("vrp_short", "Always short VIX/V2X front futures, vol-targeted", b_vrp_short,
     "Carr & Wu (2009); Eraker (2009)"),
    ("vrp_contango", "Short vol futures scaled by roll-down, flat in backwardation",
     b_vrp_contango, "Simon & Campasano (2014)"),
    ("vrp_gated", "Short vol futures only in contango and with 8/32 trend not rising",
     b_vrp_gated, "Simon & Campasano (2014)"),
]:
    register(Strategy(name=spec[0], family="vrp", description=spec[1], build=spec[2],
                      reference=spec[3], universe=list(VOL), added="2026-10-30"))


def b_long_only(c):
    return (c.adj * 0 + 10.0).where(c.adj.notna())


register(Strategy(name="bench_long_rp", family="benchmark",
                  description="Always-long, vol-targeted risk parity (beta benchmark)",
                  build=b_long_only, prior="benchmark"))
