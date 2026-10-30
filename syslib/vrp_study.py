"""Variance-risk-premium study: VRP sleeves against the trend book.

    python -m syslib.vrp_study --data data/pst --out results/vrp

Writes README.md with stand-alone stats, cost sensitivity, crisis-window P&L
and the effect of blending each VRP sleeve into multi_7 at fixed risk weights.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import drawdown, sharpe
from .data import load_panel
from .engine import backtest
from .scorecard import run_all
from .signals import Ctx
from .strategies import REGISTRY

VRP = ["vrp_short", "vrp_contango", "vrp_gated"]
VOL = ["VIX", "V2X"]
BOOKS = ["multi_7", "trend_carry"]
CRISES = {
    "Lehman (Sep-Nov 08)": ("2008-09-01", "2008-11-30"),
    "Flash crash (May 10)": ("2010-05-01", "2010-06-30"),
    "US downgrade (Aug 11)": ("2011-07-25", "2011-10-15"),
    "China deval (Aug 15)": ("2015-08-10", "2015-09-30"),
    "Volmageddon (Feb 18)": ("2018-01-26", "2018-02-28"),
    "Covid (Feb-Mar 20)": ("2020-02-19", "2020-03-31"),
}
WEIGHTS = [0.0, 0.1, 0.2, 0.3]


def stats(r):
    r = r.dropna()
    m = r.resample("ME").sum()
    return {"SR": sharpe(r), "vol": r.std() * 16, "maxDD": drawdown(r).min(),
            "worst month": m.min(), "skew (m)": m.skew()}


def blend(book, sleeve, w):
    """Risk-weighted blend, rescaled to the book's own vol over the overlap."""
    x = pd.concat([book, sleeve], axis=1).dropna()
    a, b = x.iloc[:, 0], x.iloc[:, 1]
    mix = (1 - w) * a / a.std() + w * b / b.std()
    return mix * a.std() / mix.std()


def fmt(df, pct):
    out = df.copy()
    for c in out.columns:
        if c in pct:
            out[c] = out[c].map(lambda v: f"{v:.0%}" if pd.notna(v) else "")
        else:
            out[c] = out[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    return out.to_markdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pst")
    ap.add_argument("--out", default="results/vrp")
    a = ap.parse_args()
    names = VRP + BOOKS
    shown = names + [f"{k} ex-vol" for k in BOOKS]
    base = run_all(a.data, names)
    hi = run_all(a.data, VRP, cost_mult=3.0)
    rets = pd.DataFrame({k: r.returns for k, r in base.items()})
    # the books already trade VIX/V2X through carry and trend: rerun without them
    panel = load_panel(a.data)
    for k in BOOKS:
        s = REGISTRY[k]
        c = Ctx(panel.subset([i for i in s.universe if i not in VOL]))
        rets[f"{k} ex-vol"] = backtest(c, s.build(c), k).returns
    vol_share = {k: base[k].inst_returns[VOL].sum(axis=1) for k in BOOKS}
    start = rets[VRP].fillna(0).ne(0).idxmax().max()
    rets = rets.loc[start:]

    pct = {"vol", "maxDD", "worst month"}
    alone = pd.DataFrame({k: stats(rets[k]) for k in shown}).T
    alone["SR costs x3"] = [sharpe(hi[k].returns.loc[start:]) if k in hi else np.nan
                            for k in alone.index]
    alone["corr multi_7"] = rets.corr()["multi_7"].reindex(alone.index)

    crisis = pd.DataFrame({lab: rets.loc[s:e].sum() for lab, (s, e) in CRISES.items()}).T
    crisis = crisis[shown]

    rows = []
    for sleeve in VRP:
        for w in WEIGHTS:
            st = stats(blend(rets["multi_7"], rets[sleeve], w))
            rows.append({"sleeve": sleeve, "risk weight": w, **st})
    bl = pd.DataFrame(rows).set_index(["sleeve", "risk weight"])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rets.to_parquet(out / "returns.parquet")
    lines = [
        "# Variance risk premium sleeves",
        "",
        f"Sample {rets.index[0].date()} to {rets.index[-1].date()} (VIX futures from 2006, "
        "V2X from 2013; no free vol-futures tail after 2024-03). Library engine: 20% vol "
        "target, net of spread, commission and two-sided roll costs.",
        "",
        "The books already hold this premium: " + "; ".join(
            f"{k} is short VIX {(base[k].positions['VIX'].loc[start:] < 0).mean():.0%} of days and "
            f"earns {vol_share[k].loc[start:].sum() / rets[k].sum():.0%} of its P&L from VIX/V2X "
            f"(corr with vrp_contango {vol_share[k].loc[start:].corr(rets['vrp_contango']):.2f})"
            for k in BOOKS) + ".",
        "",
        "## Stand-alone",
        "",
        fmt(alone, pct),
        "",
        "## Crisis windows (cumulative return, % of capital)",
        "",
        crisis.map(lambda v: f"{v:+.0%}").to_markdown(),
        "",
        "## Blending a VRP sleeve into multi_7 (risk weight, rescaled to multi_7 vol)",
        "",
        fmt(bl, pct),
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
