"""Run every registered strategy and write the scorecard.

    python -m syslib.scorecard --data data/pst --out results

Outputs in --out:
  scorecard.csv      one row per strategy: full-period, sub-period and
                     post-registration (live) statistics
  sectors.csv        Sharpe of each strategy's P&L by sector
  correlations.csv   correlation of daily strategy returns
  returns.parquet    daily net returns per strategy
  history.csv        appended each run: date, strategy, sharpe, live sharpe
"""
import argparse
import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import s183  # noqa: F401  registers the replica
from .data import load_panel
from .engine import backtest
from .metrics import deflated_sharpe, haircut_sharpe, sharpe, summary
from .signals import Ctx
from .strategies import REGISTRY
from .universe import GROUP_OF, SECTORS

warnings.filterwarnings("ignore")

EVAL_START = "1990-01-01"
PERIODS = {"90s": ("1990", "1999"), "00s": ("2000", "2009"), "10s": ("2010", "2019"),
           "20s": ("2020", None), "last5y": None, "post_pst": ("2024-04-01", None)}


def run_all(data_dir, names=None, cost_mult=1.0):
    panel = load_panel(data_dir)
    ctx_cache = {}
    results = {}
    for name, s in REGISTRY.items():
        if names and name not in names:
            continue
        key = tuple(sorted(s.universe))
        if key not in ctx_cache:
            ctx_cache[key] = Ctx(panel.subset(list(key)))
        c = ctx_cache[key]
        fc = s.build(c)
        kw = dict(s.engine)
        kw.setdefault("cost_mult", cost_mult)
        results[name] = backtest(c, fc, name, **kw)
        print(f"  ran {name}")
    return results


def sector_of(inst):
    g = GROUP_OF.get(inst)
    for sec, groups in SECTORS.items():
        if g in groups:
            return sec
    return "Other"


def build_scorecard(results, n_trials=None):
    rets = pd.DataFrame({k: r.returns for k, r in results.items()}).loc[EVAL_START:]
    end = rets.index.max()
    periods = dict(PERIODS)
    periods["last5y"] = (str((end - pd.DateOffset(years=5)).date()), None)
    n_trials = n_trials or len(results)
    # dispersion of Sharpe ratios across the positive-prior trials, for the DSR
    srs = [sharpe(rets[k]) for k in rets if REGISTRY[k].prior == "positive"]
    sr_std = float(np.nanstd(srs)) / np.sqrt(256) if len(srs) > 2 else None
    rows, sec_rows = [], []
    bench = rets.get("bench_long_rp")
    trend = rets.get("trend_ewmac")
    for name, r in results.items():
        s = REGISTRY[name]
        x = r.returns.loc[EVAL_START:]
        row = {"strategy": name, "family": s.family, "prior": s.prior,
               "description": s.description, "added": s.added}
        row.update(summary(x, r.costs, periods))
        row["deflated_sr_prob"] = deflated_sharpe(x, n_trials, sr_std)
        row["sr_haircut"] = haircut_sharpe(row["sharpe"], row["t_hac"], n_trials)
        row["turnover"] = r.info.get("turnover")
        row["n_instruments"] = r.info.get("n_instruments")
        if bench is not None:
            row["corr_long_rp"] = x.corr(bench)
        if trend is not None:
            row["corr_trend"] = x.corr(trend)
        live = r.returns.loc[s.added:]
        row["live_days"] = int((live != 0).sum())
        row["live_sr"] = sharpe(live) if row["live_days"] > 60 else np.nan
        row["live_ret"] = float(live.sum()) if len(live) else np.nan
        rows.append(row)
        by_sec = r.inst_returns.loc[EVAL_START:].T.groupby(sector_of).sum().T
        sec_rows.append({"strategy": name,
                         **{sec: sharpe(by_sec[sec]) for sec in by_sec.columns}})
    card = pd.DataFrame(rows).set_index("strategy")
    card = card.sort_values("sharpe", ascending=False)
    sectors = pd.DataFrame(sec_rows).set_index("strategy").reindex(card.index)
    corr = rets.corr().reindex(index=card.index, columns=card.index)
    return card, sectors, corr, rets


def write(card, sectors, corr, rets, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    card.to_csv(out / "scorecard.csv", float_format="%.4f")
    sectors.to_csv(out / "sectors.csv", float_format="%.3f")
    corr.to_csv(out / "correlations.csv", float_format="%.3f")
    rets.to_parquet(out / "returns.parquet")
    hist = card[["sharpe", "live_sr", "live_days", "end"]].copy()
    hist.insert(0, "run_date", dt.date.today().isoformat())
    hp = out / "history.csv"
    hist.to_csv(hp, mode="a", header=not hp.exists(), float_format="%.4f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pst")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--cost-mult", type=float, default=1.0)
    a = ap.parse_args()
    res = run_all(a.data, a.only, a.cost_mult)
    card, sectors, corr, rets = build_scorecard(res)
    write(card, sectors, corr, rets, a.out)
    cols = ["family", "sharpe", "sr_haircut", "t_hac", "ann_vol", "max_dd", "sr_10s", "sr_20s", "sr_last5y",
            "cost_sr", "corr_trend", "corr_long_rp", "deflated_sr_prob"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(card[[c for c in cols if c in card]].round(2).to_string())
        print(sectors.round(2).to_string())


if __name__ == "__main__":
    main()
