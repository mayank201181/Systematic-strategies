"""Render results/ into a markdown report with charts.

    python -m syslib.report --results results
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY = ["multi_7", "trend_carry", "s183_replica", "trend_ewmac", "carry", "xsmom_global",
       "skew", "bench_long_rp"]


def _pct(x):
    return "" if pd.isna(x) else f"{x:.0%}"


def _f(x, d=2):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def charts(res):
    rets = pd.read_parquet(res / "returns.parquet")
    keys = [k for k in KEY if k in rets]
    fig, ax = plt.subplots(figsize=(11, 5))
    (rets[keys].fillna(0).cumsum() * 100).plot(ax=ax, lw=1.2)
    ax.set_ylabel("cumulative return, % of capital (no compounding)")
    ax.set_title("Selected strategies, net of costs, 20% vol target")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(res / "equity.png", dpi=110)
    plt.close(fig)

    roll = rets[keys].rolling(3 * 256, min_periods=500)
    sr = roll.mean() / roll.std() * 16
    fig, ax = plt.subplots(figsize=(11, 4))
    sr.plot(ax=ax, lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Rolling 3-year Sharpe")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(res / "rolling_sharpe.png", dpi=110)
    plt.close(fig)

    corr = pd.read_csv(res / "correlations.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Correlation of daily returns, 1990-")
    fig.tight_layout()
    fig.savefig(res / "correlations.png", dpi=110)
    plt.close(fig)


def markdown(res):
    card = pd.read_csv(res / "scorecard.csv", index_col=0)
    sec = pd.read_csv(res / "sectors.csv", index_col=0)
    lines = [
        "# Strategy scorecard",
        "",
        f"Data {card['start'].min()} to {card['end'].max()}. Statistics from 1990, net of "
        "spread, commission and roll costs, 20% annual vol target, fixed capital. "
        "`SR hc` is the Sharpe after a Bonferroni multiple-testing haircut over all "
        "strategies in the table; `DSR` is the deflated-Sharpe probability. "
        "`live` columns cover P&L after the date a strategy entered the library.",
        "",
        "| strategy | family | SR | SR hc | t(HAC) | vol | maxDD | 90s | 00s | 10s | 20s | last 5y "
        "| post-Mar24 | cost SR | corr trend | corr long | DSR | live SR | live days |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n, r in card.iterrows():
        lines.append(
            f"| {n} | {r.family} | {_f(r.sharpe)} | {_f(r.sr_haircut)} | {_f(r.t_hac, 1)} | "
            f"{_pct(r.ann_vol)} | {_pct(r.max_dd)} | {_f(r.sr_90s)} | {_f(r.sr_00s)} | "
            f"{_f(r.sr_10s)} | {_f(r.sr_20s)} | {_f(r.sr_last5y)} | {_f(r.get('sr_post_pst'))} | {_f(r.cost_sr)} | "
            f"{_f(r.corr_trend)} | {_f(r.corr_long_rp)} | {_f(r.deflated_sr_prob)} | "
            f"{_f(r.live_sr)} | {int(r.live_days)} |")
    lines += ["", "## Sharpe by sector (each strategy's P&L split by where it was earned)", "",
              "| strategy | " + " | ".join(sec.columns) + " |",
              "|---|" + "---|" * len(sec.columns)]
    for n, r in sec.iterrows():
        lines.append(f"| {n} | " + " | ".join(_f(v) for v in r.values) + " |")
    lines += ["", "![equity](equity.png)", "", "![rolling](rolling_sharpe.png)", "",
              "![corr](correlations.png)", ""]
    (res / "README.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    res = Path(a.results)
    charts(res)
    markdown(res)


if __name__ == "__main__":
    main()
