# systematic-strategies

A library of systematic futures strategies on one common backtest harness, with
a scorecard that re-runs every weekday on fresh data.

Starting point: Lucas Joly's [trends-research](https://github.com/Lucas-Joly-GH/trends-research)
(Carver-style 4-sleeve composite "S183"). That repo is code-only: its Norgate
data is private. This library rebuilds S183 on Rob Carver's public
[pysystemtrade](https://github.com/pst-group/pysystemtrade) futures data and
reproduces the published result (see `results/README.md`).

## Layout

| path | what |
|---|---|
| `syslib/universe.py` | instrument groups (equity, bonds, STIR, G10/EM FX, energy, metals, ags, vol, crypto) and the S183 universe |
| `syslib/data.py` | builds daily panels from pysystemtrade CSVs (`data/pst/*.parquet`) |
| `syslib/signals.py` | forecast building blocks: EWMAC, breakout, TS/XS momentum, acceleration, carry, relative carry, skew, vol-of-vol, value, mean reversion |
| `syslib/combine.py` | forecast blending with a pooled FDM |
| `syslib/engine.py` | Carver forecast-to-position: 20% vol target, group weights, correlation-estimated IDM, 10% buffer, spread + commission + roll costs |
| `syslib/strategies.py` | the registry; one entry per strategy |
| `syslib/s183.py` | trends-research S183 replica |
| `syslib/metrics.py` | Sharpe, HAC t-stat, deflated Sharpe, multiple-testing haircut, drawdowns |
| `syslib/scorecard.py` | runs the registry, writes `results/` |
| `syslib/live.py` | extends data to today from Yahoo Finance (runs in Actions) |
| `syslib/report.py` | renders `results/README.md` and charts |

## Run locally

    pip install -r requirements.txt
    python -m syslib.scorecard --data data/pst --out results   # ~3 min
    python -m syslib.report --results results

## Add a strategy

Write a builder `f(ctx) -> forecast panel` in `syslib/strategies.py` (raw
signal through `scale_forecast`, or several through `combine`) and
`register(Strategy(...))` it. Set `added` to today's date: everything after
it is reported as live, out-of-sample P&L. The next scheduled run scores it.

## Conventions and known limits

- Returns are excess returns in % of fixed capital (no compounding), net of
  costs. Positions are fractional (no contract rounding), i.e. large capital.
- Signals at close t trade into P&L from t to t+1.
- pysystemtrade's CSVs end 2024-03-28. After that, US-listed futures are
  extended with Yahoo continuous contracts (roll gaps over 6 sigma are zeroed
  and logged in `data/live/roll_filter_log.csv`). Carry after that date comes
  from snapshots of individual contracts in `data/live/curve.csv`. Non-US
  futures (Bund, JGB, KOSPI, DAX...) have no free tail and drop out after
  2024-03 until a paid feed (Norgate, Databento, IBKR) is wired in.
