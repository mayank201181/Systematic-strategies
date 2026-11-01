# Kronos zero-shot test on our futures universe (1 Nov 2026)

**Verdict: no edge. Do not add.** Kronos-small, run zero-shot out of sample, is a
noisy anti-trend signal. It loses to a random walk on price error and its
"confidence" filter works backwards. What P&L it had came from being short
trend in a period when trend lost money.

## Setup
- Model: `NeoQuasar/Kronos-small` (24.7M) + `Kronos-Tokenizer-base`, upstream
  repo pinned at `67b630e`. The paper says pre-training data ends June 2024.
- Data: Yahoo continuous daily OHLCV for 47 US-listed futures (rates, FX,
  equity, energy, metals, ags, livestock, crypto), roll gaps removed (`prep.py`).
- Test: every trading day from 1 Jul 2024 to 30 Oct 2026 (588 days, 27.6k
  forecasts). Context 96 daily bars, 8 sampled paths, 5-day horizon
  (`forecast.py`; runs on GitHub Actions, 20 jobs, read-only token).
- Scoring (`evaluate.py`): vol-normalised forecasts, equal-risk portfolio,
  next-day P&L, costs of 0, 1 and 2% of daily vol per unit of risk traded.
  The baselines (EWMAC, 5-day reversal, long-only) use the same series.
  Jul-24 to Jun-25 is the paper's own test window. Jul-25 to Oct-26 is the
  clean holdout.

## Results
| | Gross SR | Net SR (2% vol cost) | Jul24–Jun25 net | Jul25–Oct26 net | Turnover/yr |
|---|---|---|---|---|---|
| Kronos 1-day | 0.50 | −0.52 | 0.22 | −1.02 | 213x |
| Kronos 5-day | 0.24 | −0.17 | 0.31 | −0.48 | 101x |
| Kronos 5-day smoothed | 0.39 | 0.23 | 0.71 | −0.07 | 40x |
| Kronos, tight paths only ("confident") | −0.24 | −1.25 | −0.60 | −1.76 | 143x |
| EWMAC 16/64 | −0.26 | −0.32 | −0.75 | −0.05 | 16x |
| 5-day reversal | 0.13 | −0.39 | 0.14 | −0.76 | 141x |
| Long-only | 0.89 | 0.89 | 0.88 | 0.87 | 0 |

- **Forecast skill.** The 1-day price error is 0.94 daily vols, against 0.79
  for a random walk, so Kronos is 19% worse. Hit rate is 49.9% at 1 day and
  51.0% at 5 days. The cross-sectional rank IC is 0.01 (t 0.7 to 1.3).
- **What it really is.** The Kronos 5-day forecast correlates −0.72 with
  EWMAC 8/32 and +0.44 with 5-day reversal. Its P&L correlates −0.55 to −0.68
  with our trend_ewmac, trend_carry and multi_7. After regressing on EWMAC
  and reversal, the residual alpha has SR −0.1 to −0.8 and is never
  significant. Simply being short EWMAC 8/32 did better, at about +0.47.
- **Viral claims.** "Tight spread means confident": the low-dispersion
  tercile is the worst bucket, with net SR −1.25. Sample dispersion is also a
  worse 5-day volatility forecast than a plain EWMA (rank corr 0.12 against
  0.18).

## Caveats
Only the small model was tested, at one context length (96 bars). Daily bars
only, no fine-tuning. The cost model is generic. Yahoo continuous series keep
some small roll gaps, but every signal uses the same series.
