# Variance risk premium sleeves

Sample 2007-03-06 to 2024-03-28 (VIX futures from 2006, V2X from 2013; no free vol-futures tail after 2024-03). Library engine: 20% vol target, net of spread, commission and two-sided roll costs.

The books already hold this premium: multi_7 is short VIX 87% of days and earns 23% of its P&L from VIX/V2X (corr with vrp_contango 0.87); trend_carry is short VIX 84% of days and earns 22% of its P&L from VIX/V2X (corr with vrp_contango 0.81).

## Stand-alone

|                    |   SR | vol   | maxDD   | worst month   |   skew (m) |   SR costs x3 |   corr multi_7 |
|:-------------------|-----:|:------|:--------|:--------------|-----------:|--------------:|---------------:|
| vrp_short          | 0.25 | 20%   | -94%    | -40%          |      -1.77 |          0.05 |           0.32 |
| vrp_contango       | 0.55 | 16%   | -54%    | -21%          |       0.44 |          0.28 |           0.51 |
| vrp_gated          | 0.34 | 13%   | -39%    | -13%          |       0.22 |          0.01 |           0.45 |
| multi_7            | 0.97 | 13%   | -23%    | -10%          |       0.16 |               |           1    |
| trend_carry        | 0.96 | 16%   | -26%    | -13%          |       0.38 |               |           0.97 |
| multi_7 ex-vol     | 0.93 | 12%   | -20%    | -8%           |       0.26 |               |           0.9  |
| trend_carry ex-vol | 0.92 | 15%   | -25%    | -10%          |       0.31 |               |           0.85 |

## Crisis windows (cumulative return, % of capital)

|                       | vrp_short   | vrp_contango   | vrp_gated   | multi_7   | trend_carry   | multi_7 ex-vol   | trend_carry ex-vol   |
|:----------------------|:------------|:---------------|:------------|:----------|:--------------|:-----------------|:---------------------|
| Lehman (Sep-Nov 08)   | -45%        | -6%            | -7%         | +24%      | +38%          | +18%             | +27%                 |
| Flash crash (May 10)  | -19%        | -22%           | -6%         | -4%       | -3%           | +3%              | +5%                  |
| US downgrade (Aug 11) | -18%        | -12%           | -9%         | -10%      | -8%           | -3%              | +1%                  |
| China deval (Aug 15)  | -19%        | -8%            | -9%         | -8%       | -7%           | -5%              | -4%                  |
| Volmageddon (Feb 18)  | -15%        | -10%           | -11%        | -13%      | -16%          | -7%              | -9%                  |
| Covid (Feb-Mar 20)    | -57%        | -14%           | -10%        | -10%      | -4%           | -2%              | +3%                  |

## Blending a VRP sleeve into multi_7 (risk weight, rescaled to multi_7 vol)

|                       |   SR | vol   | maxDD   | worst month   |   skew (m) |
|:----------------------|-----:|:------|:--------|:--------------|-----------:|
| ('vrp_short', 0.0)    | 0.97 | 13%   | -23%    | -10%          |       0.16 |
| ('vrp_short', 0.1)    | 0.95 | 13%   | -23%    | -10%          |      -0    |
| ('vrp_short', 0.2)    | 0.93 | 13%   | -27%    | -11%          |      -0.2  |
| ('vrp_short', 0.3)    | 0.89 | 13%   | -30%    | -13%          |      -0.45 |
| ('vrp_contango', 0.0) | 0.97 | 13%   | -23%    | -10%          |       0.16 |
| ('vrp_contango', 0.1) | 0.97 | 13%   | -23%    | -10%          |       0.1  |
| ('vrp_contango', 0.2) | 0.96 | 13%   | -24%    | -10%          |       0.04 |
| ('vrp_contango', 0.3) | 0.94 | 13%   | -25%    | -11%          |       0.01 |
| ('vrp_gated', 0.0)    | 0.97 | 13%   | -23%    | -10%          |       0.16 |
| ('vrp_gated', 0.1)    | 0.95 | 13%   | -24%    | -10%          |       0.13 |
| ('vrp_gated', 0.2)    | 0.93 | 13%   | -24%    | -10%          |       0.11 |
| ('vrp_gated', 0.3)    | 0.89 | 13%   | -25%    | -11%          |       0.1  |
