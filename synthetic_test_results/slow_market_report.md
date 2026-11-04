# NHL shot-quality model: backtest report

Backtest games: 2400; games with odds: 2400; prop player-games priced: 86400

## moneyline
- Information beyond the market: model coefficient 0.931 (t = 5.13); market coefficient 0.060. t above 2 = the model knows something the price doesn't.
- Log loss (lower is better): market 0.6899, model 0.6833, blend 0.6846

| price | EV cutoff | bets | ROI | 90% CI | seasons + | by season | PASS |
|---|---|---|---|---|---|---|---|
| med | 2% | 1802 | +8.3% | +4.4% to +12.3% | 3 | 23-24: +10.0%, 24-25: +3.4%, 25-26: +11.2% | **PASS** |
| med | 3% | 1663 | +9.3% | +5.2% to +13.5% | 3 | 23-24: +9.5%, 24-25: +4.1%, 25-26: +14.2% | **PASS** |
| med | 5% | 1396 | +11.8% | +7.2% to +16.6% | 3 | 23-24: +8.2%, 24-25: +11.4%, 25-26: +16.0% | **PASS** |
| best | 2% | 1993 | +9.6% | +5.8% to +13.4% | 3 | 23-24: +10.8%, 24-25: +5.4%, 25-26: +12.5% | **PASS** |
| best | 3% | 1850 | +10.2% | +6.1% to +14.2% | 3 | 23-24: +12.2%, 24-25: +5.1%, 25-26: +13.0% | **PASS** |
| best | 5% | 1585 | +11.2% | +6.8% to +15.5% | 3 | 23-24: +10.7%, 24-25: +8.0%, 25-26: +14.8% | **PASS** |

## totals
- Information beyond the market: model coefficient 0.940 (t = 5.90); market coefficient 0.173. t above 2 = the model knows something the price doesn't.
- Log loss (lower is better): market 0.6776, model 0.6664

| price | EV cutoff | bets | ROI | 90% CI | seasons + | by season | PASS |
|---|---|---|---|---|---|---|---|
| med | 2% | 1934 | +9.4% | +5.7% to +13.1% | 3 | 23-24: +14.1%, 24-25: +8.2%, 25-26: +6.6% | **PASS** |
| med | 3% | 1830 | +10.7% | +7.0% to +14.4% | 3 | 23-24: +15.9%, 24-25: +8.4%, 25-26: +8.4% | **PASS** |
| med | 5% | 1598 | +12.7% | +8.7% to +16.8% | 3 | 23-24: +18.6%, 24-25: +9.6%, 25-26: +11.0% | **PASS** |
| best | 2% | 2085 | +10.6% | +7.0% to +14.1% | 3 | 23-24: +16.2%, 24-25: +9.8%, 25-26: +6.3% | **PASS** |
| best | 3% | 1965 | +10.5% | +6.8% to +14.2% | 3 | 23-24: +15.1%, 24-25: +9.9%, 25-26: +7.1% | **PASS** |
| best | 5% | 1755 | +12.7% | +9.0% to +16.5% | 3 | 23-24: +18.0%, 24-25: +9.9%, 25-26: +11.2% | **PASS** |

## sog_props
- Information beyond the market: model coefficient 0.024 (t = 1.39); market coefficient 0.963. t above 2 = the model knows something the price doesn't.
- Log loss (lower is better): market 0.6744, model 0.6925

| price | EV cutoff | bets | ROI | 90% CI | seasons + | by season | PASS |
|---|---|---|---|---|---|---|---|
| med | 2% | 61449 | -2.7% | -3.4% to -2.0% | 0 | 23-24: -3.7%, 24-25: -1.9%, 25-26: -2.5% | fail |
| med | 3% | 57105 | -2.6% | -3.3% to -1.8% | 0 | 23-24: -3.3%, 24-25: -1.8%, 25-26: -2.6% | fail |
| med | 5% | 49089 | -2.4% | -3.2% to -1.6% | 0 | 23-24: -3.1%, 24-25: -1.5%, 25-26: -2.4% | fail |
| best | 2% | 69910 | -0.2% | -0.8% to +0.5% | 1 | 23-24: -1.1%, 24-25: +0.7%, 25-26: -0.0% | fail |
| best | 3% | 65842 | -0.1% | -0.8% to +0.7% | 2 | 23-24: -0.9%, 24-25: +0.6%, 25-26: +0.1% | fail |
| best | 5% | 57958 | +0.3% | -0.4% to +1.1% | 2 | 23-24: -0.8%, 24-25: +1.1%, 25-26: +0.6% | fail |

## Tails and stacks (moneyline, median price)

| flag | bets | ROI | notes |
|---|---|---|---|
| f_model_gap | 480 | +22.8% | top 10% (home) 240 @ +23.5%; bottom 10% (away) 240 @ +22.1% |
| f_hot_goalie | 480 | -15.3% | top 10% (home) 240 @ -14.6%; bottom 10% (away) 240 @ -15.9% |
| f_luck | 480 | -1.8% | top 10% (home) 240 @ +1.5%; bottom 10% (away) 240 @ -5.2% |
| f_lineup | 480 | -0.0% | top 10% (home) 240 @ -4.4%; bottom 10% (away) 240 @ +4.4% |
| f_goalie_skill | 480 | +17.1% | top 10% (home) 240 @ +8.0%; bottom 10% (away) 240 @ +26.2% |
| f_rest | 1038 | +2.3% |  |
| stack>=2 | 937 | +7.2% |  |
| stack>=3 | 307 | +21.3% |  |
| stack>=4 | 71 | +32.6% |  |

6 flags and 3 stack levels tested; expect ~1 to look good by chance.

## The gate (fixed before results)
PASS needs: at least 300 bets, ROI > 0, the lower end of the 90% bootstrap CI > 0, and positive in at least 2 of 3 seasons. Headline row = median price at 3% EV.