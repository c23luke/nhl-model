# NHL shot-quality model: backtest report

Backtest games: 2400; games with odds: 2400; prop player-games priced: 86400

## moneyline
- Information beyond the market: model coefficient 0.038 (t = 0.15); market coefficient 0.875. t above 2 = the model knows something the price doesn't.
- Log loss (lower is better): market 0.6783, model 0.6833, blend 0.6792

| price | EV cutoff | bets | ROI | 90% CI | seasons + | by season | PASS |
|---|---|---|---|---|---|---|---|
| med | 2% | 1745 | -0.1% | -4.6% to +4.5% | 1 | 23-24: +2.6%, 24-25: -0.2%, 25-26: -2.8% | fail |
| med | 3% | 1606 | +0.5% | -4.2% to +5.3% | 2 | 23-24: +3.2%, 24-25: +0.8%, 25-26: -2.6% | fail |
| med | 5% | 1335 | -0.2% | -5.4% to +5.1% | 1 | 23-24: +3.6%, 24-25: -1.8%, 25-26: -2.9% | fail |
| best | 2% | 1941 | +1.8% | -2.4% to +6.1% | 2 | 23-24: +4.6%, 24-25: +1.5%, 25-26: -1.0% | fail |
| best | 3% | 1790 | +1.8% | -2.7% to +6.3% | 2 | 23-24: +4.4%, 24-25: +1.7%, 25-26: -0.7% | fail |
| best | 5% | 1539 | +3.1% | -1.7% to +8.2% | 3 | 23-24: +5.4%, 24-25: +2.9%, 25-26: +0.9% | fail |

## totals
- Information beyond the market: model coefficient -0.042 (t = -0.23); market coefficient 1.361. t above 2 = the model knows something the price doesn't.
- Log loss (lower is better): market 0.6840, model 0.6910

| price | EV cutoff | bets | ROI | 90% CI | seasons + | by season | PASS |
|---|---|---|---|---|---|---|---|
| med | 2% | 1713 | -2.7% | -6.6% to +1.2% | 1 | 23-24: +2.7%, 24-25: -6.8%, 25-26: -4.4% | fail |
| med | 3% | 1566 | -2.9% | -7.1% to +1.1% | 1 | 23-24: +0.8%, 24-25: -7.0%, 25-26: -3.0% | fail |
| med | 5% | 1268 | -2.4% | -7.0% to +2.3% | 1 | 23-24: +0.0%, 24-25: -5.3%, 25-26: -2.4% | fail |
| best | 2% | 1920 | -1.3% | -5.0% to +2.4% | 1 | 23-24: +3.2%, 24-25: -6.0%, 25-26: -1.2% | fail |
| best | 3% | 1773 | -1.7% | -5.6% to +2.3% | 1 | 23-24: +3.3%, 24-25: -5.9%, 25-26: -2.6% | fail |
| best | 5% | 1469 | -2.7% | -6.9% to +1.7% | 0 | 23-24: -0.8%, 24-25: -5.7%, 25-26: -1.8% | fail |

## sog_props
- Information beyond the market: model coefficient 0.024 (t = 1.39); market coefficient 0.963. t above 2 = the model knows something the price doesn't.
- Log loss (lower is better): market 0.6744, model 0.6925

| price | EV cutoff | bets | ROI | 90% CI | seasons + | by season | PASS |
|---|---|---|---|---|---|---|---|
| med | 2% | 61449 | -2.7% | -3.4% to -2.0% | 0 | 23-24: -3.7%, 24-25: -1.9%, 25-26: -2.5% | fail |
| med | 3% | 57105 | -2.6% | -3.3% to -1.8% | 0 | 23-24: -3.3%, 24-25: -1.8%, 25-26: -2.6% | fail |
| med | 5% | 49089 | -2.4% | -3.2% to -1.6% | 0 | 23-24: -3.1%, 24-25: -1.5%, 25-26: -2.4% | fail |
| best | 2% | 69910 | -0.2% | -0.9% to +0.5% | 1 | 23-24: -1.1%, 24-25: +0.7%, 25-26: -0.0% | fail |
| best | 3% | 65842 | -0.1% | -0.8% to +0.6% | 2 | 23-24: -0.9%, 24-25: +0.6%, 25-26: +0.1% | fail |
| best | 5% | 57958 | +0.3% | -0.5% to +1.1% | 2 | 23-24: -0.8%, 24-25: +1.1%, 25-26: +0.6% | fail |

## Tails and stacks (moneyline, median price)

| flag | bets | ROI | notes |
|---|---|---|---|
| f_model_gap | 480 | -3.6% | top 10% (home) 240 @ -6.3%; bottom 10% (away) 240 @ -1.0% |
| f_hot_goalie | 480 | -3.2% | top 10% (home) 240 @ -4.2%; bottom 10% (away) 240 @ -2.1% |
| f_luck | 480 | -2.1% | top 10% (home) 240 @ +0.5%; bottom 10% (away) 240 @ -4.8% |
| f_lineup | 480 | -6.2% | top 10% (home) 240 @ -10.7%; bottom 10% (away) 240 @ -1.7% |
| f_goalie_skill | 480 | -0.7% | top 10% (home) 240 @ -7.7%; bottom 10% (away) 240 @ +6.2% |
| f_rest | 1038 | -2.7% |  |
| stack>=2 | 747 | -3.7% |  |
| stack>=3 | 208 | -5.0% |  |
| stack>=4 | 34 | +32.2% |  |

6 flags and 3 stack levels tested; expect ~1 to look good by chance.

## The gate (fixed before results)
PASS needs: at least 300 bets, ROI > 0, the lower end of the 90% bootstrap CI > 0, and positive in at least 2 of 3 seasons. Headline row = median price at 3% EV.