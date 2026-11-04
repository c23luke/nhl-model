# NHL model: real-data backtest results (2026-11-03)

Data: 3 backtest seasons (2023-24, 2024-25, 2025-26), 3,936 games, all with near-closing odds;
56,865 shots-on-goal prop lines matched to players.

## Fixed pass/fail gate (set before any results)
At least 300 bets, ROI > 0, lower end of the 90% CI > 0, profit in 2 of 3 seasons, at the
median book price with a 3% EV cutoff.

## As built: nothing passes
| market | bets | ROI (median price, 3% EV) | does the model know something the price doesn't? |
|---|---|---|---|
| moneyline | 1,942 | -1.3% | weak (t = 1.6) |
| totals | 1,915 | -1.6% | some (t = 2.1) |
| SOG props | 18,545 | -2.2% | yes, strongly (t = 7.5) |

## Lead 1: stacked flags on the moneyline (Caden's stack method)
Bet a side when 3+ of these point the same way, each in its own top/bottom 20%: the model's
gap vs the market, opposing goalie's last-10 save %, finishing luck over the last 15 games,
lineup strength vs the team's recent lineups, goalie skill, rest. These settings were in the
code before any real data was seen. Cutoffs use only games before each date.

| season | bets | ROI |
|---|---|---|
| 2023-24 | 85 | +5.6% |
| 2024-25 | 104 | +22.7% |
| 2025-26 | 101 | +16.7% |
| total | 290 | +15.6% (90% CI +5.6% to +25.6%) |

Fails the gate only on bet count (290 vs 300). About 3-4 bets a week. Checks run:
- odds always captured before puck drop
- right event matched in all 3,936 games
- holds at 15/25/30% tails
- holds with any single flag removed
- holds for favourites and dogs

Caution: a 60% win rate at an average price of +109 is unusually strong. Treat it as a lead
to confirm live.

## Lead 2: SOG props, model blended with the market
The model alone is overconfident. Combining it with the market price (weights fit on earlier
seasons only) gives:
- best available price, 3% EV: 7,468 bets, +3.2%, 90% CI low +1.2%, positive in both test seasons
- median price, 3% EV: 2,238 bets, +0.6%, CI low below 0, so it fails

So it only works with line shopping across several books.

## Not leads
Totals and moneyline as standalone models, and their blends, all fail.

## Next
Forward-test leads 1 and 2 live this season with a daily board, tracked against closing lines.
