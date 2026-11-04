# NHL shot-quality model

A hockey version of the ShotQuality idea. Every shot gets a goal chance. Every player is
rated from his own history, wherever he played it. Tonight's team is rated from the
players who actually dress and the goalie who actually starts.

## Run it on real data (one command)

```
export ODDS_API_KEY=your_key      # historical odds need a paid Odds API plan
./run_all.sh
```

Read `output/report.md` when it finishes. The first run downloads about 7 seasons from the
NHL API, which takes roughly an hour. Everything is cached, so you can stop the run and
start it again at any point.

## What each file does

| file | job |
|---|---|
| `config.py` | seasons, half-lives, odds settings, and the **pass/fail gate**, which is fixed before any real results are seen |
| `nhl_fetch.py` | downloads play-by-play, boxscores and shift charts from the NHL API (free, no key) |
| `odds_fetch.py` | downloads historical moneyline, totals and shots-on-goal props from the Odds API, taking each price 15 minutes before puck drop; `--estimate` shows the credit cost first |
| `parse.py` | turns the raw JSON into tables: every shot attempt with its location, type, rebound/rush/cross-ice flags, strength, and the 10 skaters on the ice |
| `xg.py` | the shot-quality model; each season is scored by a model trained only on earlier seasons |
| `ratings.py` | personnel ratings from a player's own on-ice and individual numbers, shrunk toward league average by amounts **measured** from the data, then rolled up to tonight's lineup |
| `model.py` | game model (goals for each team, giving win and total probabilities, OT handled) and the shots-on-goal prop model; refit every 14 days on past games only |
| `backtest.py` | bets wherever expected value clears the cutoff, at the median book price, runs the gate, and builds the tail/stack table |
| `synth.py` | a fake league written in the exact API formats, used to test the pipeline before real data |

## Rules built in

- No team-season stat is ever an input. A player's history follows the player.
- Nothing from the current day or later is used to rate a game.
- How much to trust a player's own number (shrinkage) is measured with split-half
  reliability on the training seasons, not picked by hand.
- Every tail flag tested is reported, winners and losers alike.

## Credit cost (Odds API)

- Game lines: 20 credits per snapshot; one snapshot covers every game starting within 90 minutes,
  about 2 per game night, so roughly 7k credits a season, ~22k for 3 seasons.
- Props: 10 credits per game, about 13k a season, ~39k for 3 seasons.
- Whole backtest: ~60k credits, so the 100k-credit plan covers it; the 20k plan covers the
  game lines plus about half a season of props.
- `odds_fetch.py --estimate` prints the exact numbers and your remaining credits. Props are
  pulled newest-first, so a smaller plan still gets the most recent seasons.
