#!/bin/bash
# Full real-data run: download -> parse -> xG -> personnel ratings -> odds -> backtest.
# Needs: ODDS_API_KEY set in the environment (paid Odds API plan for historical odds).
# Safe to stop and rerun: every download is cached.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "== setting up Python packages (one time)"
  python3 -m venv .venv
fi
source .venv/bin/activate
python3 -m pip install -q --upgrade pip
python3 -m pip install -q pandas numpy scipy scikit-learn pyarrow
: "${ODDS_API_KEY:?set ODDS_API_KEY first, e.g. export ODDS_API_KEY=yourkey}"
echo "== 1/7 NHL play-by-play, boxscores, shifts (about 1 hour the first time)"
python3 nhl_fetch.py
echo "== 2/7 parse";            python3 parse.py
echo "== 3/7 xG model";         python3 xg.py
echo "== 4/7 personnel ratings"; python3 ratings.py
echo "== 5/7 odds credit check"; python3 odds_fetch.py --estimate
echo "== 6/7 odds pulls";        python3 odds_fetch.py --games --props --reserve "${RESERVE:-500}"
echo "== 7/7 backtest";          python3 backtest.py
echo "Report: output/report.md"
