#!/bin/bash
# Saves one copy of every live data source so the live board can be built against real files.
# Costs about 4 Odds API credits. Run from ~/nhl-model with ODDS_API_KEY set.
set -u
cd "$(dirname "$0")"
mkdir -p samples && cd samples
D=$(date +%Y-%m-%d)
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
echo "1/7 NHL schedule";        curl -s "https://api-web.nhle.com/v1/schedule/$D" -o nhl_schedule.json
GID=$(python3 -c "import json;d=json.load(open('nhl_schedule.json'));g=[x for w in d['gameWeek'] for x in w['games'] if w['date']=='$D'];print(g[0]['id'] if g else '')")
echo "   first game today: $GID"
echo "2/7 NHL pregame boxscore + play-by-play"
curl -s "https://api-web.nhle.com/v1/gamecenter/$GID/boxscore" -o nhl_pregame_box.json
curl -s "https://api-web.nhle.com/v1/gamecenter/$GID/play-by-play" -o nhl_pregame_pbp.json
curl -s "https://api-web.nhle.com/v1/gamecenter/$GID/landing" -o nhl_pregame_landing.json
curl -s "https://api-web.nhle.com/v1/roster/TOR/current" -o nhl_roster_TOR.json
echo "3/7 DailyFaceoff starting goalies"; curl -s -A "$UA" "https://www.dailyfaceoff.com/starting-goalies/" -o dfo_goalies.html
echo "4/7 DailyFaceoff line combos (TOR)"; curl -s -A "$UA" "https://www.dailyfaceoff.com/teams/toronto-maple-leafs/line-combinations" -o dfo_lines_TOR.html
BOOKS="draftkings,bovada,novig,prophetx,kalshi,polymarket,polymarket_us,betopenly"
echo "5/7 Odds API moneyline + totals, your 7 books"
curl -s "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds?apiKey=$ODDS_API_KEY&bookmakers=$BOOKS&markets=h2h,totals&oddsFormat=american&includeBetLimits=true" -o odds_games.json
EID=$(python3 -c "import json;d=json.load(open('odds_games.json'));print(d[0]['id'] if isinstance(d,list) and d else '')")
echo "6/7 Odds API shots-on-goal props for one game ($EID)"
curl -s "https://api.the-odds-api.com/v4/sports/icehockey_nhl/events/$EID/odds?apiKey=$ODDS_API_KEY&bookmakers=$BOOKS&markets=player_shots_on_goal&oddsFormat=american" -o odds_props.json
echo "7/7 sizes:"; ls -la
python3 -c "import json;d=json.load(open('odds_games.json'));print('odds events:',len(d) if isinstance(d,list) else d)"
