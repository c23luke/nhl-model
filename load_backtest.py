"""One-time: load the backtest pick history into Supabase (table nhl_picks).
Usage:  SUPABASE_SERVICE_KEY=... python3 load_backtest.py
The key is the SECRET key from Supabase > Project Settings > API Keys (never the publishable one)."""
import json, os, sys, urllib.request, urllib.error

URL = "https://pbnydggxfqsmstosneth.supabase.co/rest/v1/nhl_picks?on_conflict=strategy_id,source,game_id,pick"
KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
if not KEY:
    sys.exit("set SUPABASE_SERVICE_KEY first")
COLS = ["strategy_id", "source", "game_id", "game_date", "start_utc", "season", "home", "away", "market", "pick",
        "player", "line", "model_prob", "market_prob", "bet_prob", "ev", "best_american", "best_decimal",
        "median_decimal", "stake_units", "tier", "evidence", "status", "result", "profit_units"]
rows = json.load(open(os.path.join(os.path.dirname(__file__), "output", "backtest_picks.json")))
rows = [{c: r.get(c) for c in COLS} for r in rows]
for r in rows:
    r["game_date"] = str(r["game_date"])[:10]
headers = {"apikey": KEY, "Content-Type": "application/json", "Prefer": "resolution=ignore-duplicates,return=minimal"}
if KEY.startswith("eyJ"):
    headers["Authorization"] = f"Bearer {KEY}"
done = 0
for i in range(0, len(rows), 300):
    req = urllib.request.Request(URL, data=json.dumps(rows[i:i + 300]).encode(), headers=headers, method="POST")
    try:
        urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        sys.exit(f"Supabase refused ({e.code}): {e.read().decode()[:300]}")
    done += len(rows[i:i + 300])
    print(f"  loaded {done}/{len(rows)}")
print("done")
