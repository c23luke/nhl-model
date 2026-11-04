"""Download historical NHL odds from The Odds API (paid plans only for historical data).

Usage:
  ODDS_API_KEY=... python odds_fetch.py --estimate          # what it would cost; spends nothing
  ODDS_API_KEY=... python odds_fetch.py --games             # moneyline + totals snapshots
  ODDS_API_KEY=... python odds_fetch.py --props             # shots-on-goal props, per event
  add --reserve N to always leave N credits unspent (default 500)

Snapshot time = puck drop minus config.SNAPSHOT_MINUTES_BEFORE, so prices are near closing.
Game lines: one call per 90-minute window of start times (see windows()).
  cost = 10 x 2 markets x 1 region = 20 credits per call.
Props: one call per game, 10 credits when the book had lines (empty responses are free).
Everything is cached; reruns only pull what is missing.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import config
import jsonio

BASE = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"
REMAINING = {"n": None}


def _call(path, params):
    params = dict(params, apiKey=config.ODDS_API_KEY)
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    for i in range(4):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                rem = r.headers.get("x-requests-remaining")
                if rem is not None:
                    REMAINING["n"] = float(rem)
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:300]
            if e.code in (401, 403):
                sys.exit(f"Odds API refused the key ({e.code}): {body}")
            if e.code == 422:
                return None          # e.g. no data for that snapshot
            if e.code == 429:
                time.sleep(3 * (i + 1))
                continue
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def credits_remaining():
    _call("/sports", {})       # free call, but returns the quota headers
    return REMAINING["n"]


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def start_times():
    """Distinct puck-drop times for every cached regular-season game in backtest seasons."""
    times = set()
    for season in config.BACKTEST_SEASONS:
        for p in sorted((config.RAW_NHL / season).glob("*_pbp.json.gz")):
            g = jsonio.load(p)
            t = g.get("startTimeUTC")
            if t:
                times.add(t.replace(".000", ""))
    return sorted(times)


def snap_time(start_iso):
    t = datetime.strptime(start_iso[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return iso(t - timedelta(minutes=config.SNAPSHOT_MINUTES_BEFORE))


def game_file(start_iso):
    return config.RAW_ODDS / "games" / f"{start_iso[:19].replace(':', '')}.json.gz"


def prop_file(event_id):
    return config.RAW_ODDS / "props" / f"{event_id}.json.gz"


def windows(times, minutes=90):
    """Group puck-drop times: one snapshot (first start minus 15 min) covers every game
    starting within `minutes` of the first one. Saves ~60% of credits; later games in a
    window are priced up to ~1.5 h before their start instead of 15 min."""
    out, cur = [], []
    for t in times:
        if cur and _dt(t) - _dt(cur[0]) > timedelta(minutes=minutes):
            out.append(cur)
            cur = []
        cur.append(t)
    if cur:
        out.append(cur)
    return out


def _dt(s):
    return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")


def fetch_games(reserve):
    todo = [w for w in windows(start_times()) if not all(jsonio.exists(game_file(s)) for s in w)]
    print(f"game-line snapshots to pull: {len(todo)}  (~{20 * len(todo)} credits)")
    for i, w in enumerate(todo):
        if REMAINING["n"] is not None and REMAINING["n"] - 20 < reserve:
            print(f"\nstopping: {REMAINING['n']:.0f} credits left, reserve {reserve}")
            return
        d = _call(f"/historical/sports/{SPORT}/odds",
                  {"regions": config.ODDS_REGIONS, "markets": config.GAME_MARKETS,
                   "oddsFormat": "american", "date": snap_time(w[0])})
        for s in w:                      # same snapshot filed under each start time it covers
            jsonio.save(game_file(s), d or {"data": []})
        print(f"  {i + 1}/{len(todo)}  credits left {REMAINING['n']}", end="\r", flush=True)
    print()


def events_needing_props():
    """(event_id, commence_time) pairs from cached game-line snapshots, newest first,
    limited to dates where the Odds API has props (from 2023-05-03)."""
    out = {}
    for f in (config.RAW_ODDS / "games").glob("*.json.gz"):
        for ev in jsonio.load(f).get("data", []):
            if ev["commence_time"] >= "2023-05-03":
                out[ev["id"]] = ev["commence_time"]
    return sorted(out.items(), key=lambda kv: kv[1], reverse=True)


def fetch_props(reserve):
    todo = [(e, t) for e, t in events_needing_props() if not jsonio.exists(prop_file(e))]
    print(f"prop events to pull: {len(todo)}  (~{10 * len(todo)} credits, newest first)")
    for i, (eid, t) in enumerate(todo):
        if REMAINING["n"] is not None and REMAINING["n"] - 10 < reserve:
            print(f"\nstopping: {REMAINING['n']:.0f} credits left, reserve {reserve}")
            return
        d = _call(f"/historical/sports/{SPORT}/events/{eid}/odds",
                  {"regions": config.ODDS_REGIONS, "markets": config.PROP_MARKETS,
                   "oddsFormat": "american", "date": snap_time(t)})
        jsonio.save(prop_file(eid), d or {"data": {}})
        print(f"  {i + 1}/{len(todo)}  credits left {REMAINING['n']}", end="\r", flush=True)
    print()


def estimate():
    n_times = len([w for w in windows(start_times()) if not all(jsonio.exists(game_file(s)) for s in w)])
    seasons_with_props = [s for s in config.BACKTEST_SEASONS if s >= "20232024"]
    games = sum(len(list((config.RAW_NHL / s).glob("*_pbp.json.gz"))) for s in seasons_with_props)
    print(f"credits remaining on key : {credits_remaining()}")
    print(f"game-line snapshots      : {n_times} x 20 = {20 * n_times} credits")
    print(f"prop events (upper bound): {games} x 10 = {10 * games} credits")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--games", action="store_true")
    ap.add_argument("--props", action="store_true")
    ap.add_argument("--reserve", type=float, default=500)
    a = ap.parse_args()
    if not config.ODDS_API_KEY:
        sys.exit("set ODDS_API_KEY first")
    if a.estimate:
        estimate()
    if a.games:
        credits_remaining()
        fetch_games(a.reserve)
    if a.props:
        credits_remaining()
        fetch_props(a.reserve)
