"""Download and cache NHL API data: play-by-play, boxscore, shift charts.

Usage:  python nhl_fetch.py 20232024 20242025 ...      (defaults to all seasons in config)

Every file is cached under data/raw_nhl/<season>/<gameId>_<kind>.json and never
re-downloaded, so the script can be stopped and restarted at any point.
Regular-season game ids are <startYear>02<0001..>; we walk them until the
season runs out (several misses in a row).
"""
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import config
import jsonio

UA = {"User-Agent": "nhl-shot-model/0.1 (personal research)"}
ENDPOINTS = {
    "pbp": "https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play",
    "box": "https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore",
    "shifts": "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}",
}


def _get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:
                time.sleep(5 * (i + 1))
                continue
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed after {tries} tries: {url}")


def path_for(season, gid, kind):
    return config.RAW_NHL / season / f"{gid}_{kind}.json.gz"


def fetch_game(season, gid):
    """Fetch all three endpoints for one game. Returns False if the game does not exist
    or has not been played yet."""
    p = path_for(season, gid, "pbp")
    if not jsonio.exists(p):
        data = _get(ENDPOINTS["pbp"].format(gid=gid))
        if data is None:
            return False
        if data.get("gameState") not in ("OFF", "FINAL"):
            return False          # scheduled / live: do not cache
        jsonio.save(p, data)
    for kind in ("box", "shifts"):
        q = path_for(season, gid, kind)
        if not jsonio.exists(q):
            d = _get(ENDPOINTS[kind].format(gid=gid))
            if d is not None:
                jsonio.save(q, d)
    return True


def fetch_season(season, workers=6, max_games=1500):
    start = season[:4]
    found, misses, n = 0, 0, 1
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        while n <= max_games and misses < 3:
            batch = [f"{start}02{k:04d}" for k in range(n, n + workers * 4)]
            results = list(ex.map(lambda g: fetch_game(season, g), batch))
            for ok in results:
                if ok:
                    found += 1
                    misses = 0
                else:
                    misses += 1
            n += len(batch)
            print(f"  {season}: {found} games cached  ({time.time() - t0:.0f}s)", end="\r", flush=True)
    print(f"  {season}: {found} games cached  ({time.time() - t0:.0f}s)        ")
    return found


if __name__ == "__main__":
    seasons = sys.argv[1:] or (config.XG_TRAIN_SEASONS + config.BACKTEST_SEASONS + [config.LIVE_SEASON])
    jsonio.compact(config.RAW_NHL)       # shrink anything saved uncompressed by an earlier run
    for s in seasons:
        fetch_season(s)
