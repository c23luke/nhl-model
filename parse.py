"""Turn cached NHL API json into flat tables (parquet under data/tables/).

games.parquet    one row per game: date, teams, final + regulation score, OT/SO
shots.parquet    one row per shot attempt (incl. blocked) with xG features + who was on ice
players.parquet  one row per player per game: TOI by strength, individual + on-ice numbers
"""
import json
import math
import sys
import unicodedata
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
import jsonio

TABLES = config.DATA / "tables"
SHOT_EVENTS = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}
SHOT_TYPES = {"wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around"}
NET_X = 89.0


def norm_name(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch == " ").strip()


def mmss(s):
    if not s:
        return 0
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def _name(d):
    if isinstance(d, dict):
        return d.get("default", "")
    return d or ""


def et_date(start_utc, fallback):
    """Game date in US Eastern (betting day). Approximate DST-free offset is fine for dating."""
    if not start_utc:
        return fallback
    t = datetime.strptime(start_utc[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return (t - timedelta(hours=5)).strftime("%Y-%m-%d")


# ------------------------------------------------------------------ shifts
def parse_shifts(shift_json):
    """List of (player_id, team_id, abs_start_sec, abs_end_sec)."""
    out = []
    for r in (shift_json or {}).get("data", []):
        if r.get("typeCode", 517) != 517 or r.get("eventDescription"):
            continue                      # goal markers etc, not shifts
        p = int(r["period"])
        if p > 4:
            continue                      # shootout "shifts" are not ice time
        st, en = mmss(r.get("startTime")), mmss(r.get("endTime"))
        if en <= st:
            continue
        base = (p - 1) * 1200
        out.append((int(r["playerId"]), int(r["teamId"]), base + st, base + en))
    return out


# ------------------------------------------------------------------ one game
def parse_game(pbp, box, shifts_json):
    gid = pbp["id"]
    home, away = pbp["homeTeam"], pbp["awayTeam"]
    hid, aid = home["id"], away["id"]
    roster = {r["playerId"]: r for r in pbp.get("rosterSpots", [])}
    goalies = {pid for pid, r in roster.items() if r.get("positionCode") == "G"}

    plays = sorted(pbp.get("plays", []), key=lambda p: p.get("sortOrder", 0))
    last_period = max([p.get("periodDescriptor", {}).get("number", 3) for p in plays] or [3])
    shootout = any(p.get("periodDescriptor", {}).get("periodType") == "SO" for p in plays)
    game_len = max(3600, (min(last_period, 4) - 1) * 1200 + 300) if last_period > 3 else 3600

    # --- shift timeline (who is on ice each second)
    sh = parse_shifts(shifts_json)
    T = int(game_len) + 2
    sk = {hid: np.zeros(T, np.int16), aid: np.zeros(T, np.int16)}
    gl = {hid: np.zeros(T, np.int16), aid: np.zeros(T, np.int16)}
    for pid, tid, s, e in sh:
        if tid not in sk:
            continue
        e = min(e, T - 1)
        (gl if pid in goalies else sk)[tid][s:e] += 1
    have_shifts = len(sh) > 0

    def state_at(t, team):
        """(own skaters, opp skaters, opp goalie in net) from shifts at second t."""
        t = min(max(int(t), 0), T - 1)
        opp = aid if team == hid else hid
        return int(sk[team][t]), int(sk[opp][t]), int(gl[opp][t] > 0)

    sk_shifts = {}
    for team in (hid, aid):
        arr = np.array([(pid, s, e) for pid, tid, s, e in sh if tid == team and pid not in goalies],
                       dtype=np.int64).reshape(-1, 3)
        sk_shifts[team] = arr

    def on_ice(t, team):
        a = sk_shifts[team]
        return a[(a[:, 1] < t) & (t <= a[:, 2]), 0].tolist()

    # --- shots
    rows = []
    hs = as_ = 0
    reg_h = reg_a = 0
    prev = None
    for p in plays:
        typ = p.get("typeDescKey")
        per = p.get("periodDescriptor", {}).get("number", 1)
        ptype = p.get("periodDescriptor", {}).get("periodType", "REG")
        if ptype == "SO":
            continue
        t = (per - 1) * 1200 + mmss(p.get("timeInPeriod"))
        d = p.get("details", {}) or {}
        team = d.get("eventOwnerTeamId")
        if typ in SHOT_EVENTS and team in (hid, aid) and d.get("xCoord") is not None:
            is_home = team == hid
            x, y = float(d["xCoord"]), float(d.get("yCoord") or 0)
            # orient so the shooting team attacks +x
            side = p.get("homeTeamDefendingSide")
            if side in ("left", "right"):
                attack_pos = (side == "left") == is_home
            else:
                attack_pos = (x > 0) if d.get("zoneCode") == "O" else (x < 0)
            if not attack_pos:
                x, y = -x, -y
            dist = math.hypot(NET_X - x, y)
            ang = math.degrees(math.atan2(abs(y), NET_X - x))
            sc = p.get("situationCode") or "1551"
            ag, ask, hsk, hg = (int(c) for c in sc.zfill(4)[:4])
            own_sk, opp_sk = (hsk, ask) if is_home else (ask, hsk)
            opp_g = ag if is_home else hg
            # previous event features
            dt, pdx, pdy, prev_shot_same, prev_zone, prev_type = 99, 0.0, 0.0, 0, "N", "none"
            if prev is not None and prev["per"] == per:
                dt = t - prev["t"]
                prev_type = prev["typ"]
                same = prev["team"] == team
                px, py = prev["x"], prev["y"]
                if px is not None:
                    # same rink, same period: apply this shooter's flip to the raw prev coords
                    qx, qy = (px, py) if attack_pos else (-px, -py)
                    pdx, pdy = x - qx, y - qy
                    prev_zone = "O" if qx > 25 else ("D" if qx < -25 else "N")
                prev_shot_same = int(same and prev["typ"] in SHOT_EVENTS)
            shooter = d.get("scoringPlayerId") if typ == "goal" else d.get("shootingPlayerId")
            goal = typ == "goal"
            diff = (hs - as_) if is_home else (as_ - hs)
            stype = d.get("shotType", "other")
            rows.append({
                "game_id": gid, "team_id": team, "opp_id": aid if is_home else hid,
                "is_home": int(is_home), "period": per, "t": t,
                "shooter": shooter, "goalie": d.get("goalieInNetId"),
                "event": typ, "goal": int(goal),
                "on_goal": int(typ in ("goal", "shot-on-goal")), "blocked": int(typ == "blocked-shot"),
                "dist": dist, "angle": ang, "x": x, "y": y,
                "shot_type": stype if stype in SHOT_TYPES else "other",
                "own_sk": own_sk, "opp_sk": opp_sk, "empty_net": int(opp_g == 0),
                "score_diff": max(-3, min(3, diff)),
                "dt_prev": min(dt, 99), "prev_type": prev_type,
                "rebound": int(prev_shot_same and dt <= 3),
                "rush": int(prev_zone in ("N", "D") and dt <= 4),
                "cross_ice": int(abs(pdy) > 20 and dt <= 3 and prev_zone == "O"),
                "speed_prev": (math.hypot(pdx, pdy) / dt) if 0 < dt < 99 else 0.0,
            })
            if goal:
                if is_home:
                    hs += 1
                    reg_h += per <= 3
                else:
                    as_ += 1
                    reg_a += per <= 3
        rx = d.get("xCoord")
        prev = {"per": per, "t": t, "typ": typ, "team": team,
                "x": float(rx) if rx is not None else None,
                "y": float(d.get("yCoord") or 0)}

    shots = pd.DataFrame(rows)

    # --- on-ice attribution for each shot (5v5 only for on-ice stats)
    if have_shifts and len(shots):
        shots["state_ok"] = [
            state_at(t, tm) for t, tm in zip(shots["t"], shots["team_id"])]
        shots["five_v_five"] = [int(a == 5 and b == 5 and c == 1) for a, b, c in shots["state_ok"]]
        shots = shots.drop(columns="state_ok")
        shots["for_on_ice"] = [on_ice(t, tm) for t, tm in zip(shots["t"], shots["team_id"])]
        shots["against_on_ice"] = [on_ice(t, op) for t, op in zip(shots["t"], shots["opp_id"])]
    elif len(shots):
        shots["five_v_five"] = ((shots.own_sk == 5) & (shots.opp_sk == 5) & (shots.empty_net == 0)).astype(int)
        shots["for_on_ice"] = [[] for _ in range(len(shots))]
        shots["against_on_ice"] = [[] for _ in range(len(shots))]

    # --- strength-split TOI per player from shifts
    toi = {}
    if have_shifts:
        st = {}
        for team, opp in ((hid, aid), (aid, hid)):
            ev = (sk[team] == 5) & (sk[opp] == 5) & (gl[team] > 0) & (gl[opp] > 0)
            both_g = (gl[team] > 0) & (gl[opp] > 0)       # a pulled goalie is not a power play
            pp = (sk[team] > sk[opp]) & both_g
            pk = (sk[team] < sk[opp]) & both_g
            st[team] = {k: np.concatenate([[0], np.cumsum(v.astype(np.int32))]) for k, v in
                        (("ev", ev), ("pp", pp), ("pk", pk))}
        for pid, tid, s, e in sh:
            if pid in goalies or tid not in st or s >= T - 1:
                continue
            e = min(e, T - 1)
            r = toi.setdefault(pid, {"toi": 0, "toi_5v5": 0, "toi_pp": 0, "toi_pk": 0})
            r["toi"] += e - s
            for k, col in (("ev", "toi_5v5"), ("pp", "toi_pp"), ("pk", "toi_pk")):
                c = st[tid][k]
                r[col] += int(c[e] - c[s])

    # --- player rows from the boxscore (who dressed, TOI, SOG, starter)
    prow = []
    pbg = (box or {}).get("playerByGameStats", {})
    for side, tid, oid in (("homeTeam", hid, aid), ("awayTeam", aid, hid)):
        grp = pbg.get(side, {})
        for pos_key in ("forwards", "defense", "goalies"):
            for pl in grp.get(pos_key, []):
                pid = pl["playerId"]
                box_toi = mmss(pl.get("toi") or "0:00")
                if box_toi == 0:
                    continue          # dressed but did not play
                r = {"game_id": gid, "player_id": pid, "team_id": tid, "opp_id": oid,
                     "is_home": int(tid == hid), "pos": pos_key[0].upper(),
                     "name": _name(pl.get("name")), "box_toi": box_toi,
                     "sog": pl.get("sog", pl.get("shots", 0)) or 0,
                     "goals": pl.get("goals", 0) or 0}
                if pos_key == "goalies":
                    r["starter"] = int(bool(pl.get("starter", False)))
                r.update(toi.get(pid, {"toi": box_toi, "toi_5v5": np.nan, "toi_pp": np.nan, "toi_pk": np.nan}))
                prow.append(r)
    players = pd.DataFrame(prow)
    if len(players) and "starter" in players:
        # fallback: starter = goalie with the most TOI when the flag is missing
        g = players[players.pos == "G"]
        for tid, grp in g.groupby("team_id"):
            if grp["starter"].sum() == 0:
                players.loc[grp["box_toi"].idxmax(), "starter"] = 1

    # full names for odds matching
    if len(players):
        nm = {pid: f"{_name(r.get('firstName'))} {_name(r.get('lastName'))}" for pid, r in roster.items()}
        players["full_name"] = players.player_id.map(nm).fillna(players["name"])

    game = {
        "game_id": gid, "season": str(pbp.get("season")),
        "date": et_date(pbp.get("startTimeUTC"), pbp.get("gameDate")),
        "start_utc": (pbp.get("startTimeUTC") or "")[:19],
        "home_id": hid, "away_id": aid,
        "home_abbrev": home.get("abbrev"), "away_abbrev": away.get("abbrev"),
        "home_name": f"{_name(home.get('placeName'))} {_name(home.get('commonName'))}".strip(),
        "away_name": f"{_name(away.get('placeName'))} {_name(away.get('commonName'))}".strip(),
        "home_goals": home.get("score", hs),     # final, incl. the shootout goal
        "away_goals": away.get("score", as_),
        "home_reg": reg_h, "away_reg": reg_a,
        "went_ot": int(last_period > 3), "shootout": int(shootout),
        "has_shifts": int(have_shifts),
    }
    return game, shots, players


def build(seasons=None):
    seasons = seasons or (config.XG_TRAIN_SEASONS + config.BACKTEST_SEASONS + [config.LIVE_SEASON])
    TABLES.mkdir(parents=True, exist_ok=True)
    G, S, P = [], [], []
    for season in seasons:
        if not (config.RAW_NHL / season).exists():
            continue
        files = sorted((config.RAW_NHL / season).glob("*_pbp.json.gz"))
        for i, f in enumerate(files):
            gid = f.name.split("_")[0]
            pbp = jsonio.load(f)
            if pbp.get("gameType") != 2:
                continue
            bf = f.with_name(f"{gid}_box.json.gz")
            sf = f.with_name(f"{gid}_shifts.json.gz")
            box = jsonio.load(bf) if jsonio.exists(bf) else {}
            shf = jsonio.load(sf) if jsonio.exists(sf) else {}
            g, s, p = parse_game(pbp, box, shf)
            G.append(g)
            if len(s):
                s["season"] = g["season"]
                S.append(s)
            if len(p):
                p["season"] = g["season"]
                P.append(p)
            if i % 200 == 0:
                print(f"  parsed {season}: {i}/{len(files)}", end="\r", flush=True)
        print(f"  parsed {season}: {len(files)} games            ")
    if not G:
        empty = pd.DataFrame()
        return empty, empty, empty
    games = pd.DataFrame(G).sort_values(["date", "game_id"]).reset_index(drop=True)
    shots = pd.concat(S, ignore_index=True)
    players = pd.concat(P, ignore_index=True)
    games.to_parquet(TABLES / "games.parquet")
    shots.to_parquet(TABLES / "shots.parquet")
    players.to_parquet(TABLES / "players.parquet")
    print(f"games {len(games)}  shots {len(shots)}  player-games {len(players)}  "
          f"games missing shift data: {int((games.has_shifts == 0).sum())}")
    return games, shots, players


if __name__ == "__main__":
    build(sys.argv[1:] or None)
