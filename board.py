"""Tonight's board. Safe to run as often as you like: every run re-pulls lineups, goalies and
prices, re-rates each game, and updates the picks in Supabase.

A pick is READY only when all four checks pass:
  goalies  both starters marked Confirmed on DailyFaceoff -- or, for a game bet, the bet still
           qualifies with the other goalie in net for every team whose starter isn't confirmed yet
           (props still wait for confirmed goalies)
  lineups  both teams' lines updated within 14 hours of puck drop and fully matched to NHL ids
  prices   at least 2 of the books priced the side
  time     puck drop is at least 10 minutes away
  gtd      for each game-time decision in either lineup, the bet still qualifies with him sitting
           (his spot taken by the team's most-used extra skater); props on that player wait outright
Otherwise it is WAITING and says what it is waiting on.

Usage: ODDS_API_KEY=... SUPABASE_SERVICE_KEY=... python board.py [--date YYYY-MM-DD] [--dry]
"""
import argparse
import itertools
import json
import os
import pickle
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

import config
import lineups as LU
import model
import pricing as PX
import strategies as S
from parse import TABLES, norm_name
from ratings import Rater

STAKE = {"ml_stack3": 1.0, "ml_blend5": 1.0, "sog_blend5": 0.5}        # BETS; everything else is tracking
MIN_MINUTES = 10


def odds_call(path, params):
    params = dict(params, apiKey=config.ODDS_API_KEY, bookmakers=",".join(PX.BOOKS), oddsFormat="american")
    return LU.get(f"https://api.the-odds-api.com/v4{path}?{urllib.parse.urlencode(params)}", as_json=True)


def last_token(s):
    t = norm_name(s).split()
    return t[-1] if t else ""


def find_event(events, home_name, away_name, start):
    for e in events:
        if last_token(e["home_team"]) == last_token(home_name) and last_token(e["away_team"]) == last_token(away_name):
            ct = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
            if abs((ct - start).total_seconds()) < 4 * 3600:
                return e
    return None


def nb_over(mu, alpha, line):
    return model.p_over_nb(mu, alpha, line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--dry", action="store_true", help="print picks, write nothing")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    date_et = a.date or (now - timedelta(hours=5)).strftime("%Y-%m-%d")

    # ---------------- what we know as of this morning
    if not (TABLES / "rater_state.pkl").exists() or not (config.MODEL_FILES / "live_models.pkl").exists():
        print("morning prep hasn't finished yet - nothing to do this run")
        return
    R = Rater.load(TABLES / "rater_state.pkl")
    with open(config.MODEL_FILES / "live_models.pkl", "rb") as f:
        models = pickle.load(f)
    fh = pd.read_parquet(config.MODEL_FILES / "flag_history.parquet")
    blend = json.loads((config.MODEL_FILES / "blend.json").read_text())

    sched = LU.get(f"https://api-web.nhle.com/v1/schedule/{date_et}", as_json=True)
    games = [g for w in sched.get("gameWeek", []) if w["date"] == date_et for g in w["games"]
             if g.get("gameType") == 2]
    upcoming = [g for g in games if g["gameState"] in ("FUT", "PRE")]
    print(f"{date_et}: {len(games)} games, {len(upcoming)} not started")
    if not upcoming:
        return

    try:
        goalies = LU.fetch_starting_goalies(date_et)
    except Exception as e:
        print("  starting goalies page failed:", e)
        goalies = {}
    events = odds_call("/sports/icehockey_nhl/odds", {"markets": "h2h,totals", "includeBetLimits": "true"})

    rows, props, info = [], [], {}
    for g in upcoming:
        H, A = g["homeTeam"], g["awayTeam"]
        ha, aa = H["abbrev"], A["abbrev"]
        start = datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00"))
        try:      # tonight's NHL roster (active players incl. scratches), used to match ids by jersey
            spots = LU.get(f"https://api-web.nhle.com/v1/gamecenter/{g['id']}/play-by-play",
                           as_json=True).get("rosterSpots", [])
        except Exception:
            spots = []
        lu = {}
        for ab, tid in ((ha, H["id"]), (aa, A["id"])):
            try:
                lu[ab] = LU.team_lineup(ab, start, [s for s in spots if s.get("teamId") == tid])
            except Exception as e:
                lu[ab] = {"ok": False, "why": f"{ab} lineup fetch failed: {e}"}
        gl = {}
        for ab in (ha, aa):
            x = goalies.get(ab)
            gid = None
            if x and lu[ab].get("ok"):
                m = LU.match(x["name"], lu[ab]["ids"])
                gid = m[0] if m else None
            gl[ab] = {"id": gid or (lu[ab].get("projected_goalie_id") if lu[ab].get("ok") else None),
                      "name": (x or {}).get("name") or ((lu[ab].get("projected_goalie") or [None])[0] if lu[ab].get("ok") else None),
                      "confirmed": bool(x and x.get("confirmed") and gid)}
        if not (lu[ha].get("ok") and lu[aa].get("ok")):
            info[g["id"]] = {"skip": "; ".join(lu[x]["why"] for x in (ha, aa) if not lu[x].get("ok"))}
            continue
        # players on the lines page who aren't on the NHL's roster for tonight: the team's best extra skater at
        # the same position goes in; further down, the bet is also tested with each of the next two extras
        repl = []
        for ab in (ha, aa):
            for nm, pid, pos in lu[ab].get("off_roster", []):
                used = {p for p, _ in lu[ab]["skaters"]}
                cands = sorted(((R.skater(p, q)["e5"], p) for p, q in lu[ab].get("extras", []) if q == pos and p not in used), reverse=True)
                if not cands:
                    continue                               # nobody to put in: stays off-roster, lineup waits
                lu[ab]["skaters"] = [(cands[0][1] if p == pid else p, q) for p, q in lu[ab]["skaters"]]
                repl.append((ab, nm, cands[0][1], [c[1] for c in cands[1:3]]))
            tonight = set(lu[ab].get("tonight_ids", []))
            if tonight:
                lu[ab]["on_tonight_roster"] = all(p in tonight for p, _ in lu[ab]["skaters"])
        hname = f"{H['placeName']['default']} {H['commonName']['default']}"
        aname = f"{A['placeName']['default']} {A['commonName']['default']}"
        gg = SimpleNamespace(game_id=g["id"], season=config.LIVE_SEASON, start_utc=g["startTimeUTC"][:19],
                             home_id=H["id"], away_id=A["id"], home_name=hname, away_name=aname,
                             home_abbrev=ha, away_abbrev=aa, home_goals=None, away_goals=None,
                             home_reg=None, away_reg=None, went_ot=None)
        row, pr = R.game_features(gg, date_et, {H["id"]: (lu[ha]["skaters"], gl[ha]["id"]),
                                                 A["id"]: (lu[aa]["skaters"], gl[aa]["id"])})
        if row is None:
            continue
        ids_to_name = {}
        for ab in (ha, aa):
            ids_to_name.update({v[0]: k.title() for k, v in lu[ab]["ids"].items()})
        for p in pr:
            p["name"] = ids_to_name.get(p["player_id"], str(p["player_id"]))
        ev = find_event(events, hname, aname, start)
        # game-time decisions: re-rate the game with each one sitting, his spot taken by the team's
        # extra skater at the same position who plays the most (or an average player if there is none)
        alts = []
        for ab, tid in ((ha, H["id"]), (aa, A["id"])):
            for k, (nm, pid, pos) in enumerate(lu[ab].get("flagged_ids", [])):
                used = {p for p, _ in lu[ab]["skaters"]}
                if pid not in used:
                    continue                               # already swapped out (not on tonight's roster)
                subs = [(R.skater(p, q)["e5"], p) for p, q in lu[ab].get("extras", []) if q == pos and p not in used]
                sub = max(subs)[1] if subs else -(1000 + k)
                sk = [(sub if p == pid else p, q) for p, q in lu[ab]["skaters"]]
                ups = {H["id"]: (lu[ha]["skaters"], gl[ha]["id"]), A["id"]: (lu[aa]["skaters"], gl[aa]["id"])}
                ups[tid] = (sk, gl[ab]["id"])
                arow, apr = R.game_features(gg, date_et, ups)
                if arow is not None:
                    sub_name = ids_to_name.get(sub, "an average depth player")
                    alts.append({"team": ab, "name": nm, "sub": sub_name, "row": arow, "props": apr})
        # a swapped-in extra: also rate the game with each of the next extras in his place instead
        for ab, nm, first, others in repl:
            tid = H["id"] if ab == ha else A["id"]
            for c in others:
                sk = [(c if p == first else p, q) for p, q in lu[ab]["skaters"]]
                ups = {H["id"]: (lu[ha]["skaters"], gl[ha]["id"]), A["id"]: (lu[aa]["skaters"], gl[aa]["id"])}
                ups[tid] = (sk, gl[ab]["id"])
                arow, apr = R.game_features(gg, date_et, ups)
                if arow is not None:
                    alts.append({"team": ab, "name": nm, "key": f"{nm}|{c}", "kind": "roster", "sub": ids_to_name.get(c, str(c)),
                                 "first": ids_to_name.get(first, str(first)), "row": arow, "props": apr})
        # unconfirmed starting goalies: re-rate the game with the other goalie in net, for every mix of
        # the unconfirmed teams; a game bet that qualifies in all of them doesn't need to wait
        unconf = [(ab, tid) for ab, tid in ((ha, H["id"]), (aa, A["id"])) if not gl[ab]["confirmed"]]
        other = {}
        for ab, _ in unconf:
            o = [(n, x) for n, x in lu[ab].get("goalie_ids", []) if x != gl[ab]["id"]]
            other[ab] = o[0] if o and gl[ab]["id"] else None
        galts = []
        if unconf and all(other[ab] for ab, _ in unconf):
            for n in range(1, len(unconf) + 1):
                for combo in itertools.combinations(unconf, n):
                    ups = {H["id"]: (lu[ha]["skaters"], gl[ha]["id"]), A["id"]: (lu[aa]["skaters"], gl[aa]["id"])}
                    for ab, tid in combo:
                        ups[tid] = (lu[ab]["skaters"], other[ab][1])
                    arow, apr = R.game_features(gg, date_et, ups)
                    if arow is not None:
                        galts.append({"teams": tuple(ab for ab, _ in combo), "row": arow, "props": apr})
        if len(galts) != 2 ** len(unconf) - 1:
            galts = None                                   # couldn't test every case: wait for confirmation
        info[g["id"]] = {"start": start, "lu": lu, "gl": gl, "ev": ev, "names": ids_to_name, "alts": alts,
                         "galts": galts, "gother": other,
                         "repl": [(ab, nm, ids_to_name.get(first, str(first))) for ab, nm, first, _ in repl]}
        rows.append(row)
        props.extend(pr)

    tg = pd.DataFrame(rows)
    pr = pd.DataFrame(props)
    if tg.empty:
        print("nothing rateable yet:", {k: v.get("skip") for k, v in info.items()})
        return
    shop = {gid: (PX.shop_moneyline(i["ev"]), PX.shop_total(i["ev"])) for gid, i in info.items() if i.get("ev")}
    lines = {gid: (t[1]["line"] if t[1] else np.nan) for gid, t in shop.items()}
    gp, pp = model.predict_live(models, tg, pr, lines)
    d = tg.merge(gp, on="game_id")
    d["mkt_p_home"] = d.game_id.map(lambda x: shop[x][0]["fair_home"] if x in shop else np.nan)
    d = S.add_flags(d)

    names = {}
    for i in info.values():
        names.update(i.get("names", {}))
    def game_picks(r, emit):
        """Run every game-level rule on one rated game; emit(...) is called for each pick."""
        ml, tot = shop[r.game_id]
        prior = fh
        net = S.stack_votes(pd.DataFrame([r]), prior).iloc[0]
        for sid, q, k in (("ml_stack3", 0.20, 3), ("ml_stack2", 0.20, 2), ("ml_stack3_25", 0.25, 3)):
            if q != 0.20:
                old = S.TAIL
                S.TAIL = q
                n2 = S.stack_votes(pd.DataFrame([r]), prior).iloc[0]
                ev = S.evidence(r, prior, n2 > 0, names) if abs(n2) >= k else None
                S.TAIL = old
                nn = n2
            else:
                nn = net
                ev = S.evidence(r, prior, nn > 0, names) if abs(nn) >= k else None
            if abs(nn) >= k:
                home = nn > 0
                tier = "★ 4+ flags" if abs(nn) >= 4 else f"{abs(int(nn))} flags"
                emit(r, sid, r.home_abbrev if home else r.away_abbrev, "moneyline", ml["home" if home else "away"],
                     ev, {"tier": tier, "model_prob": round(float(r.p_home if home else 1 - r.p_home), 4),
                          "market_prob": round(float(r.mkt_p_home if home else 1 - r.mkt_p_home), 4)})
        # tracking: the same 3+ stack with the rest flag left out (rest counts every back-to-back,
        # not just a 20% tail, so it may be adding weaker bets)
        n5 = net - np.sign(r.f_rest)
        if abs(n5) >= 3:
            home = n5 > 0
            emit(r, "ml_stack3_norest", r.home_abbrev if home else r.away_abbrev, "moneyline", ml["home" if home else "away"],
                 [e for e in S.evidence(r, prior, home, names) if e["flag"] != "Rest"],
                 {"tier": f"{abs(int(n5))} flags, rest not counted",
                  "model_prob": round(float(r.p_home if home else 1 - r.p_home), 4),
                  "market_prob": round(float(r.mkt_p_home if home else 1 - r.mkt_p_home), 4)})
        # single flags (10% tails), tracking only
        for sid, f in (("ml_flag_lineup", "f_lineup"), ("ml_flag_hotgoalie", "f_hot_goalie")):
            hi, lo = prior[f].quantile(0.9), prior[f].quantile(0.1)
            if r[f] >= hi or r[f] <= lo:
                home = r[f] >= hi
                emit(r, sid, r.home_abbrev if home else r.away_abbrev, "moneyline", ml["home" if home else "away"],
                     S.evidence(r, prior, home, names) or [{"flag": f, "text": f"{f} in its 10% tail"}])
        if r.f_rest != 0:
            home = r.f_rest > 0
            emit(r, "ml_flag_rest", r.home_abbrev if home else r.away_abbrev, "moneyline", ml["home" if home else "away"],
                 [{"flag": "Rest", "text": f"{r.away_abbrev if home else r.home_abbrev} on the second night of a back-to-back"}])
        # model + market moneyline, best price after fee, 5%+
        b = blend["moneyline"]
        ph = float(S.blend(b, np.array([r.mkt_p_home]), np.array([r.p_home]))[0])
        for home, rows_, p in ((True, ml["home"], ph), (False, ml["away"], 1 - ph)):
            if rows_ and p * rows_[0]["after_fee"] - 1 > 0.05:
                pm = r.p_home if home else 1 - r.p_home
                pk = r.mkt_p_home if home else 1 - r.mkt_p_home
                evv = p * rows_[0]["after_fee"] - 1
                emit(r, "ml_blend5", r.home_abbrev if home else r.away_abbrev, "moneyline", rows_,
                     [{"flag": "Model + market", "text": f"Model {pm:.1%}, market {pk:.1%}; mixed = {p:.1%} vs best "
                                                        f"price {rows_[0]['after_fee_american']:+d} after fee, {evv:+.1%} expected"}],
                     {"model_prob": round(float(pm), 4), "market_prob": round(float(pk), 4),
                      "bet_prob": round(p, 4), "ev": round(evv, 4)})
        # totals model (tracking), median price, 3%+
        if tot and not np.isnan(r.p_over):
            for over, rows_ in ((True, tot["over"]), (False, tot["under"])):
                if not rows_:
                    continue
                pw = r.p_over if over else 1 - r.p_over - r.p_push
                med = float(np.median([x["decimal"] for x in rows_]))
                if pw * med + r.p_push - 1 > 0.03:
                    emit(r, "totals_model", f"{'Over' if over else 'Under'} {tot['line']}", "total", rows_,
                         [{"flag": "Goal model", "text": f"Model expects {r.lam_home + r.lam_away:.2f} regulation goals; "
                                                         f"{'over' if over else 'under'} {tot['line']} {pw:.1%}"}],
                         {"line": tot["line"], "model_prob": round(float(pw), 4)})


    # the same rules, run on each "game-time decision sits" version of a game: which picks still qualify
    # and on each "other goalie in net" version: keys (game_id, "goalie", teams)
    holds = {}                                    # (game_id, team, name) -> {(strategy, pick), ...}
    for gid, i in info.items():
        variants = [((gid, a["team"], a.get("key", a["name"])), a) for a in i.get("alts", [])]
        variants += [((gid, "goalie", a["teams"]), a) for a in (i.get("galts") or [])]
        for key, alt in variants:
            ga, pa = model.predict_live(models, pd.DataFrame([alt["row"]]), pd.DataFrame(alt["props"]), lines)
            da = pd.DataFrame([alt["row"]]).merge(ga, on="game_id")
            da["mkt_p_home"] = shop[gid][0]["fair_home"] if gid in shop else np.nan
            da = S.add_flags(da)
            got = set()
            if gid in shop:
                game_picks(da.iloc[0], lambda r, sid, pick, *a, **k: got.add((sid, pick)))
            holds[key] = got

    picks = []

    def base(r, strategy, pick, market, side_rows, evidence, extra=None):
        i = info[r.game_id]
        start = i["start"]
        lu, gl = i["lu"], i["gl"]
        ha, aa = r.home_abbrev, r.away_abbrev
        unconf = [ab for ab in (ha, aa) if not gl[ab]["confirmed"]]
        goalie_ok = not unconf
        waiting, notes = [], []
        if unconf and market != "shots on goal" and i.get("galts"):
            goalie_ok = all((strategy, pick) in holds.get((r.game_id, "goalie", a["teams"]), set()) for a in i["galts"])
            if goalie_ok:
                notes.append("holds whichever goalie starts: " + "; ".join(
                    f"{ab} {gl[ab]['name']} or {i['gother'][ab][0]}" for ab in unconf))
        checks = {
            "goalies": goalie_ok,
            "lineups": all(LU.lineup_ok(lu[x]) for x in (ha, aa)),
            "prices": len(side_rows) >= 2,
            "time": (start - now).total_seconds() / 60 >= MIN_MINUTES,
        }
        for ab in (ha, aa):
            if not gl[ab]["confirmed"] and not goalie_ok:
                waiting.append(f"{ab} goalie not confirmed (projected {gl[ab]['name'] or 'unknown'})")
            if not lu[ab]["fresh"]:
                hrs = lu[ab]["hours_old"]
                age = f"last updated {hrs:.0f}h before puck drop" if hrs else "no update time"
                if LU.lineup_ok(lu[ab]):
                    notes.append(f"{ab} lines page {age}; trusted: all 18 on tonight's NHL roster "
                                 f"(goalie and game-time decisions tested separately)")
                else:
                    why = []
                    if hrs and hrs > 96:
                        why.append("older than 4 days")
                    if not lu[ab]["on_tonight_roster"]:
                        why.append("NHL roster for tonight not posted yet" if not lu[ab].get("tonight_ids")
                                   else "not all 18 on tonight's NHL roster")
                    waiting.append(f"{ab} lines not updated today ({age}" + (f"; {'; '.join(why)}" if why else "") + ")")
            if lu[ab]["missing"]:
                waiting.append(f"{ab} players not matched to NHL records: {', '.join(lu[ab]['missing'])}")
            elif not lu[ab]["complete"]:
                f_, d_ = lu[ab]["n_listed"]
                waiting.append(f"{ab} lineup incomplete ({f_} forwards, {d_} defense listed)")
        # game-time decisions in either lineup
        checks["gtd"] = True
        for ab, nm, first in i.get("repl", []):
            notes.append(f"{nm} ({ab}) not on tonight's NHL roster; {first} in for him")
        for alt in i.get("alts", []):
            if alt.get("kind") == "roster":            # who replaces an off-roster player
                if market == "shots on goal":
                    continue
                if (strategy, pick) in holds.get((r.game_id, alt["team"], alt["key"]), set()):
                    notes.append(f"holds with {alt['sub']} in for {alt['name']} instead of {alt['first']}")
                else:
                    checks["gtd"] = False
                    waiting.append(f"waiting on who replaces {alt['name']} ({alt['team']}): the bet depends on it")
                continue
            who = f"{alt['name']} ({alt['team']}, game-time decision)"
            if market == "shots on goal":
                if (extra or {}).get("player") and norm_name(extra["player"]) == norm_name(alt["name"]):
                    checks["gtd"] = False
                    waiting.append(f"waiting on {who}: no prop bet until he's confirmed in")
            elif (strategy, pick) in holds.get((r.game_id, alt["team"], alt["name"]), set()):
                notes.append(f"holds with or without {alt['name']} ({alt['sub']} in for him)")
            else:
                checks["gtd"] = False
                waiting.append(f"waiting on {who}: this bet only qualifies if he plays")
        if not checks["prices"]:
            waiting.append("fewer than 2 books priced")
        if not checks["time"]:
            waiting.append("too close to puck drop")
        checks["waiting_on"] = "; ".join(waiting)
        checks["lineup_note"] = "; ".join(notes)
        checks["goalies_named"] = {ab: gl[ab]["name"] for ab in (ha, aa)}
        stake = STAKE.get(strategy, 0.0)
        plan, left = PX.fill_plan(side_rows, stake or 1.0)
        best = side_rows[0] if side_rows else None
        p = {"strategy_id": strategy, "source": "live", "game_id": int(r.game_id), "game_date": date_et,
             "start_utc": start.isoformat(), "season": config.LIVE_SEASON, "home": ha, "away": aa,
             "market": market, "pick": pick, "stake_units": stake,
             "evidence": list(evidence) + [{"flag": "Lineup check", "text": n} for n in notes], "checks": checks,
             "status": "ready" if all(checks[k] for k in ("goalies", "lineups", "prices", "time", "gtd")) else "waiting",
             "prices": side_rows[:8], "priced_at": now.isoformat(), "fill_plan": plan,
             "best_book": best["name"] if best else None, "best_american": best["american"] if best else None,
             "best_decimal": best["after_fee"] if best else None,
             "median_decimal": float(np.median([x["decimal"] for x in side_rows])) if side_rows else None}
        p.update(extra or {})
        picks.append(p)

    for _, r in d.iterrows():
        if r.game_id in shop:
            game_picks(r, base)

    # ---------------- shots on goal props
    if len(pp):
        pp = pp.merge(pr[["game_id", "player_id", "name"]], on=["game_id", "player_id"])
        for gid, grp in pp.groupby("game_id"):
            ev = info[gid].get("ev")
            if not ev or (info[gid]["start"] - now).total_seconds() > 3 * 3600:
                continue                          # props only pulled within 3 hours of puck drop
            try:
                pe = odds_call(f"/sports/icehockey_nhl/events/{ev['id']}/odds", {"markets": "player_shots_on_goal"})
            except Exception as e:
                print("  props pull failed:", e)
                continue
            lines_ = {norm_name(k): v for k, v in PX.shop_props(pe).items()}
            r = d[d.game_id == gid].iloc[0]
            dup = grp.name.map(norm_name).value_counts()
            for x in grp.itertuples():
                if dup.get(norm_name(x.name), 0) > 1:
                    continue                      # two players share this name; the odds can't tell them apart
                q = lines_.get(norm_name(x.name))
                if not q or q["fair_over"] is None:
                    continue
                po, ppush = nb_over(x.mu_sog, x.alpha, q["line"])
                pmod = min(max(po / (1 - ppush), 1e-4), 1 - 1e-4)
                pb = float(S.blend(blend["sog"], np.array([q["fair_over"]]), np.array([pmod]))[0]) * (1 - ppush)
                for over, rows_ in ((True, q["over"]), (False, q["under"])):
                    if len(rows_) < 1:
                        continue
                    pw = pb if over else 1 - pb - ppush
                    med = float(np.median([y["decimal"] for y in rows_]))
                    txt = (f"Model expects {x.mu_sog:.2f} shots; market {'over' if over else 'under'} "
                           f"{(q['fair_over'] if over else 1 - q['fair_over']):.1%}, mixed {pw:.1%}")
                    pick = f"{x.name} {'Over' if over else 'Under'} {q['line']}"
                    extra = {"player": x.name, "line": q["line"], "bet_prob": round(pw, 4)}
                    if pw * med + ppush - 1 > 0.05:
                        base(r, "sog_blend5", pick, "shots on goal", rows_, [{"flag": "Model + market", "text": txt}],
                             dict(extra, ev=round(pw * med + ppush - 1, 4)))
                    if pw * rows_[0]["after_fee"] + ppush - 1 > 0.03:
                        base(r, "sog_blend_best3", pick, "shots on goal", rows_, [{"flag": "Model + market", "text": txt}],
                             dict(extra, ev=round(pw * rows_[0]["after_fee"] + ppush - 1, 4)))

    if a.dry or not os.environ.get("GITHUB_ACTIONS"):
        print_board(picks, info)            # picks only print on your own machine, never in public logs
    else:
        n_ready = sum(p["status"] == "ready" for p in picks)
        print(f"{len(picks)} picks updated ({n_ready} ready); see the dashboard")
    if a.dry:
        return
    write(picks, date_et, [int(g["id"]) for g in upcoming])


def print_board(picks, info):
    for p in sorted(picks, key=lambda p: (STAKE.get(p["strategy_id"], 0) == 0, p["start_utc"])):
        tab = "BET  " if STAKE.get(p["strategy_id"]) else "track"
        plan = "; ".join(f"${x['risk']:.2f} at {x['name']} {x['american']:+d}" for x in p["fill_plan"])
        print(f"[{tab}] {p['status'].upper():7s} {p['away']}@{p['home']} {p['strategy_id']:16s} {p['pick']:28s} "
              f"{plan} | {p['checks']['waiting_on'] or 'all checks passed'}")
    for gid, i in info.items():
        if i.get("skip"):
            print(f"  skipped game {gid}: {i['skip']}")


def write(picks, date_et, upcoming_ids):
    import supa
    existing = supa.select("nhl_picks", {"select": "id,strategy_id,game_id,pick,status,best_decimal,best_book,"
                                                   "best_american,fill_plan,checks,start_utc", "source": "eq.live",
                                         "game_date": f"eq.{date_et}"})
    key = lambda p: (p["strategy_id"], int(p["game_id"]), p["pick"])
    old = {key(p): p for p in existing}
    out, alerts = [], []
    for p in picks:
        o = old.get(key(p))
        if o and o["status"] in ("ready", "placed") and o.get("best_decimal"):
            # the price we said to bet at stays as it was when the pick first turned ready;
            # the latest price is kept as the closing price
            p["close_decimal"] = p["best_decimal"]
            # what to bet right now: the card shows this, the record keeps the price from when it turned ready
            p["checks"]["now"] = {"book": p["best_book"], "american": p["best_american"],
                                  "decimal": p["best_decimal"], "plan": p["fill_plan"], "at": p["priced_at"]}
            for k in ("best_decimal", "best_book", "best_american", "fill_plan"):
                p[k] = o[k]
            if o["status"] == "placed":
                p["status"] = "placed"
            else:
                p["status"] = "ready"
            # after it turned ready, a goalie, lineup or game-time-decision check now fails (late scratch,
            # goalie switch): it may already be bet, so it stays, but the card says so
            before = (o.get("checks") or {}).get("late_change")
            if not all(p["checks"].get(k) for k in ("goalies", "lineups", "gtd")):
                p["checks"]["late_change"] = p["checks"].get("waiting_on") or "a goalie or lineup check now fails"
                if not before:
                    alerts.append(f"{p['away']} at {p['home']}, {p['pick']}: {p['checks']['late_change']}")
            elif before:
                p["checks"]["late_change"] = before
        p["updated_at"] = datetime.now(timezone.utc).isoformat()
        out.append(p)
    cols = set().union(*[p.keys() for p in out]) if out else set()
    out = [{c: p.get(c) for c in cols} for p in out]
    if out:
        supa.upsert("nhl_picks", out, "strategy_id,source,game_id,pick")
    # picks that stopped qualifying (a goalie or lineup changed): marked void, nothing else touched.
    # A ready pick stays ready because it may already have been bet.
    new_keys = {key(p) for p in picks}
    gone = [{"id": o["id"], "status": "void"} for k, o in old.items()
            if k not in new_keys and o["game_id"] in upcoming_ids and o["status"] == "waiting"]
    for g in gone:
        supa.patch("nhl_picks", {"id": f"eq.{g['id']}"}, {"status": "void", "updated_at": datetime.now(timezone.utc).isoformat()})
    # a ready or placed pick that no longer comes up at all before puck drop (goalie, lineup or price changed)
    for k, o in old.items():
        ck = o.get("checks") or {}
        if (k not in new_keys and o["game_id"] in upcoming_ids and o["status"] in ("ready", "placed")
                and not ck.get("late_change")):
            ck["late_change"] = "no longer qualifies on the latest goalies, lineups or prices"
            supa.patch("nhl_picks", {"id": f"eq.{o['id']}"}, {"checks": ck, "updated_at": datetime.now(timezone.utc).isoformat()})
            alerts.append(f"{o['pick']} ({k[0]}): {ck['late_change']}")
    print(f"wrote {len(out)} picks, voided {len(gone)}, late changes {len(alerts)}")



if __name__ == "__main__":
    main()
