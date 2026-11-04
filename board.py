"""Tonight's board. Safe to run as often as you like: every run re-pulls lineups, goalies and
prices, re-rates each game, and updates the picks in Supabase.

A pick is READY only when all four checks pass:
  goalies  both starters marked Confirmed on DailyFaceoff
  lineups  both teams' lines updated within 14 hours of puck drop and fully matched to NHL ids
  prices   at least 2 of the books priced the side
  time     puck drop is at least 10 minutes away
Otherwise it is WAITING and says what it is waiting on.

Usage: ODDS_API_KEY=... SUPABASE_SERVICE_KEY=... python board.py [--date YYYY-MM-DD] [--dry]
"""
import argparse
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
        lu = {}
        for ab in (ha, aa):
            try:
                lu[ab] = LU.team_lineup(ab, start)
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
        info[g["id"]] = {"start": start, "lu": lu, "gl": gl, "ev": ev, "names": ids_to_name}
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

    picks = []

    def base(r, strategy, pick, market, side_rows, evidence, extra=None):
        i = info[r.game_id]
        start = i["start"]
        lu, gl = i["lu"], i["gl"]
        ha, aa = r.home_abbrev, r.away_abbrev
        checks = {
            "goalies": gl[ha]["confirmed"] and gl[aa]["confirmed"],
            "lineups": all(lu[x]["fresh"] and lu[x]["complete"] for x in (ha, aa)),
            "prices": len(side_rows) >= 2,
            "time": (start - now).total_seconds() / 60 >= MIN_MINUTES,
        }
        waiting = []
        for ab in (ha, aa):
            if not gl[ab]["confirmed"]:
                waiting.append(f"{ab} goalie not confirmed (projected {gl[ab]['name'] or 'unknown'})")
            if not lu[ab]["fresh"]:
                hrs = lu[ab]["hours_old"]
                waiting.append(f"{ab} lines not updated today" + (f" (last {hrs:.0f}h before puck drop)" if hrs else ""))
            if lu[ab]["missing"]:
                waiting.append(f"{ab} players not matched to NHL records: {', '.join(lu[ab]['missing'])}")
            elif not lu[ab]["complete"]:
                f_, d_ = lu[ab]["n_listed"]
                waiting.append(f"{ab} lineup incomplete ({f_} forwards, {d_} defense listed)")
        if not checks["prices"]:
            waiting.append("fewer than 2 books priced")
        if not checks["time"]:
            waiting.append("too close to puck drop")
        checks["waiting_on"] = "; ".join(waiting)
        checks["goalies_named"] = {ab: gl[ab]["name"] for ab in (ha, aa)}
        stake = STAKE.get(strategy, 0.0)
        plan, left = PX.fill_plan(side_rows, stake or 1.0)
        best = side_rows[0] if side_rows else None
        p = {"strategy_id": strategy, "source": "live", "game_id": int(r.game_id), "game_date": date_et,
             "start_utc": start.isoformat(), "season": config.LIVE_SEASON, "home": ha, "away": aa,
             "market": market, "pick": pick, "stake_units": stake, "evidence": evidence, "checks": checks,
             "status": "ready" if all(checks[k] for k in ("goalies", "lineups", "prices", "time")) else "waiting",
             "prices": side_rows[:8], "priced_at": now.isoformat(), "fill_plan": plan,
             "best_book": best["name"] if best else None, "best_american": best["american"] if best else None,
             "best_decimal": best["after_fee"] if best else None,
             "median_decimal": float(np.median([x["decimal"] for x in side_rows])) if side_rows else None}
        p.update(extra or {})
        picks.append(p)

    names = {}
    for i in info.values():
        names.update(i.get("names", {}))
    for _, r in d.iterrows():
        if r.game_id not in shop:
            continue
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
                base(r, sid, r.home_abbrev if home else r.away_abbrev, "moneyline", ml["home" if home else "away"],
                     ev, {"tier": tier, "model_prob": round(float(r.p_home if home else 1 - r.p_home), 4),
                          "market_prob": round(float(r.mkt_p_home if home else 1 - r.mkt_p_home), 4)})
        # single flags (10% tails), tracking only
        for sid, f in (("ml_flag_lineup", "f_lineup"), ("ml_flag_hotgoalie", "f_hot_goalie")):
            hi, lo = prior[f].quantile(0.9), prior[f].quantile(0.1)
            if r[f] >= hi or r[f] <= lo:
                home = r[f] >= hi
                base(r, sid, r.home_abbrev if home else r.away_abbrev, "moneyline", ml["home" if home else "away"],
                     S.evidence(r, prior, home, names) or [{"flag": f, "text": f"{f} in its 10% tail"}])
        if r.f_rest != 0:
            home = r.f_rest > 0
            base(r, "ml_flag_rest", r.home_abbrev if home else r.away_abbrev, "moneyline", ml["home" if home else "away"],
                 [{"flag": "Rest", "text": f"{r.away_abbrev if home else r.home_abbrev} on the second night of a back-to-back"}])
        # model + market moneyline, best price after fee, 5%+
        b = blend["moneyline"]
        ph = float(S.blend(b, np.array([r.mkt_p_home]), np.array([r.p_home]))[0])
        for home, rows_, p in ((True, ml["home"], ph), (False, ml["away"], 1 - ph)):
            if rows_ and p * rows_[0]["after_fee"] - 1 > 0.05:
                pm = r.p_home if home else 1 - r.p_home
                pk = r.mkt_p_home if home else 1 - r.mkt_p_home
                evv = p * rows_[0]["after_fee"] - 1
                base(r, "ml_blend5", r.home_abbrev if home else r.away_abbrev, "moneyline", rows_,
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
                    base(r, "totals_model", f"{'Over' if over else 'Under'} {tot['line']}", "total", rows_,
                         [{"flag": "Goal model", "text": f"Model expects {r.lam_home + r.lam_away:.2f} regulation goals; "
                                                         f"{'over' if over else 'under'} {tot['line']} {pw:.1%}"}],
                         {"line": tot["line"], "model_prob": round(float(pw), 4)})

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
                                                   "best_american,fill_plan", "source": "eq.live",
                                         "game_date": f"eq.{date_et}"})
    key = lambda p: (p["strategy_id"], int(p["game_id"]), p["pick"])
    old = {key(p): p for p in existing}
    out = []
    for p in picks:
        o = old.get(key(p))
        if o and o["status"] in ("ready", "placed") and o.get("best_decimal"):
            # the price we said to bet at stays as it was when the pick first turned ready;
            # the latest price is kept as the closing price
            p["close_decimal"] = p["best_decimal"]
            for k in ("best_decimal", "best_book", "best_american", "fill_plan"):
                p[k] = o[k]
            if o["status"] == "placed":
                p["status"] = "placed"
            else:
                p["status"] = "ready"
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
    print(f"wrote {len(out)} picks, voided {len(gone)}")


if __name__ == "__main__":
    main()
