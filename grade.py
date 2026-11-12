"""Grade finished games: result, profit at the price the board told you to bet, closing price.

Only picks that turned READY count (that is when you would have bet). Picks still WAITING at
puck drop never cleared their checks, so they are marked void and left out of the record.
Profit is in units: a win pays the full stake (the fill plan was sized to win exactly that),
a loss costs what the fill plan risked.
CLV = price you got / closing price - 1 (after fees, best book both times).

Usage: SUPABASE_SERVICE_KEY=... python grade.py
"""
from datetime import datetime, timedelta, timezone

import lineups as LU
import pricing as PX
import supa
from parse import norm_name


def final(game_id):
    b = LU.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore", as_json=True)
    if b.get("gameState") not in ("OFF", "FINAL"):
        return None
    sog = {}
    for side in ("homeTeam", "awayTeam"):
        for grp in ("forwards", "defense"):
            for p in b.get("playerByGameStats", {}).get(side, {}).get(grp, []):
                full = p.get("name", {}).get("default", "")
                sog[p["playerId"]] = p.get("sog", p.get("shots", 0)) or 0
                sog[norm_name(full)] = sog[p["playerId"]]
    roster = LU.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play", as_json=True).get("rosterSpots", [])
    for r in roster:
        n = norm_name(f"{r['firstName']['default']} {r['lastName']['default']}")
        if r["playerId"] in sog:
            sog[n] = sog[r["playerId"]]
    starters = {}
    for side, ab in (("homeTeam", b["homeTeam"]["abbrev"]), ("awayTeam", b["awayTeam"]["abbrev"])):
        for g in b.get("playerByGameStats", {}).get(side, {}).get("goalies", []):
            if g.get("starter"):
                starters[ab] = norm_name(g.get("name", {}).get("default", ""))
    return {"home": b["homeTeam"]["score"], "away": b["awayTeam"]["score"], "starters": starters,
            "home_ab": b["homeTeam"]["abbrev"], "away_ab": b["awayTeam"]["abbrev"], "sog": sog}


def main():
    today = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%d")
    open_ = supa.select("nhl_picks", {"select": "*", "source": "eq.live", "result": "is.null",
                                      "status": "in.(waiting,ready,placed)", "game_date": f"lte.{today}"})
    by_game = {}
    for p in open_:
        by_game.setdefault(p["game_id"], []).append(p)
    n = 0
    for gid, ps in by_game.items():
        f = final(gid)
        if not f:
            continue
        for p in ps:
            upd = {"updated_at": datetime.now(timezone.utc).isoformat()}
            if p["status"] == "waiting":
                upd["status"] = "void"
                supa.patch("nhl_picks", {"id": f"eq.{p['id']}"}, upd)
                continue
            if p["market"] == "moneyline":
                winner = f["home_ab"] if f["home"] > f["away"] else f["away_ab"]
                res = "W" if p["pick"] == winner else "L"
            elif p["market"] == "total":
                side, line = p["pick"].split()[0], float(p["line"])
                tot = f["home"] + f["away"]
                res = "P" if tot == line else ("W" if (tot > line) == (side == "Over") else "L")
            else:
                got = f["sog"].get(norm_name(p.get("player") or ""))
                if got is None:
                    upd.update(status="void")
                    supa.patch("nhl_picks", {"id": f"eq.{p['id']}"}, upd)
                    continue
                line, over = float(p["line"]), " Over " in f" {p['pick']} "
                res = "P" if got == line else ("W" if (got > line) == over else "L")
            stake = float(p["stake_units"] or 0) or 1.0
            risk_u = sum(x["risk"] for x in (p.get("fill_plan") or [])) / PX.UNIT_TO_WIN
            if not p.get("fill_plan"):
                risk_u = stake / ((p["best_decimal"] or 2.0) - 1)
            profit = stake if res == "W" else (0.0 if res == "P" else -risk_u)
            upd.update(status="graded", result=res, profit_units=round(profit, 4))
            eg = (p.get("checks") or {}).get("early_goalies")
            if eg and f.get("starters"):       # early test: did the projected goalies both start?
                def same(a, b):                # boxscore gives "S. Cossa"; projection gives "Sebastian Cossa"
                    a, b = norm_name(a or ""), b or ""
                    return bool(a and b) and a.split()[-1] == b.split()[-1] and a[:1] == b[:1]
                upd["checks"] = dict(p["checks"], goalies_right=all(same(eg.get(ab), f["starters"].get(ab)) for ab in eg))
            if p.get("close_decimal") and p.get("best_decimal"):
                upd["clv"] = round(float(p["best_decimal"]) / float(p["close_decimal"]) - 1, 4)
            supa.patch("nhl_picks", {"id": f"eq.{p['id']}"}, upd)
            n += 1
    print(f"graded {n} picks")


if __name__ == "__main__":
    main()
