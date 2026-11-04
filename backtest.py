"""Walk-forward backtest against near-closing prices. Writes output/report.md.

Prices: for each side we bet at the MEDIAN price across US books (what a normal account
gets), and also report the BEST price (line shopping). The market's own probability is the
median of each book's no-vig probability.

Answers, per market (moneyline, totals, shots-on-goal props):
  1. Does the model carry information the market doesn't?  (outcome ~ market + model)
  2. Flat-stake ROI on bets where expected value > threshold, by season, with a
     game-clustered bootstrap CI, judged against the fixed gate in config.GATE.
  3. Tails (Caden's stack method): results in the extreme 10% of each flag and in games
     where several flags stack on one side, priced against the market.
"""
import json
import unicodedata
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
import jsonio
import model
from parse import TABLES, norm_name

OUT = config.OUT
RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ odds helpers
def dec(american):
    a = float(american)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def novig(p_a, p_b):
    s = p_a + p_b
    return p_a / s


def last_token(name):
    t = norm_name(name).split()
    return t[-1] if t else ""


def game_snapshot_path(start_utc):
    return config.RAW_ODDS / "games" / f"{start_utc[:19].replace(':', '')}.json.gz"


def market_for_game(g):
    """Return dict with moneyline + totals market info for one NHL game, or None."""
    f = game_snapshot_path(g.start_utc)
    if not jsonio.exists(f):
        return None
    data = jsonio.load(f).get("data", [])
    hk, ak = last_token(g.home_name), last_token(g.away_name)
    ev = next((e for e in data if last_token(e["home_team"]) == hk and last_token(e["away_team"]) == ak), None)
    if ev is None:
        return None
    ml_fair, ml_h, ml_a, tots = [], [], [], []
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            o = {x["name"]: x for x in mk.get("outcomes", [])}
            if mk["key"] == "h2h" and ev["home_team"] in o and ev["away_team"] in o:
                ph, pa = 1 / dec(o[ev["home_team"]]["price"]), 1 / dec(o[ev["away_team"]]["price"])
                ml_fair.append(novig(ph, pa))
                ml_h.append(dec(o[ev["home_team"]]["price"]))
                ml_a.append(dec(o[ev["away_team"]]["price"]))
            if mk["key"] == "totals" and "Over" in o and "Under" in o:
                tots.append((o["Over"]["point"], dec(o["Over"]["price"]), dec(o["Under"]["price"])))
    if not ml_fair:
        return None
    res = {"event_id": ev["id"], "mkt_p_home": float(np.median(ml_fair)),
           "home_med": float(np.median(ml_h)), "away_med": float(np.median(ml_a)),
           "home_best": float(np.max(ml_h)), "away_best": float(np.max(ml_a)), "n_books": len(ml_fair)}
    if tots:
        pts = pd.Series([t[0] for t in tots])
        L = float(pts.mode().iloc[0])
        at = [t for t in tots if t[0] == L]
        res.update({"total_line": L,
                    "mkt_p_over": float(np.median([novig(1 / o, 1 / u) for _, o, u in at])),
                    "over_med": float(np.median([o for _, o, _ in at])),
                    "under_med": float(np.median([u for _, _, u in at])),
                    "over_best": float(max(o for _, o, _ in at)),
                    "under_best": float(max(u for _, _, u in at))})
    return res


def props_for_event(event_id):
    f = config.RAW_ODDS / "props" / f"{event_id}.json.gz"
    if not jsonio.exists(f):
        return {}
    ev = jsonio.load(f).get("data") or {}
    by = {}
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] != "player_shots_on_goal":
                continue
            per = {}
            for o in mk.get("outcomes", []):
                per.setdefault((o.get("description"), o.get("point")), {})[o["name"]] = dec(o["price"])
            for (who, pt), d in per.items():
                if "Over" in d and "Under" in d and who:
                    by.setdefault(norm_name(who), []).append((pt, d["Over"], d["Under"]))
    out = {}
    for who, rows in by.items():
        L = float(pd.Series([r[0] for r in rows]).mode().iloc[0])
        at = [r for r in rows if r[0] == L]
        out[who] = {"line": L, "mkt_p_over": float(np.median([novig(1 / o, 1 / u) for _, o, u in at])),
                    "over_med": float(np.median([o for _, o, _ in at])),
                    "under_med": float(np.median([u for _, _, u in at])),
                    "over_best": float(max(o for _, o, _ in at)),
                    "under_best": float(max(u for _, _, u in at)), "n_books": len(at)}
    return out


# ------------------------------------------------------------------ stats helpers
def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def info_test(y, p_mkt, p_mod):
    """Logistic regression y ~ a + b*logit(market) + c*logit(model). IRLS with SEs.
    c > 0 with |t| > 2 means the model knows something the price does not."""
    X = np.column_stack([np.ones(len(y)), logit(p_mkt), logit(p_mod)])
    beta = np.zeros(3)
    for _ in range(50):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return {"b_market": beta[1], "b_model": beta[2], "se_model": se[2], "t_model": beta[2] / se[2]}


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def boot_roi(bets, level):
    if len(bets) == 0:
        return (np.nan, np.nan)
    g = bets.groupby("game_id")["profit"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = RNG.integers(0, len(g), size=(4000, len(g)))
    roi = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.quantile(roi, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def gate(bets):
    G = config.GATE
    n = len(bets)
    roi = bets.profit.sum() / n if n else np.nan
    lo, hi = boot_roi(bets, G["ci_level"])
    by = bets.groupby("season").profit.mean() if n else pd.Series(dtype=float)
    pos = int((by > 0).sum())
    ok = n >= G["min_bets"] and roi > G["min_roi"] and lo > 0 and pos >= G["min_seasons_positive"]
    return {"bets": n, "roi": roi, "ci_lo": lo, "ci_hi": hi, "seasons_positive": pos,
            "by_season": by.to_dict(), "PASS": bool(ok)}


# ------------------------------------------------------------------ bet builders
def settle_two_way(p_a, p_push, dec_a, dec_b, won_a, push, thr, label_a, label_b):
    """Pick the side with the bigger EV if > thr. Returns (side, ev, profit) or None."""
    p_b = 1 - p_a - p_push
    ev_a = p_a * dec_a + p_push - 1
    ev_b = p_b * dec_b + p_push - 1
    if max(ev_a, ev_b) <= thr:
        return None
    if ev_a >= ev_b:
        return label_a, ev_a, (0.0 if push else (dec_a - 1 if won_a else -1.0))
    return label_b, ev_b, (0.0 if push else (dec_b - 1 if (not won_a) else -1.0))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    tg = pd.read_parquet(TABLES / "team_games.parquet")
    pr = pd.read_parquet(TABLES / "prop_rows.parquet")
    bt = tg[tg.season.isin(config.BACKTEST_SEASONS)]

    mk = {}
    for g in bt.itertuples():
        m = market_for_game(g)
        if m:
            mk[g.game_id] = m
    lines = {gid: m.get("total_line", np.nan) for gid, m in mk.items()}
    preds = model.predict_games(tg, lines)
    if preds.empty or not mk:
        print(f"not enough data to backtest: {len(preds)} model predictions, {len(mk)} games with odds")
        return {}
    G = bt.merge(preds, on="game_id").merge(pd.DataFrame.from_dict(mk, orient="index")
                                             .rename_axis("game_id").reset_index(), on="game_id")
    G["home_won"] = (G.home_goals > G.away_goals).astype(int)
    G["final_total"] = G.home_goals + G.away_goals
    rep = {"coverage": {"backtest_games": len(bt), "games_with_odds": len(G)}}
    lines_md = []

    # ---------------- moneyline
    ml = {}
    if len(G):
        ml["info"] = info_test(G.home_won.values, G.mkt_p_home.values, G.p_home.values)
        ml["logloss"] = {"market": logloss(G.home_won.values, G.mkt_p_home.values),
                         "model": logloss(G.home_won.values, G.p_home.values),
                         "blend": logloss(G.home_won.values, 0.5 * G.mkt_p_home + 0.5 * G.p_home)}
        for price in ("med", "best"):
            for thr in config.EDGE_THRESHOLDS:
                rows = []
                for r in G.itertuples():
                    s = settle_two_way(r.p_home, 0.0, getattr(r, f"home_{price}"), getattr(r, f"away_{price}"),
                                       r.home_won == 1, False, thr, "home", "away")
                    if s:
                        rows.append({"game_id": r.game_id, "season": r.season, "side": s[0],
                                     "ev": s[1], "profit": s[2]})
                b = pd.DataFrame(rows, columns=["game_id", "season", "side", "ev", "profit"])
                ml[f"{price}@{thr}"] = gate(b)
                if price == "med" and thr == config.HEADLINE_EDGE:
                    b.to_csv(OUT / "bets_moneyline.csv", index=False)
    rep["moneyline"] = ml

    # ---------------- totals
    tt = {}
    T = G.dropna(subset=["total_line", "p_over"]) if "total_line" in G else G.iloc[0:0]
    if len(T):
        nonpush = T[T.final_total != T.total_line]
        y = (nonpush.final_total > nonpush.total_line).astype(int).values
        p_mod = (nonpush.p_over / (1 - nonpush.p_push)).values
        tt["info"] = info_test(y, nonpush.mkt_p_over.values, p_mod)
        tt["logloss"] = {"market": logloss(y, nonpush.mkt_p_over.values), "model": logloss(y, p_mod)}
        for price in ("med", "best"):
            for thr in config.EDGE_THRESHOLDS:
                rows = []
                for r in T.itertuples():
                    push = r.final_total == r.total_line
                    s = settle_two_way(r.p_over, r.p_push, getattr(r, f"over_{price}"), getattr(r, f"under_{price}"),
                                       r.final_total > r.total_line, push, thr, "over", "under")
                    if s:
                        rows.append({"game_id": r.game_id, "season": r.season, "side": s[0],
                                     "ev": s[1], "profit": s[2]})
                b = pd.DataFrame(rows, columns=["game_id", "season", "side", "ev", "profit"])
                tt[f"{price}@{thr}"] = gate(b)
                if price == "med" and thr == config.HEADLINE_EDGE:
                    b.to_csv(OUT / "bets_totals.csv", index=False)
    rep["totals"] = tt

    # ---------------- shots-on-goal props
    pp = {}
    P = model.predict_props(pr)
    if len(P):
        P = P.merge(pr[["game_id", "player_id", "name", "season", "sog"]], on=["game_id", "player_id"])
        P = P[P.season.isin(config.BACKTEST_SEASONS)]
        ev_of_game = {gid: m["event_id"] for gid, m in mk.items()}
        rows = []
        for gid, grp in P.groupby("game_id"):
            if gid not in ev_of_game:
                continue
            lines_ = props_for_event(ev_of_game[gid])
            if not lines_:
                continue
            for r in grp.itertuples():
                q = lines_.get(norm_name(r.name))
                if not q:
                    continue
                po, ppush = model.p_over_nb(r.mu_sog, r.alpha, q["line"])
                rows.append({"game_id": gid, "season": r.season, "player_id": r.player_id, "name": r.name,
                             "sog": r.sog, "mu": r.mu_sog, "p_over": po, "p_push": ppush, **q})
        PB = pd.DataFrame(rows)
        rep["coverage"]["prop_player_games_priced"] = len(PB)
        if len(PB):
            np_ = PB[PB.sog != PB.line]
            y = (np_.sog > np_.line).astype(int).values
            pm = (np_.p_over / (1 - np_.p_push)).values
            pp["info"] = info_test(y, np_.mkt_p_over.values, pm)
            pp["logloss"] = {"market": logloss(y, np_.mkt_p_over.values), "model": logloss(y, pm)}
            for price in ("med", "best"):
                for thr in config.EDGE_THRESHOLDS:
                    out = []
                    for r in PB.itertuples():
                        push = r.sog == r.line
                        s = settle_two_way(r.p_over, r.p_push, getattr(r, f"over_{price}"),
                                           getattr(r, f"under_{price}"), r.sog > r.line, push, thr, "over", "under")
                        if s:
                            out.append({"game_id": r.game_id, "season": r.season, "player": r.name,
                                        "line": r.line, "side": s[0], "ev": s[1], "profit": s[2]})
                    b = pd.DataFrame(out, columns=["game_id", "season", "player", "line", "side", "ev", "profit"])
                    pp[f"{price}@{thr}"] = gate(b)
                    if price == "med" and thr == config.HEADLINE_EDGE:
                        b.to_csv(OUT / "bets_sog_props.csv", index=False)
    rep["sog_props"] = pp

    # ---------------- tails / stacks on the moneyline (priced against the market)
    rep["tails"] = tails(G) if len(G) else {}

    (OUT / "results.json").write_text(json.dumps(rep, indent=2, default=float))
    write_report(rep)
    return rep


def tails(G):
    """Bet the side each flag points to, ONLY in that flag's extreme 10%, at the median price.
    Every flag tested is listed, hits and misses alike, so nothing is cherry-picked."""
    d = G.copy()
    d["gap"] = d.p_home - d.mkt_p_home
    # flags oriented so positive = points to HOME
    d["f_model_gap"] = d.gap
    d["f_hot_goalie"] = d.away_g_sv10.fillna(0.905) - d.home_g_sv10.fillna(0.905)   # fade the hot goalie
    d["f_luck"] = d.away_luck15 - d.home_luck15                                    # fade the lucky finisher
    d["f_lineup"] = d.home_lineup_vs_recent - d.away_lineup_vs_recent              # who is missing players
    d["f_rest"] = d.away_b2b - d.home_b2b
    d["f_goalie_skill"] = d.home_g_gsax - d.away_g_gsax
    flags = ["f_model_gap", "f_hot_goalie", "f_luck", "f_lineup", "f_goalie_skill"]
    out = {}

    def bet_result(sub, home_side):
        prof = np.where(home_side, np.where(sub.home_won == 1, sub.home_med - 1, -1),
                        np.where(sub.home_won == 0, sub.away_med - 1, -1))
        return len(sub), float(np.mean(prof)) if len(sub) else np.nan

    for f in flags:
        hi, lo = d[f].quantile(0.9), d[f].quantile(0.1)
        n1, r1 = bet_result(d[d[f] >= hi], True)
        n2, r2 = bet_result(d[d[f] <= lo], False)
        allb = pd.concat([d[d[f] >= hi].assign(h=True), d[d[f] <= lo].assign(h=False)])
        n, r = bet_result(allb, allb.h.values)
        out[f] = {"bets": n, "roi": r, "top10_home_bets": n1, "top10_roi": r1, "bot10_away_bets": n2,
                  "bot10_roi": r2}
    # rest is binary: bet the rested side vs a back-to-back team
    rb = d[d.f_rest != 0]
    n, r = bet_result(rb, (rb.f_rest > 0).values)
    out["f_rest"] = {"bets": n, "roi": r}
    # stacks: how many tail flags point the same way
    z = pd.DataFrame({f: np.sign(np.where(d[f] >= d[f].quantile(0.8), 1, np.where(d[f] <= d[f].quantile(0.2), -1, 0)))
                      for f in flags})
    z["f_rest"] = np.sign(d.f_rest.values)
    net = z.sum(axis=1)
    for k in (2, 3, 4):
        sel = d[np.abs(net) >= k]
        n, r = bet_result(sel, (net[np.abs(net) >= k] > 0).values)
        out[f"stack>={k}"] = {"bets": n, "roi": r}
    out["_note"] = f"{len(flags) + 1} flags and 3 stack levels tested; expect ~1 to look good by chance."
    return out


def write_report(rep):
    L = ["# NHL shot-quality model: backtest report", ""]
    c = rep["coverage"]
    L += [f"Backtest games: {c['backtest_games']}; games with odds: {c['games_with_odds']}; "
          f"prop player-games priced: {c.get('prop_player_games_priced', 0)}", ""]
    for mkt in ("moneyline", "totals", "sog_props"):
        m = rep.get(mkt, {})
        L.append(f"## {mkt}")
        if not m:
            L += ["No data.", ""]
            continue
        i = m.get("info", {})
        if i:
            L.append(f"- Information beyond the market: model coefficient {i['b_model']:.3f} "
                     f"(t = {i['t_model']:.2f}); market coefficient {i['b_market']:.3f}. "
                     f"t above 2 = the model knows something the price doesn't.")
        ll = m.get("logloss", {})
        if ll:
            L.append("- Log loss (lower is better): " + ", ".join(f"{k} {v:.4f}" for k, v in ll.items()))
        L += ["", "| price | EV cutoff | bets | ROI | 90% CI | seasons + | by season | PASS |",
              "|---|---|---|---|---|---|---|---|"]
        for k, v in m.items():
            if "@" not in k:
                continue
            price, thr = k.split("@")
            bs = ", ".join(f"{s[2:4]}-{s[6:8]}: {x:+.1%}" for s, x in v["by_season"].items())
            L.append(f"| {price} | {float(thr):.0%} | {v['bets']} | {v['roi']:+.1%} | "
                     f"{v['ci_lo']:+.1%} to {v['ci_hi']:+.1%} | {v['seasons_positive']} | {bs} | "
                     f"{'**PASS**' if v['PASS'] else 'fail'} |")
        L.append("")
    t = rep.get("tails", {})
    if t:
        L += ["## Tails and stacks (moneyline, median price)", "",
              "| flag | bets | ROI | notes |", "|---|---|---|---|"]
        for k, v in t.items():
            if k.startswith("_"):
                continue
            extra = ""
            if "top10_roi" in v:
                extra = f"top 10% (home) {v['top10_home_bets']} @ {v['top10_roi']:+.1%}; " \
                        f"bottom 10% (away) {v['bot10_away_bets']} @ {v['bot10_roi']:+.1%}"
            L.append(f"| {k} | {v['bets']} | {v['roi']:+.1%} | {extra} |")
        L += ["", t.get("_note", ""), ""]
    g = config.GATE
    L += ["## The gate (fixed before results)",
          f"PASS needs: at least {g['min_bets']} bets, ROI > 0, the lower end of the "
          f"{g['ci_level']:.0%} bootstrap CI > 0, and positive in at least {g['min_seasons_positive']} "
          f"of {len(config.BACKTEST_SEASONS)} seasons. Headline row = median price at "
          f"{config.HEADLINE_EDGE:.0%} EV."]
    (OUT / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    run()
