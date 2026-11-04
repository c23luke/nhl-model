"""Pre-game PERSONNEL ratings. A player's history follows the player, never the jersey.

For each game we rate the players who actually dressed, using only games played on
EARLIER dates (any team, any season), then roll those players up to a team number
weighted by how much each one plays. No team-season stat is ever an input.

Shrinkage strength (how many minutes/shots before we trust a player's own number)
is MEASURED by split-half reliability on the xG-training seasons, not picked by feel.

Outputs (data/tables):
  team_games.parquet    one row per game, home_* / away_* pre-game features + result
  prop_rows.parquet     one row per skater-game, pre-game shot features + actual SOG
  shrinkage.json        the measured shrinkage constants
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

import config
from parse import TABLES

# accumulator layout for skaters
SK = ["toi", "toi5", "toipp", "toipk",
      "on_xgf5", "on_xga5", "on_cf5", "on_ca5", "off_xgf5", "off_xga5", "off_toi5",
      "on_xgf_pp", "on_xga_pk", "ixg", "igoal", "icf5", "isog5", "icfpp", "isogpp",
      "isogpk", "icf_all", "isog_all"]
SKI = {k: i for i, k in enumerate(SK)}
GK = ["xga", "ga", "sa"]
RECENT_HL = 8   # games, for expected ice time


def _explode(shots, col, mask):
    s = shots.loc[mask, ["game_id", col, "xg", "goal"]].explode(col).dropna(subset=[col])
    s[col] = s[col].astype("int64")
    s["xg"] = s["xg"].fillna(0.0)
    return s


def player_game_table(shots, players):
    """Per skater-game raw contributions (what happened in that game)."""
    sk = players[players.pos != "G"].copy()
    key = ["game_id", "player_id"]
    shots = shots.copy()
    shots["xg0"] = shots["xg"].fillna(0.0)
    five = shots.five_v_five == 1
    pp = shots.own_sk > shots.opp_sk
    pk_against = shots.own_sk > shots.opp_sk   # shots BY the team on the PP = against the PK

    def agg(col, mask, name_x, name_c=None):
        e = _explode(shots, col, mask).rename(columns={col: "player_id"})
        g = e.groupby(key).agg(**{name_x: ("xg", "sum")})
        if name_c:
            g[name_c] = e.groupby(key).size()
        return g

    parts = [agg("for_on_ice", five, "on_xgf5", "on_cf5"),
             agg("against_on_ice", five, "on_xga5", "on_ca5"),
             agg("for_on_ice", pp, "on_xgf_pp"),
             agg("against_on_ice", pk_against, "on_xga_pk")]
    s = shots.dropna(subset=["shooter"]).copy()
    s["player_id"] = s["shooter"].astype("int64")
    ne = s.empty_net == 0
    ind = pd.DataFrame({
        "ixg": s[ne].groupby(key)["xg0"].sum(),
        "igoal": s[ne].groupby(key)["goal"].sum(),
        "icf5": s[s.five_v_five == 1].groupby(key).size(),
        "isog5": s[(s.five_v_five == 1)].groupby(key)["on_goal"].sum(),
        "icfpp": s[s.own_sk > s.opp_sk].groupby(key).size(),
        "isogpp": s[s.own_sk > s.opp_sk].groupby(key)["on_goal"].sum(),
        "isogpk": s[s.own_sk < s.opp_sk].groupby(key)["on_goal"].sum(),
        "icf_all": s.groupby(key).size(),
        "isog_all": s.groupby(key)["on_goal"].sum(),
    })
    sk = sk.set_index(key).join(parts + [ind]).reset_index()
    sk = sk.rename(columns={"toi_5v5": "toi5", "toi_pp": "toipp", "toi_pk": "toipk"})
    for c in SK:
        if c not in sk:
            sk[c] = 0.0
    for c in ["toi5", "toipp", "toipk"]:
        sk[c] = sk[c].fillna(0.0)
    num = [c for c in SK if c not in ("off_xgf5", "off_xga5", "off_toi5")]
    sk[num] = sk[num].fillna(0.0)
    # off-ice: team 5v5 totals in that game minus this player's on-ice
    team5 = shots[five].groupby(["game_id", "team_id"])["xg0"].sum()
    opp5 = shots[five].groupby(["game_id", "opp_id"])["xg0"].sum()
    ttoi = sk.groupby(["game_id", "team_id"])["toi5"].sum() / 5.0
    idx = pd.MultiIndex.from_arrays([sk.game_id, sk.team_id])
    sk["off_xgf5"] = team5.reindex(idx).fillna(0).values - sk["on_xgf5"].values
    sk["off_xga5"] = opp5.reindex(pd.MultiIndex.from_arrays([sk.game_id, sk.team_id])).fillna(0).values - sk["on_xga5"].values
    sk["off_toi5"] = np.maximum(ttoi.reindex(idx).fillna(0).values - sk["toi5"].values, 0)
    return sk


def goalie_game_table(shots, players):
    g = players[players.pos == "G"][["game_id", "player_id", "team_id", "starter", "box_toi"]]
    s = shots[(shots.blocked == 0) & (shots.empty_net == 0)].dropna(subset=["goalie"]).copy()
    s["player_id"] = s["goalie"].astype("int64")
    a = s.groupby(["game_id", "player_id"]).agg(xga=("xg", "sum"), ga=("goal", "sum"),
                                                 sa=("on_goal", "sum")).reset_index()
    return g.merge(a, on=["game_id", "player_id"], how="left").fillna({"xga": 0, "ga": 0, "sa": 0})


# ------------------------------------------------------------- measured shrinkage
def split_half_k(df, num, den, min_den):
    """Reliability of num/den between odd and even games within a player-season.
    Returns k (in den units) such that weight on own data = den/(den+k)."""
    d = df.sort_values("game_id").copy()
    d["half"] = d.groupby(["player_id", "season"]).cumcount() % 2
    h = d.groupby(["player_id", "season", "half"])[[num, den]].sum().unstack("half")
    h = h.dropna()
    h = h[(h[(den, 0)] >= min_den) & (h[(den, 1)] >= min_den)]
    if len(h) < 30:
        return float(min_den * 4)
    r0 = h[(num, 0)] / h[(den, 0)]
    r1 = h[(num, 1)] / h[(den, 1)]
    r = np.corrcoef(r0, r1)[0, 1]
    e_half = float(np.median((h[(den, 0)] + h[(den, 1)]) / 2))
    r = min(max(r, 0.02), 0.98)
    return e_half * (1 - r) / r


def measure_shrinkage(sk, gk, seasons):
    a = sk[sk.season.isin(seasons)]
    b = gk[gk.season.isin(seasons)]
    k = {
        "on_xgf5": split_half_k(a, "on_xgf5", "toi5", 3000),
        "on_xga5": split_half_k(a, "on_xga5", "toi5", 3000),
        "on_ca5": split_half_k(a, "on_ca5", "toi5", 3000),
        "ppx": split_half_k(a, "on_xgf_pp", "toipp", 600),
        "pkx": split_half_k(a, "on_xga_pk", "toipk", 600),
        "isogpk": split_half_k(a, "isogpk", "toipk", 600),
        "rel_xgf5": split_half_k(a.assign(r=a.on_xgf5 - a.off_xgf5 * a.toi5 / a.off_toi5.clip(lower=1)),
                                 "r", "toi5", 3000),
        "rel_xga5": split_half_k(a.assign(r=a.on_xga5 - a.off_xga5 * a.toi5 / a.off_toi5.clip(lower=1)),
                                 "r", "toi5", 3000),
        "icf5": split_half_k(a, "icf5", "toi5", 3000),
        "isog_per_cf": split_half_k(a, "isog_all", "icf_all", 40),
        "isogpp": split_half_k(a, "isogpp", "toipp", 600),
        # finishing: goals above xG per unit of xG
        "finish": split_half_k(a.assign(f=a.igoal - a.ixg), "f", "ixg", 3),
        # goalie: goals saved above expected per unit of xG faced
        "gsax": split_half_k(b.assign(f=b.xga - b.ga), "f", "xga", 10),
    }
    return {kk: float(v) for kk, v in k.items()}


class Rater:
    """Holds every player's running history and turns a lineup into team features.
    Used by the historical walk and, from the saved state, by the live board, so both
    rate a lineup in exactly the same way."""

    def __init__(self, K, prior):
        self.K, self.prior = K, prior
        self.dS = 0.5 ** (1 / config.PLAYER_HALF_LIFE_GAMES)
        self.dG = 0.5 ** (1 / config.GOALIE_HALF_LIFE_GAMES)
        self.dR = 0.5 ** (1 / RECENT_HL)
        self.acc, self.rec, self.gacc = {}, {}, {}
        self.grec, self.tm_fin, self.team_lineup_hist = {}, {}, {}
        self.team_last_game = {}
        self.last_date = None

    # ---- single players
    def skater(self, pid, pos):
        K, prior = self.K, self.prior
        a = self.acc.get(pid, np.zeros(len(SK)))
        t5 = a[SKI["toi5"]]
        sh = lambda num, k, pr: (num + pr * k) / (t5 + k)
        xgf = sh(a[SKI["on_xgf5"]], K["on_xgf5"], prior["xgf60"] / 3600) * 3600
        xga = sh(a[SKI["on_xga5"]], K["on_xga5"], prior["xga60"] / 3600) * 3600
        off_rate_f = a[SKI["off_xgf5"]] / max(a[SKI["off_toi5"]], 1)
        off_rate_a = a[SKI["off_xga5"]] / max(a[SKI["off_toi5"]], 1)
        relf = (a[SKI["on_xgf5"]] - off_rate_f * t5) / (t5 + K["rel_xgf5"]) * 3600
        rela = (a[SKI["on_xga5"]] - off_rate_a * t5) / (t5 + K["rel_xga5"]) * 3600
        cf60 = sh(a[SKI["icf5"]], K["icf5"], prior["cf60"] / 3600) * 3600
        spc = (a[SKI["isog_all"]] + prior["sog_per_cf"] * K["isog_per_cf"]) / (a[SKI["icf_all"]] + K["isog_per_cf"])
        tpp = a[SKI["toipp"]]
        sogpp = (a[SKI["isogpp"]] + prior["sogpp60"] / 3600 * K["isogpp"]) / (tpp + K["isogpp"]) * 3600
        tpk = a[SKI["toipk"]]
        sogpk = (a[SKI["isogpk"]] + prior["sogpk60"] / 3600 * K["isogpk"]) / (tpk + K["isogpk"]) * 3600
        ppx = (a[SKI["on_xgf_pp"]] + prior["ppxgf60"] / 3600 * K["ppx"]) / (tpp + K["ppx"]) * 3600
        pkx = (a[SKI["on_xga_pk"]] + prior["pkxga60"] / 3600 * K["pkx"]) / (tpk + K["pkx"]) * 3600
        ca60 = sh(a[SKI["on_ca5"]], K["on_ca5"], prior["ca60"] / 3600) * 3600
        fin = (a[SKI["igoal"]] - a[SKI["ixg"]]) / (a[SKI["ixg"]] + K["finish"])
        ixg60 = a[SKI["ixg"]] / max(a[SKI["toi"]], 1) * 3600
        r = self.rec.get(pid)
        if r is not None and r[4] > 0:
            e5, epp, epk, etoi = r[0] / r[4], r[1] / r[4], r[2] / r[4], r[3] / r[4]
        else:
            e5, epp, epk = prior["toi5"][pos], prior["toipp"][pos], prior["toipk"][pos]
            etoi = e5 + epp + epk
        return dict(xgf=xgf, xga=xga, ca60=ca60, relf=relf, rela=rela, cf60=cf60, spc=spc, sogpp=sogpp,
                    sogpk=sogpk, ppx=ppx, pkx=pkx, fin=fin, ixg60=ixg60, e5=e5, epp=epp,
                    epk=epk, etoi=etoi, hist5=t5)

    def goalie(self, pid):
        a = self.gacc.get(pid, np.zeros(3))
        gsax_rate = (a[0] - a[1]) / (a[0] + self.K["gsax"])
        last = self.grec.get(pid, [])[-10:]
        ga10, sa10 = sum(x[0] for x in last), sum(x[1] for x in last)
        sv10 = 1 - ga10 / sa10 if sa10 > 0 else np.nan
        return gsax_rate, a[0], sv10, len(last)

    # ---- a team's lineup
    def side(self, players, goalie_id, tid, date):
        rats = [(pid, pos, self.skater(pid, pos)) for pid, pos in players]
        if not rats:
            return None
        w5 = np.array([r["e5"] for _, _, r in rats])
        wpp = np.array([r["epp"] for _, _, r in rats])
        wpk = np.array([r["epk"] for _, _, r in rats])
        w5n = w5 / w5.sum() if w5.sum() > 0 else np.full(len(w5), 1 / len(w5))
        wppn = wpp / wpp.sum() if wpp.sum() > 0 else w5n
        wpkn = wpk / wpk.sum() if wpk.sum() > 0 else w5n
        arr = lambda k: np.array([r[k] for _, _, r in rats])
        ixgw = arr("ixg60") * arr("etoi")
        fin = float((arr("fin") * ixgw).sum() / ixgw.sum()) if ixgw.sum() > 0 else 0.0
        gs, gx, sv10, gn = self.goalie(goalie_id) if goalie_id else (0.0, 0.0, np.nan, 0)
        last = self.team_last_game.get(tid)
        rest = (pd.Timestamp(date) - pd.Timestamp(last)).days if last else 7
        tf = self.tm_fin.get(tid, [])[-15:]
        off5 = float((arr("xgf") * w5n).sum())
        hist = self.team_lineup_hist.get(tid, [])[-10:]
        out = dict(
            off5=off5, def5=float((arr("xga") * w5n).sum()),
            ca60=float((arr("ca60") * w5n).sum()),
            reloff5=float((arr("relf") * w5).sum() / 60.0),
            reldef5=float((arr("rela") * w5).sum() / 60.0),
            pp=float((arr("ppx") * wppn).sum()), pk=float((arr("pkx") * wpkn).sum()),
            cf60=float((arr("cf60") * w5n).sum()), fin=fin,
            g_gsax=gs, g_xga_hist=gx, g_sv10=sv10, g_n10=gn, goalie_id=goalie_id,
            n_low_hist=int((arr("hist5") < 300 * 60).sum()), n_sk=len(rats),
            rest=min(rest, 7), b2b=int(rest == 1),
            luck15=float(np.sum(tf)) if tf else 0.0, n15=len(tf),
            lineup_vs_recent=off5 - float(np.mean(hist)) if hist else 0.0,
        )
        return out, rats

    def game_features(self, g, date, lineups):
        """lineups: {team_id: ([(player_id, pos), ...], starting_goalie_id)}.
        Returns (team-game row, prop rows) using only history before `date`."""
        row = {"game_id": g.game_id, "season": g.season, "date": date, "start_utc": g.start_utc,
               "home_id": g.home_id, "away_id": g.away_id, "home_name": g.home_name,
               "away_name": g.away_name, "home_abbrev": g.home_abbrev, "away_abbrev": g.away_abbrev,
               "home_goals": g.home_goals, "away_goals": g.away_goals,
               "home_reg": g.home_reg, "away_reg": g.away_reg, "went_ot": g.went_ot}
        sides = {}
        for side, tid in (("home", g.home_id), ("away", g.away_id)):
            pl, gid = lineups.get(tid, ([], None))
            res = self.side(pl, gid, tid, date)
            if res is None:
                return None, []
            sides[side] = res
        for side in ("home", "away"):
            for k, v in sides[side][0].items():
                row[f"{side}_{k}"] = v
        props = []
        for side, opp in (("home", "away"), ("away", "home")):
            o = sides[opp][0]
            tid = g.home_id if side == "home" else g.away_id
            for pid, pos, r in sides[side][1]:
                props.append({"game_id": g.game_id, "date": date, "season": g.season, "start_utc": g.start_utc,
                              "player_id": pid, "pos": pos, "team_id": tid, "is_home": int(side == "home"),
                              "e5": r["e5"], "epp": r["epp"], "epk": r["epk"], "cf60": r["cf60"],
                              "spc": r["spc"], "sogpp": r["sogpp"], "sogpk": r["sogpk"], "hist5": r["hist5"],
                              "opp_ca60": o["ca60"], "opp_xga60": o["def5"], "opp_pk": o["pk"], "opp_b2b": o["b2b"]})
        return row, props

    # ---- after a day's games are played
    def update_game(self, g, date, sg, gg, team_xg):
        if sg is None:
            return
        vals = sg[SK].to_numpy(float)
        for pid, v, t5, tpp, tpk, tt in zip(sg.player_id.values, vals, sg.toi5.values, sg.toipp.values,
                                            sg.toipk.values, sg.box_toi.values):
            self.acc[pid] = self.acc.get(pid, np.zeros(len(SK))) * self.dS + np.nan_to_num(v)
            self.rec[pid] = self.rec.get(pid, np.zeros(5)) * self.dR + np.array([t5, tpp, tpk, tt, 1.0])
        if gg is not None:
            for r in gg.itertuples():
                self.gacc[r.player_id] = self.gacc.get(r.player_id, np.zeros(3)) * self.dG + np.array([r.xga, r.ga, r.sa])
                if r.sa > 0:
                    self.grec.setdefault(r.player_id, []).append((r.ga, r.sa))
        for tid in (g.home_id, g.away_id):
            self.team_last_game[tid] = date
            if (g.game_id, tid) in team_xg.index:
                x = team_xg.loc[(g.game_id, tid)]
                self.tm_fin.setdefault(tid, []).append(float(x.g - x.xg))
        self.last_date = date

    def end_of_day(self, day_rows):
        for tr_ in day_rows:
            self.team_lineup_hist.setdefault(tr_["home_id"], []).append(tr_["home_off5"])
            self.team_lineup_hist.setdefault(tr_["away_id"], []).append(tr_["away_off5"])

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        import pickle
        r = cls.__new__(cls)
        with open(path, "rb") as f:
            r.__dict__.update(pickle.load(f))
        return r


# ------------------------------------------------------------- the walk
def build():
    shots = pd.read_parquet(TABLES / "shots_xg.parquet")
    players = pd.read_parquet(TABLES / "players.parquet")
    games = pd.read_parquet(TABLES / "games.parquet")
    sk = player_game_table(shots, players)
    gk = goalie_game_table(shots, players)
    sk["season"] = sk.game_id.map(games.set_index("game_id").season)
    gk["season"] = gk.game_id.map(games.set_index("game_id").season)
    K = measure_shrinkage(sk, gk, config.XG_TRAIN_SEASONS)
    (TABLES / "shrinkage.json").write_text(json.dumps(K, indent=2))
    print("measured shrinkage (exposure needed for 50% trust in a player's own number):")
    for kk, v in K.items():
        unit = {"isog_per_cf": "attempts", "finish": "xG", "gsax": "xG faced"}.get(kk, "seconds")
        print(f"  {kk:12s} {v:10.1f} {unit}")

    # league priors from the training seasons
    tr = sk[sk.season.isin(config.XG_TRAIN_SEASONS)]
    T5 = tr.toi5.sum()
    prior = {
        "xgf60": tr.on_xgf5.sum() / T5 * 3600, "xga60": tr.on_xga5.sum() / T5 * 3600,
        "cf60": tr.icf5.sum() / T5 * 3600, "ca60": tr.on_ca5.sum() / T5 * 3600, "sog_per_cf": tr.isog_all.sum() / max(tr.icf_all.sum(), 1),
        "sogpp60": tr.isogpp.sum() / max(tr.toipp.sum(), 1) * 3600,
        "sogpk60": tr.isogpk.sum() / max(tr.toipk.sum(), 1) * 3600,
        "ppxgf60": tr.on_xgf_pp.sum() / max(tr.toipp.sum(), 1) * 3600,
        "pkxga60": tr.on_xga_pk.sum() / max(tr.toipk.sum(), 1) * 3600,
        "toi5": {"F": tr[tr.pos == "F"].toi5.median(), "D": tr[tr.pos == "D"].toi5.median()},
        "toipp": {"F": tr[tr.pos == "F"].toipp.median(), "D": tr[tr.pos == "D"].toipp.median()},
        "toipk": {"F": tr[tr.pos == "F"].toipk.median(), "D": tr[tr.pos == "D"].toipk.median()},
    }

    R = Rater(K, prior)
    sk_by_game = {g: d for g, d in sk.groupby("game_id")}
    gk_by_game = {g: d for g, d in gk.groupby("game_id")}
    team_xg = shots[shots.empty_net == 0].groupby(["game_id", "team_id"]).agg(
        xg=("xg", "sum"), g=("goal", "sum"))

    team_rows, prop_rows = [], []
    for date, day in games.groupby("date", sort=True):
        # ---------- 1) snapshot ratings for every game on this date (nothing from today used)
        day_rows = []
        for g in day.itertuples():
            sg = sk_by_game.get(g.game_id)
            gg = gk_by_game.get(g.game_id)
            if sg is None or gg is None:
                continue
            lineups = {}
            for tid in (g.home_id, g.away_id):
                lin = sg[sg.team_id == tid]
                st = gg[(gg.team_id == tid) & (gg.starter == 1)]
                lineups[tid] = (list(zip(lin.player_id, lin.pos)), int(st.player_id.iloc[0]) if len(st) else None)
            row, props = R.game_features(g, date, lineups)
            if row is None:
                continue
            names = dict(zip(sg.player_id, sg.full_name))
            sogs = dict(zip(sg.player_id, sg.sog))
            tois = dict(zip(sg.player_id, sg.box_toi))
            for p in props:
                p.update(name=names.get(p["player_id"]), sog=sogs.get(p["player_id"]),
                         toi_actual=tois.get(p["player_id"]))
            team_rows.append(row)
            day_rows.append(row)
            prop_rows.extend(props)
        # ---------- 2) update with today's results
        for g in day.itertuples():
            R.update_game(g, date, sk_by_game.get(g.game_id), gk_by_game.get(g.game_id), team_xg)
        R.end_of_day(day_rows)

    R.save(TABLES / "rater_state.pkl")
    tg = pd.DataFrame(team_rows)
    pr = pd.DataFrame(prop_rows)
    tg.to_parquet(TABLES / "team_games.parquet")
    pr.to_parquet(TABLES / "prop_rows.parquet")
    print(f"team-game rows {len(tg)}   skater prop rows {len(pr)}")
    return tg, pr, K


if __name__ == "__main__":
    build()
