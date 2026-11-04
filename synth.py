"""Synthetic NHL universe, written in the exact NHL API + Odds API json formats, used to
test the pipeline end to end before real data exists.

Truth is known, so two markets can be written:
  --market true   prices = the true probabilities (+ vig). A correct model must LOSE here
                  (ROI near minus the vig). Any profit = leakage / a bug.
  --market stale  game prices built from each team's LAST-season roster and an average
                  goalie (a market that is slow on personnel). A personnel model should win.
Props are priced from the true mean in both modes (small noise), so props must not win.

Usage: NHL_MODEL_ROOT=/some/dir python synth.py --market stale --games-per-team 50
"""
import argparse
import json
import math
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

import config
import jsonio

RNG = np.random.default_rng(11)
N_TEAMS = 32
SEG = 40                       # seconds per 5v5 segment; 30 per period
PF = np.array([0.35, 0.30, 0.22, 0.13])
PD = np.array([0.42, 0.34, 0.24])
BASE_RATE = 48 / 3600          # attempts per second per team at 5v5 (incl. blocked)
PP_MULT, PK_MULT = 1.9, 0.45
P_UNBLOCK, P_MISS = 0.75, 0.33
P_REB, P_RUSH = 0.08, 0.15
HOME_RATE = 1.04


def sig(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------ shot geometry (true xG)
def sample_shot(rng, rush, rebound):
    if rebound:
        dist = rng.gamma(3, 3) + 3
    elif rng.random() < 0.4:
        dist = rng.gamma(4, 4) + 4
    else:
        dist = rng.uniform(25, 62)
    y = float(np.clip(rng.normal(0, 13), -40, 40))
    if abs(y) > dist:
        y = math.copysign(dist * 0.9, y)
    x = 89 - math.sqrt(max(dist ** 2 - y ** 2, 1))
    ang = math.degrees(math.atan2(abs(y), 89 - x))
    stype = rng.choice(["wrist", "snap", "slap", "backhand", "tip-in"], p=[0.5, 0.2, 0.12, 0.1, 0.08])
    tadj = {"wrist": 0, "snap": 0.15, "slap": -0.1, "backhand": -0.2, "tip-in": 0.3}[stype]
    base = -0.45 - 0.085 * dist - 0.008 * ang + tadj + 1.0 * rebound + 0.55 * rush
    return x, y, stype, base


def _pbar_table():
    """Mean goal prob per UNBLOCKED attempt as a function of delta (finish - goalie),
    averaged over the shot mix, including the rebound follow-ups. Used for exact pricing."""
    rng = np.random.default_rng(5)
    bases, wts = [], []
    for _ in range(40000):
        rush = rng.random() < P_RUSH
        bases.append(sample_shot(rng, rush, False)[3]); wts.append(1.0)
        bases.append(sample_shot(rng, False, True)[3]); wts.append(P_REB)
    bases, wts = np.array(bases), np.array(wts)
    deltas = np.linspace(-1, 1, 81)
    return deltas, np.array([(sig(bases + d) * wts).sum() / wts.sum() for d in deltas])


DELTAS, PBAR = _pbar_table()


def pbar(d, pp=False):
    return np.interp(d + (0.35 if pp else 0.0), DELTAS, PBAR)


# ------------------------------------------------------------ players
class Universe:
    def __init__(self):
        self.next_pid = 8470000
        self.players = {}
        self.rosters = {}
        self.prev_rosters = {}
        for t in range(1, N_TEAMS + 1):
            self.rosters[t] = {"F": [self.new("F") for _ in range(14)],
                               "D": [self.new("D") for _ in range(7)],
                               "G": [self.new("G"), self.new("G")]}
            # depth: sort by quality so line 1 is best
            self.rosters[t]["F"].sort(key=lambda p: -self.players[p]["off"])
            self.rosters[t]["D"].sort(key=lambda p: -(self.players[p]["off"] + self.players[p]["def"]))
            g = self.rosters[t]["G"]
            g.sort(key=lambda p: -self.players[p]["gk"])

    def new(self, pos):
        pid = self.next_pid
        self.next_pid += 1
        self.players[pid] = {
            "pos": pos, "first": f"First{pid}", "last": f"Last{pid}",
            "off": RNG.normal(0, 0.16), "def": RNG.normal(0, 0.14),
            "w": RNG.lognormal(0.0, 0.35) * (1.15 if pos == "F" else 0.75),
            "fin": RNG.normal(0, 0.06), "gk": RNG.normal(0, 0.09) if pos == "G" else 0.0,
        }
        return pid

    def offseason(self):
        self.prev_rosters = {t: {k: list(v) for k, v in r.items()} for t, r in self.rosters.items()}
        for p in self.players.values():
            p["off"] += RNG.normal(0, 0.04)
            p["def"] += RNG.normal(0, 0.04)
            p["gk"] += RNG.normal(0, 0.05) if p["pos"] == "G" else 0
        # 40% of skaters change teams, 10% retire -> rookies
        for pos in ("F", "D", "G"):
            pool = []
            for t in self.rosters:
                keep = []
                for pid in self.rosters[t][pos]:
                    r = RNG.random()
                    if r < 0.10:
                        continue
                    (pool if r < 0.45 else keep).append(pid)
                self.rosters[t][pos] = keep
            RNG.shuffle(pool)
            need = {"F": 14, "D": 7, "G": 2}[pos]
            for t in self.rosters:
                while len(self.rosters[t][pos]) < need:
                    self.rosters[t][pos].append(pool.pop() if pool else self.new(pos))
        for t in self.rosters:
            self.rosters[t]["F"].sort(key=lambda p: -self.players[p]["off"])
            self.rosters[t]["D"].sort(key=lambda p: -(self.players[p]["off"] + self.players[p]["def"]))
            self.rosters[t]["G"].sort(key=lambda p: -self.players[p]["gk"])


# ------------------------------------------------------------ exact expectations
def lineup_units(F, D):
    lines = [F[0:3], F[3:6], F[6:9], F[9:12]]
    pairs = [D[0:2], D[2:4], D[4:6]]
    return lines, pairs


def expected(U, A, B, gB, home_a):
    """Expected regulation goals for A, expected SOG for each A skater, SOG variance.
    A, B = (F list, D list) dressed lineups; gB = opposing goalie."""
    P = U.players
    la, pa = lineup_units(*A)
    lb, pb = lineup_units(*B)
    hm = HOME_RATE if home_a else 1 / HOME_RATE
    # B's on-ice defence by (line, pair)
    defB = np.array([[np.mean([P[x]["def"] for x in lb[i] + pb[j]]) for j in range(3)] for i in range(4)])
    wB = np.outer(PF, PD)
    att = defaultdict(float)
    var = defaultdict(float)
    seg_mean = defaultdict(float)      # mean SOG per 5v5 segment (for the mixture variance)
    lam = 0.0
    n5 = 90 - 3 * 2 * 3.0            # expected 5v5 segments (3 PPs each way, 3 segments each)
    for i in range(4):
        for j in range(3):
            on = la[i] + pa[j]
            offA = np.mean([P[x]["off"] for x in on])
            w = np.array([P[x]["w"] for x in on])
            rates = BASE_RATE * hm * np.exp(offA - defB) * SEG * (1 + P_REB)   # 4x3 over B configs
            for k, pid in enumerate(on):
                share = w[k] / w.sum()
                m_seg = (wB * rates).sum() * share
                pg = pbar(P[pid]["fin"] - P[gB]["gk"])
                q = P_UNBLOCK * (pg + (1 - pg) * (1 - P_MISS))
                att[pid] += PF[i] * PD[j] * m_seg * q * n5
                # variance of a Poisson mixture over B's configs, per segment
                ev2 = (wB * (rates * share * q) ** 2).sum()
                var[pid] += PF[i] * PD[j] * (m_seg * q + ev2) * n5
                seg_mean[pid] += PF[i] * PD[j] * m_seg * q
                lam += PF[i] * PD[j] * m_seg * P_UNBLOCK * pg * n5
    for pid, m in seg_mean.items():
        var[pid] -= n5 * m * m           # Var = E[L] + E[L^2] - (E L)^2, per segment
    # power play: A's unit 1 vs B's PK unit, 3 PPs x 3 segments
    ppu = la[0] + pa[0]
    pku = lb[2][:2] + pb[1]
    offA = np.mean([P[x]["off"] for x in ppu])
    defpk = np.mean([P[x]["def"] for x in pku])
    r = BASE_RATE * hm * PP_MULT * math.exp(offA - defpk) * SEG * (1 + P_REB)
    w = np.array([P[x]["w"] for x in ppu])
    for k, pid in enumerate(ppu):
        pg = pbar(P[pid]["fin"] - P[gB]["gk"], pp=True)
        q = P_UNBLOCK * (pg + (1 - pg) * (1 - P_MISS))
        m = r * w[k] / w.sum() * 9
        att[pid] += m * q
        var[pid] += m * q
        lam += m * P_UNBLOCK * pg
    # short-handed: A's PK unit attacks at PK_MULT during B's 3 PPs
    pku_a = la[2][:2] + pa[1]
    ppu_b = lb[0] + pb[0]
    offA = np.mean([P[x]["off"] for x in pku_a])
    defpp = np.mean([P[x]["def"] for x in ppu_b])
    r = BASE_RATE * hm * PK_MULT * math.exp(offA - defpp) * SEG * (1 + P_REB)
    w = np.array([P[x]["w"] for x in pku_a])
    for k, pid in enumerate(pku_a):
        pg = pbar(P[pid]["fin"] - P[gB]["gk"])
        q = P_UNBLOCK * (pg + (1 - pg) * (1 - P_MISS))
        m = r * w[k] / w.sum() * 9
        att[pid] += m * q
        var[pid] += m * q
        lam += m * P_UNBLOCK * pg
    return lam, dict(att), dict(var)


# ------------------------------------------------------------ game simulation
def mmss(s):
    s = int(round(s))
    return f"{s // 60:02d}:{s % 60:02d}"


def simulate_game(U, gid, season, day, start_utc, home, away, lineups, goalies):
    P = U.players
    plays, shifts = [], []
    sort = [0]
    score = {home: 0, away: 0}
    box = {t: defaultdict(lambda: {"toi": 0.0, "sog": 0, "goals": 0}) for t in (home, away)}

    def add_play(per, t_in, typ, team, details, sit, ptype="REG"):
        sort[0] += 1
        d = dict(details)
        d["eventOwnerTeamId"] = team
        side = "left" if per % 2 == 1 else "right"
        plays.append({"eventId": sort[0], "sortOrder": sort[0], "typeDescKey": typ,
                      "periodDescriptor": {"number": per, "periodType": ptype},
                      "timeInPeriod": mmss(t_in), "situationCode": sit,
                      "homeTeamDefendingSide": side, "details": d})

    def raw_xy(per, team, x, y):
        home_attacks_pos = (per % 2 == 1)        # home defends left in odd periods
        attacks_pos = home_attacks_pos if team == home else not home_attacks_pos
        return (x, y) if attacks_pos else (-x, -y)

    def shift_rec(pid, team, per, a, b):
        shifts.append({"playerId": pid, "teamId": team, "period": per, "startTime": mmss(a),
                       "endTime": mmss(b), "typeCode": 517, "eventDescription": None})
        box[team][pid]["toi"] += b - a

    units = {t: lineup_units(*lineups[t]) for t in (home, away)}
    # penalties: disjoint 3-segment blocks
    kh, ka = RNG.poisson(3), RNG.poisson(3)
    blocks = RNG.choice(30, size=min(kh + ka, 30), replace=False)
    pp_team = {}
    for n, b in enumerate(blocks):
        team = home if n < kh else away
        for s in range(3):
            pp_team[b * 3 + s] = team

    def attempts(per, a, b, att_team, opp, on_att, on_def, mult, pp):
        dur = b - a
        offA = np.mean([P[x]["off"] for x in on_att])
        defB = np.mean([P[x]["def"] for x in on_def])
        hm = HOME_RATE if att_team == home else 1 / HOME_RATE
        rate = BASE_RATE * hm * mult * math.exp(offA - defB)
        n = RNG.poisson(rate * dur)
        times = np.sort(RNG.uniform(a, b, n))
        w = np.array([P[x]["w"] for x in on_att])
        out = []
        for t in times:
            seq = [(t, False)]
            if RNG.random() < P_REB and t + 2 < b:
                seq.append((t + RNG.integers(1, 3), True))
            for tt, reb in seq:
                out.append((tt, reb, on_att[RNG.choice(len(on_att), p=w / w.sum())]))
        return out

    def emit_shots(per, shots, att_team, opp, sit_fn, pp):
        for t, reb, shooter in shots:
            rush = (not reb) and RNG.random() < P_RUSH
            x, y, stype, base = sample_shot(RNG, rush, reb)
            if rush:
                rx, ry = raw_xy(per, att_team, 0.0, float(RNG.uniform(-30, 30)))
                add_play(per, max(t - 3, 0), "takeaway", att_team, {"xCoord": rx, "yCoord": ry, "zoneCode": "N"},
                         sit_fn())
            gid_ = goalies[opp]
            rx, ry = raw_xy(per, att_team, x, y)
            det = {"xCoord": rx, "yCoord": ry, "zoneCode": "O", "shotType": stype,
                   "goalieInNetId": gid_}
            if RNG.random() > P_UNBLOCK:
                det["shootingPlayerId"] = shooter
                add_play(per, t, "blocked-shot", att_team, det, sit_fn())
                continue
            pg = sig(base + P[shooter]["fin"] - P[gid_]["gk"] + (0.35 if pp else 0))
            if RNG.random() < pg:
                det["scoringPlayerId"] = shooter
                score[att_team] += 1
                box[att_team][shooter]["goals"] += 1
                box[att_team][shooter]["sog"] += 1
                add_play(per, t, "goal", att_team, det, sit_fn())
            elif RNG.random() < P_MISS:
                det["shootingPlayerId"] = shooter
                add_play(per, t, "missed-shot", att_team, det, sit_fn())
            else:
                det["shootingPlayerId"] = shooter
                box[att_team][shooter]["sog"] += 1
                add_play(per, t, "shot-on-goal", att_team, det, sit_fn())

    for per in (1, 2, 3):
        add_play(per, 0, "faceoff", home, {"xCoord": 0, "yCoord": 0, "zoneCode": "N"}, "1551")
        for t in (home, away):
            shift_rec(goalies[t], t, per, 0, 1200)
        for sgi in range(30):
            gseg = (per - 1) * 30 + sgi
            a, b = sgi * SEG, (sgi + 1) * SEG
            ppt = pp_team.get(gseg)
            on = {}
            for t in (home, away):
                lines, pairs = units[t]
                if ppt is None:
                    on[t] = lines[RNG.choice(4, p=PF)] + pairs[RNG.choice(3, p=PD)]
                elif ppt == t:
                    on[t] = lines[0] + pairs[0]
                else:
                    on[t] = lines[2][:2] + pairs[1]
                for pid in on[t]:
                    shift_rec(pid, t, per, a, b)
            hs, as_ = len(on[home]), len(on[away])
            sit = f"1{as_}{hs}1"
            for t, o in ((home, away), (away, home)):
                mult = 1.0 if ppt is None else (PP_MULT if ppt == t else PK_MULT)
                sh = attempts(per, a, b, t, o, on[t], on[o], mult, ppt == t)
                emit_shots(per, sh, t, o, lambda: sit, ppt == t)
    plays.sort(key=lambda p: (p["periodDescriptor"]["number"], p["timeInPeriod"], p["sortOrder"]))
    for i, p in enumerate(plays):
        p["sortOrder"] = i + 1
    went_ot = shootout = 0
    if score[home] == score[away]:
        went_ot = 1
        per = 4
        for t in (home, away):
            shift_rec(goalies[t], t, per, 0, 300)
        tcur, done = 0.0, False
        while tcur < 300 and not done:
            a, b = tcur, min(tcur + 45, 300)
            on = {t: units[t][0][RNG.integers(0, 2)][:2] + units[t][1][0][:1] for t in (home, away)}
            for t in (home, away):
                for pid in on[t]:
                    shift_rec(pid, t, per, a, b)
            evs = []
            for t, o in ((home, away), (away, home)):
                for s in attempts(per, a, b, t, o, on[t], on[o], 1.4, False):
                    evs.append((s[0], t, o, s))
            evs.sort(key=lambda e: e[0])
            for _, t, o, s in evs:
                before = score[t]
                emit_shots(per, [s], t, o, lambda: "1331", False)
                if score[t] > before:
                    done = True
                    break
            tcur = b
        if not done:
            shootout = 1
            winner = home if RNG.random() < 0.52 else away
            score[winner] += 1
            sort[0] += 1
            plays.append({"eventId": sort[0], "sortOrder": len(plays) + 1, "typeDescKey": "goal",
                          "periodDescriptor": {"number": 5, "periodType": "SO"}, "timeInPeriod": "00:00",
                          "situationCode": "0101", "details": {"eventOwnerTeamId": winner}})
    roster_spots = []
    for t in (home, away):
        for pid in lineups[t][0] + lineups[t][1] + [goalies[t]]:
            q = P[pid]
            roster_spots.append({"teamId": t, "playerId": pid, "firstName": {"default": q["first"]},
                                 "lastName": {"default": q["last"]}, "positionCode": q["pos"] if q["pos"] != "F" else "C"})
    team_obj = lambda t: {"id": t, "abbrev": f"T{t:02d}", "placeName": {"default": f"City{t}"},
                          "commonName": {"default": f"Club{t}"}, "score": score[t]}
    pbp = {"id": gid, "season": int(season), "gameType": 2, "gameDate": day, "startTimeUTC": start_utc,
           "gameState": "OFF", "homeTeam": team_obj(home), "awayTeam": team_obj(away),
           "plays": plays, "rosterSpots": roster_spots}
    pbg = {}
    for side, t in (("homeTeam", home), ("awayTeam", away)):
        mk = lambda pid: {"playerId": pid, "name": {"default": f"{P[pid]['first'][0]}. {P[pid]['last']}"},
                          "toi": mmss(box[t][pid]["toi"]), "sog": box[t][pid]["sog"], "goals": box[t][pid]["goals"]}
        pbg[side] = {"forwards": [mk(p) for p in lineups[t][0]], "defense": [mk(p) for p in lineups[t][1]],
                     "goalies": [dict(mk(goalies[t]), starter=True)]}
    return pbp, {"playerByGameStats": pbg}, {"data": shifts}, score, went_ot


# ------------------------------------------------------------ odds writing
def american(dec_):
    return int(round((dec_ - 1) * 100)) if dec_ >= 2 else int(round(-100 / (dec_ - 1)))


def priced(p, hold=0.045, noise=0.008):
    q = min(max(p + RNG.normal(0, noise), 0.02), 0.98)
    return american(1 / (q * (1 + hold / 2))), american(1 / ((1 - q) * (1 + hold / 2)))


def p_home_win(lh, la, q=0.52):
    from scipy.stats import poisson
    k = np.arange(16)
    J = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    return np.tril(J, -1).sum() + np.trace(J) * q, J


def total_probs(J, line):
    tot = np.zeros(33)
    for h in range(16):
        for a in range(16):
            tot[h + a + (h == a)] += J[h, a]
    idx = np.arange(33)
    return tot[idx > line].sum(), tot[idx == line].sum()


def write(root_games, root_odds, args):
    U = Universe()
    seasons = config.XG_TRAIN_SEASONS + config.BACKTEST_SEASONS
    by_time = defaultdict(list)
    for si, season in enumerate(seasons):
        if si > 0:
            U.offseason()
        prev = U.prev_rosters if si > 0 else U.rosters
        n_games = N_TEAMS * args.games_per_team // 2
        d0 = date(int(season[:4]), 10, 8)
        last_day = {}
        made, day_i = 0, 0
        while made < n_games:
            day = d0 + timedelta(days=day_i)
            day_i += 1
            teams = list(range(1, N_TEAMS + 1))
            wts = np.array([0.35 if last_day.get(t) == day - timedelta(days=1) else 1.0 for t in teams])
            k = min(16, (n_games - made) * 2)
            chosen = list(RNG.choice(teams, size=k, replace=False, p=wts / wts.sum()))
            for gi in range(0, len(chosen), 2):
                home, away = int(chosen[gi]), int(chosen[gi + 1])
                made += 1
                gid = int(f"{season[:4]}02{made:04d}")
                hour = "23:00:00" if gi < 8 else "02:00:00"
                ddate = day if hour == "23:00:00" else day + timedelta(days=1)
                start = f"{ddate.isoformat()}T{hour}Z"
                lineups, goalies = {}, {}
                for t in (home, away):
                    F, D = list(U.rosters[t]["F"]), list(U.rosters[t]["D"])
                    if RNG.random() < 0.30:                       # a top-9 forward is out
                        out = F.pop(RNG.integers(0, 9))
                        F.append(out)
                    if RNG.random() < 0.15:
                        out = D.pop(RNG.integers(0, 4))
                        D.append(out)
                    lineups[t] = (F[:12], D[:6])
                    b2b = last_day.get(t) == day - timedelta(days=1)
                    goalies[t] = U.rosters[t]["G"][1] if RNG.random() < (0.7 if b2b else 0.15) else U.rosters[t]["G"][0]
                    last_day[t] = day
                pbp, box, sh, score, ot = simulate_game(U, gid, season, day.isoformat(), start, home, away,
                                                         lineups, goalies)
                folder = root_games / season
                folder.mkdir(parents=True, exist_ok=True)
                jsonio.save(folder / f"{gid}_pbp.json.gz", pbp)
                jsonio.save(folder / f"{gid}_box.json.gz", box)
                jsonio.save(folder / f"{gid}_shifts.json.gz", sh)
                if season not in config.BACKTEST_SEASONS:
                    continue
                # ---------------- market
                lh, atth, varh = expected(U, lineups[home], lineups[away], goalies[away], True)
                la, atta, vara = expected(U, lineups[away], lineups[home], goalies[home], False)
                if args.market == "stale":
                    def stale_lineup(t):
                        F = [p for p in prev[t]["F"]][:12]
                        D = [p for p in prev[t]["D"]][:6]
                        return (F, D)
                    avg_g = min(U.players, key=lambda p: abs(U.players[p]["gk"]) if U.players[p]["pos"] == "G" else 9)
                    mh, _, _ = expected(U, stale_lineup(home), stale_lineup(away), avg_g, True)
                    ma, _, _ = expected(U, stale_lineup(away), stale_lineup(home), avg_g, False)
                else:
                    mh, ma = lh, la
                p_h, J = p_home_win(mh, ma)
                line = math.floor(mh + ma) + 0.5
                po, _ = total_probs(J, line)
                ev_id = f"ev{gid}"
                books = []
                for bk in ("bookA", "bookB", "bookC"):
                    hp, ap = priced(p_h)
                    op, up = priced(po)
                    books.append({"key": bk, "title": bk, "markets": [
                        {"key": "h2h", "outcomes": [{"name": f"City{home} Club{home}", "price": hp},
                                                    {"name": f"City{away} Club{away}", "price": ap}]},
                        {"key": "totals", "outcomes": [{"name": "Over", "price": op, "point": line},
                                                       {"name": "Under", "price": up, "point": line}]}]})
                by_time[start].append({"id": ev_id, "sport_key": "icehockey_nhl", "commence_time": start,
                                       "home_team": f"City{home} Club{home}",
                                       "away_team": f"City{away} Club{away}", "bookmakers": books})
                # props: true mean + small noise, NB from exact moments
                from scipy.stats import nbinom
                pbooks = []
                for bk in ("bookA", "bookB"):
                    outs = []
                    for att, var in ((atth, varh), (atta, vara)):
                        for pid, mu in att.items():
                            mu_n = mu * math.exp(RNG.normal(0, 0.06))
                            v = max(var[pid], mu + 1e-6) * (mu_n / mu) ** 2
                            L = max(0.5, math.floor(mu_n) + 0.5)
                            n_ = mu_n ** 2 / max(v - mu_n, 1e-6) if v > mu_n + 1e-6 else 1e6
                            pover = 1 - nbinom.cdf(math.floor(L), n_, n_ / (n_ + mu_n))
                            op, up = priced(float(pover), hold=0.07, noise=0.0)
                            name = f"First{pid} Last{pid}"
                            outs += [{"name": "Over", "description": name, "price": op, "point": L},
                                     {"name": "Under", "description": name, "price": up, "point": L}]
                    pbooks.append({"key": bk, "markets": [{"key": "player_shots_on_goal", "outcomes": outs}]})
                jsonio.save(root_odds / "props" / f"{ev_id}.json.gz", {"data": {"id": ev_id, "bookmakers": pbooks}})
        print(f"  simulated {season}: {made} games")
    gdir = root_odds / "games"
    gdir.mkdir(parents=True, exist_ok=True)
    for start, evs in by_time.items():
        jsonio.save(gdir / f"{start[:19].replace(':', '')}.json.gz", {"data": evs})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["true", "stale"], default="stale")
    ap.add_argument("--games-per-team", type=int, default=50)
    a = ap.parse_args()
    write(config.RAW_NHL, config.RAW_ODDS, a)
