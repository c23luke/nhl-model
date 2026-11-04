"""The betting angles, written once and used by both the backtest export and the live board.

Every pick carries its own evidence in plain words, built from the actual numbers, so a
flag can never fire as a bare label.
"""
import numpy as np
import pandas as pd

TAIL = 0.20          # a flag counts when it is in the top or bottom 20% of all earlier games
MIN_PRIOR = 300      # earlier games needed before tails are trusted
FLAGS = ["f_model_gap", "f_hot_goalie", "f_luck", "f_lineup", "f_goalie_skill"]


def add_flags(d):
    """Flags oriented so a positive value points to the HOME team."""
    d = d.copy()
    d["f_model_gap"] = d.p_home - d.mkt_p_home
    d["f_hot_goalie"] = d.away_g_sv10.fillna(0.905) - d.home_g_sv10.fillna(0.905)   # fade the hot goalie
    d["f_luck"] = d.away_luck15 - d.home_luck15                                    # fade lucky finishing
    d["f_lineup"] = d.home_lineup_vs_recent - d.away_lineup_vs_recent              # who is missing players
    d["f_goalie_skill"] = d.home_g_gsax - d.away_g_gsax
    d["f_rest"] = d.away_b2b - d.home_b2b
    return d


def _pct(prior, v):
    return float((prior < v).mean())


def evidence(r, prior, side_home, names):
    """Plain-words reason for every flag that agrees with the pick."""
    H, A = r.home_abbrev, r.away_abbrev
    us, them = (H, A) if side_home else (A, H)
    s = 1 if side_home else -1
    out = []

    def tail(f):
        p = _pct(prior[f], r[f])
        return p if s > 0 else 1 - p

    if s * r.f_model_gap > 0 and tail("f_model_gap") >= 1 - TAIL:
        pm, pk = (r.p_home, r.mkt_p_home) if side_home else (1 - r.p_home, 1 - r.mkt_p_home)
        out.append({"flag": "Model vs market",
                    "text": f"Model gives {us} {pm:.1%}, market {pk:.1%} ({(pm - pk) * 100:+.1f} pts, "
                            f"bigger than {tail('f_model_gap'):.0%} of games)"})
    if s * r.f_hot_goalie > 0 and tail("f_hot_goalie") >= 1 - TAIL:
        g_them = r.away_g_sv10 if side_home else r.home_g_sv10
        g_us = r.home_g_sv10 if side_home else r.away_g_sv10
        gid = r.away_goalie_id if side_home else r.home_goalie_id
        sv = lambda x: f".{int(round(x * 1000)):03d}" if x == x else "no recent starts"
        out.append({"flag": "Fade the hot goalie",
                    "text": f"{them} goalie {names.get(gid, gid)} {sv(g_them)} save % last 10 "
                            f"vs {us} {sv(g_us)}; hot runs don't hold"})
    if s * r.f_luck > 0 and tail("f_luck") >= 1 - TAIL:
        l_them = r.away_luck15 if side_home else r.home_luck15
        l_us = r.home_luck15 if side_home else r.away_luck15
        out.append({"flag": "Finishing luck",
                    "text": f"{them} {l_them:+.1f} goals vs expected over last 15, {us} {l_us:+.1f}; "
                            f"finishing swings mostly reverse"})
    if s * r.f_lineup > 0 and tail("f_lineup") >= 1 - TAIL:
        u = r.home_lineup_vs_recent if side_home else r.away_lineup_vs_recent
        t = r.away_lineup_vs_recent if side_home else r.home_lineup_vs_recent
        out.append({"flag": "Tonight's lineup",
                    "text": f"{them} lineup attack {t:+.2f} xG/60 vs its last 10 games (players out), "
                            f"{us} {u:+.2f}"})
    if s * r.f_goalie_skill > 0 and tail("f_goalie_skill") >= 1 - TAIL:
        gu = r.home_g_gsax if side_home else r.away_g_gsax
        gt = r.away_g_gsax if side_home else r.home_g_gsax
        gid_u = r.home_goalie_id if side_home else r.away_goalie_id
        gid_t = r.away_goalie_id if side_home else r.home_goalie_id
        out.append({"flag": "Goalie skill",
                    "text": f"{us} starter {names.get(gid_u, gid_u)} saves {gu * 100:+.1f} goals per 100 xG "
                            f"vs {them} starter {names.get(gid_t, gid_t)} {gt * 100:+.1f}"})
    if s * r.f_rest > 0:
        out.append({"flag": "Rest", "text": f"{them} on the second night of a back-to-back, {us} rested"})
    return out


def stack_votes(today, prior):
    z = pd.DataFrame({f: np.where(today[f] >= prior[f].quantile(1 - TAIL), 1,
                                  np.where(today[f] <= prior[f].quantile(TAIL), -1, 0)) for f in FLAGS},
                     index=today.index)
    z["f_rest"] = np.sign(today.f_rest.values)
    return z.sum(axis=1)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_blend(y, p_mkt, p_mod):
    """y ~ a + b*logit(market) + c*logit(model). Returns (a, b, c)."""
    X = np.column_stack([np.ones(len(y)), logit(p_mkt), logit(p_mod)])
    b = np.zeros(3)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(X @ b)))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        st = np.linalg.solve(H, X.T @ (y - mu))
        b += st
        if np.abs(st).max() < 1e-9:
            break
    return b


def blend(b, p_mkt, p_mod):
    return 1 / (1 + np.exp(-(b[0] + b[1] * logit(p_mkt) + b[2] * logit(p_mod))))


def american(dec):
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))
