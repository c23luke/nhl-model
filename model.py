"""Walk-forward models.

Game model: Poisson regression of each team's REGULATION goals on tonight's personnel
features (its skaters' attack vs the opponent's skaters' defence, power play vs penalty
kill, the starting goalie's shrunk GSAx, shooters' measured finishing, home, rest).
Win and total probabilities come from the two goal rates, with OT/shootout handled:
a regulation tie adds one goal to the final total, and the winner of OT is the home team
at the rate measured in training.

Prop model: Poisson regression of each skater's shots on goal on his expected ice time x
his shrunk shot rates, plus the opponent lineup's shot attempts allowed. Over-dispersion
measured from training residuals (negative binomial).

Every prediction for date D uses a model fit only on games before the 14-day block that
contains D.
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor

REFIT_DAYS = 14
MAXG = 16


# ---------------------------------------------------------------- game model
def team_rows(tg):
    """Two rows per game, one per team, features from that team's point of view."""
    out = []
    for us, them in (("home", "away"), ("away", "home")):
        d = pd.DataFrame({
            "game_id": tg.game_id, "date": tg.date, "season": tg.season, "side": us,
            "log_off": np.log(tg[f"{us}_off5"].clip(lower=0.5)),
            "log_oppdef": np.log(tg[f"{them}_def5"].clip(lower=0.5)),
            "reloff": tg[f"{us}_reloff5"], "opp_reldef": tg[f"{them}_reldef5"],
            "log_pp": np.log(tg[f"{us}_pp"].clip(lower=0.5)),
            "log_opp_pk": np.log(tg[f"{them}_pk"].clip(lower=0.5)),
            "fin": tg[f"{us}_fin"], "opp_gsax": tg[f"{them}_g_gsax"],
            "home": int(us == "home"), "b2b": tg[f"{us}_b2b"], "opp_b2b": tg[f"{them}_b2b"],
            "y": tg[f"{us}_reg"],
        })
        out.append(d)
    return pd.concat(out, ignore_index=True)


GAME_FEATS = ["log_off", "log_oppdef", "reloff", "opp_reldef", "log_pp", "log_opp_pk",
              "fin", "opp_gsax", "home", "b2b", "opp_b2b"]


def _fit_poisson(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = PoissonRegressor(alpha=1e-4, max_iter=500).fit((X - mu) / sd, y)
    return m, mu, sd


def win_total_probs(lh, la, q_home_ot, lines):
    """P(home wins incl OT/SO) and, for each total line, (P over, P push)."""
    k = np.arange(MAXG)
    ph, pa = poisson.pmf(k, lh), poisson.pmf(k, la)
    J = np.outer(ph, pa)                   # J[h, a] regulation score
    p_home_reg = np.tril(J, -1).sum()
    p_tie = np.trace(J)
    p_home = p_home_reg + p_tie * q_home_ot
    tot = np.zeros(2 * MAXG + 1)
    for h in range(MAXG):
        for a in range(MAXG):
            tot[h + a + (1 if h == a else 0)] += J[h, a]
    res = {}
    for L in lines:
        if L is None or np.isnan(L):
            continue
        idx = np.arange(len(tot))
        res[L] = (float(tot[idx > L].sum()), float(tot[idx == L].sum()))
    return float(p_home), res


def predict_games(tg, total_lines):
    """Walk-forward game predictions. total_lines: dict game_id -> line (or NaN)."""
    rows = team_rows(tg)
    first_season = sorted(tg.season.unique())[0]
    dates = pd.to_datetime(tg.date)
    t0 = dates.min()
    tg = tg.assign(block=((dates - t0).dt.days // REFIT_DAYS).values)
    rows["block"] = rows.game_id.map(tg.set_index("game_id").block)
    preds = []
    for b in sorted(tg.block.unique()):
        train = rows[(rows.block < b) & (rows.season != first_season)]
        if len(train) < 3000:
            continue
        m, mu, sd = _fit_poisson(train[GAME_FEATS].to_numpy(float), train.y.to_numpy(float))
        tgt = tg[(tg.block < b) & (tg.season != first_season)]
        ties = tgt[tgt.home_reg == tgt.away_reg]
        q = float((ties.home_goals > ties.away_goals).mean()) if len(ties) > 50 else 0.5
        test = rows[rows.block == b]
        lam = m.predict((test[GAME_FEATS].to_numpy(float) - mu) / sd)
        test = test.assign(lam=lam)
        piv = test.pivot(index="game_id", columns="side", values="lam")
        for gid, r in piv.iterrows():
            L = total_lines.get(gid, np.nan)
            p_home, tot = win_total_probs(r["home"], r["away"], q, [L])
            po, pp = tot.get(L, (np.nan, np.nan))
            preds.append({"game_id": gid, "lam_home": r["home"], "lam_away": r["away"],
                          "p_home": p_home, "total_line_used": L, "p_over": po, "p_push": pp,
                          "q_home_ot": q})
    return pd.DataFrame(preds, columns=["game_id", "lam_home", "lam_away", "p_home", "total_line_used",
                                        "p_over", "p_push", "q_home_ot"])


# ---------------------------------------------------------------- prop model
def prop_features(pr):
    base = (pr.e5 / 3600 * pr.cf60 * pr.spc + pr.epp / 3600 * pr.sogpp + pr.epk / 3600 * pr.sogpk)
    return pd.DataFrame({
        "log_base": np.log(base.clip(lower=0.05)),
        "log_opp_ca": np.log(pr.opp_ca60.clip(lower=10)),
        "home": pr.is_home, "opp_b2b": pr.opp_b2b,
        "log_hist": np.log1p(pr.hist5 / 3600),
        "is_d": (pr.pos == "D").astype(int),
    })


def predict_props(pr):
    first_season = sorted(pr.season.unique())[0]
    dates = pd.to_datetime(pr.date)
    blk = ((dates - dates.min()).dt.days // REFIT_DAYS).values
    pr = pr.assign(block=blk)
    X_all = prop_features(pr).to_numpy(float)
    out = []
    for b in sorted(np.unique(blk)):
        tr = (pr.block < b) & (pr.season != first_season)
        if tr.sum() < 20000:
            continue
        m, mu, sd = _fit_poisson(X_all[tr.values], pr.sog[tr].to_numpy(float))
        mtr = m.predict((X_all[tr.values] - mu) / sd)
        ytr = pr.sog[tr].to_numpy(float)
        alpha = max(float(np.mean(((ytr - mtr) ** 2 - mtr) / mtr ** 2)), 1e-3)  # NB2 dispersion
        te = (pr.block == b).values
        mte = m.predict((X_all[te] - mu) / sd)
        out.append(pd.DataFrame({"game_id": pr.game_id[te].values, "player_id": pr.player_id[te].values,
                                 "mu_sog": mte, "alpha": alpha}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def p_over_nb(mu, alpha, line):
    """P(SOG > line) under NB2(mu, alpha); half-point lines only have no push."""
    n = 1.0 / alpha
    p = n / (n + mu)
    k = np.floor(line)
    p_le = nbinom.cdf(k, n, p)
    p_eq = nbinom.pmf(line, n, p) if float(line).is_integer() else 0.0
    if float(line).is_integer():
        return 1 - p_le, p_eq          # over wins if > line, push if == line
    return 1 - p_le, 0.0


# ---------------------------------------------------------------- live: fit once, predict tonight
def fit_live(tg, pr):
    """Fit the game and prop models on every finished game (first season skipped, its ratings
    are cold). Returns a dict the live board can pickle."""
    first = sorted(tg.season.unique())[0]
    done = tg[tg.home_reg.notna() & (tg.season != first)]
    rows = team_rows(done)
    gm, gmu, gsd = _fit_poisson(rows[GAME_FEATS].to_numpy(float), rows.y.to_numpy(float))
    ties = done[done.home_reg == done.away_reg]
    q = float((ties.home_goals > ties.away_goals).mean())
    pdone = pr[pr.sog.notna() & (pr.season != first)]
    X = prop_features(pdone).to_numpy(float)
    y = pdone.sog.to_numpy(float)
    pm, pmu, psd = _fit_poisson(X, y)
    mhat = pm.predict((X - pmu) / psd)
    alpha = max(float(np.mean(((y - mhat) ** 2 - mhat) / mhat ** 2)), 1e-3)
    return {"game": (gm, gmu, gsd), "q_home_ot": q, "prop": (pm, pmu, psd), "alpha": alpha,
            "fit_through": str(done.date.max())}


def predict_live(models, tg_today, pr_today, total_lines):
    gm, gmu, gsd = models["game"]
    rows = team_rows(tg_today.assign(home_reg=0, away_reg=0))
    lam = gm.predict((rows[GAME_FEATS].to_numpy(float) - gmu) / gsd)
    piv = rows.assign(lam=lam).pivot(index="game_id", columns="side", values="lam")
    games = []
    for gid, r in piv.iterrows():
        L = total_lines.get(gid, np.nan)
        p_home, tot = win_total_probs(r["home"], r["away"], models["q_home_ot"], [L])
        po, pp = tot.get(L, (np.nan, np.nan))
        games.append({"game_id": gid, "lam_home": r["home"], "lam_away": r["away"], "p_home": p_home,
                      "total_line_used": L, "p_over": po, "p_push": pp})
    pm, pmu, psd = models["prop"]
    props = pr_today[["game_id", "player_id"]].copy()
    if len(pr_today):
        props["mu_sog"] = pm.predict((prop_features(pr_today).to_numpy(float) - pmu) / psd)
        props["alpha"] = models["alpha"]
    return pd.DataFrame(games), props
