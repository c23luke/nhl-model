"""Expected-goals model: chance a league-average shooter scores on this shot vs an
average goalie. Shooter and goalie skill are added later in ratings.py, on top.

Walk-forward: shots in season S are scored by a model fit only on seasons before S.
Blocked shots get no xG (they never reach the net) but still count as attempts.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

import config
from parse import TABLES

NUM = ["dist", "angle", "x", "y", "own_sk", "opp_sk", "empty_net", "score_diff",
       "dt_prev", "rebound", "rush", "cross_ice", "speed_prev", "is_home"]
CAT = ["shot_type", "prev_type"]
PREV_TYPES = ["shot-on-goal", "missed-shot", "blocked-shot", "goal", "faceoff", "hit",
              "giveaway", "takeaway", "none"]


def features(df):
    X = df[NUM].astype(float).copy()
    X["shot_type"] = pd.Categorical(df["shot_type"], categories=sorted(
        ["wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around", "other"])).codes
    pt = df["prev_type"].where(df["prev_type"].isin(PREV_TYPES), "none")
    X["prev_type"] = pd.Categorical(pt, categories=PREV_TYPES).codes
    return X


def fit(df):
    X = features(df)
    cat_mask = [c in CAT for c in X.columns]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                       min_samples_leaf=200, l2_regularization=1.0,
                                       categorical_features=cat_mask, random_state=0)
    m.fit(X, df["goal"].values)
    return m


def run():
    shots = pd.read_parquet(TABLES / "shots.parquet")
    shots["xg"] = np.nan
    unb = shots["blocked"] == 0
    seasons = sorted(shots["season"].unique())
    first_bt = min(config.BACKTEST_SEASONS)
    report = []
    for s in seasons:
        train = shots[unb & (shots.season < s)]
        test_idx = shots.index[unb & (shots.season == s)]
        if len(train) < 20000:
            # earliest seasons: in-sample fit, they only feed the xG training pool / warm ratings
            train = shots[unb & (shots.season <= s)]
        m = fit(train)
        p = m.predict_proba(features(shots.loc[test_idx]))[:, 1]
        shots.loc[test_idx, "xg"] = p
        y = shots.loc[test_idx, "goal"].values
        report.append({"season": s, "shots": len(test_idx), "goals": int(y.sum()),
                       "xg_sum": float(p.sum()), "logloss": log_loss(y, p, labels=[0, 1]),
                       "auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
                       "out_of_sample": s >= first_bt or len(shots[unb & (shots.season < s)]) >= 20000})
    shots.to_parquet(TABLES / "shots_xg.parquet")
    rep = pd.DataFrame(report)
    print("xG model by season (goals vs xG should be close; AUC ~0.75+ is normal for public data):")
    print(rep.to_string(index=False))
    return shots, rep


if __name__ == "__main__":
    run()
