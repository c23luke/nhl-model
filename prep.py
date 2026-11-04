"""Morning prep: bring last night's games in, rebuild ratings, refit the live models.

Writes data/tables/rater_state.pkl (every player's history as of this morning) and
model_files/live_models.pkl (game + prop models fit on every finished game).
"""
import pickle
import time

import pandas as pd

import config
import model
import nhl_fetch
import parse
import ratings
import xg
import jsonio
from parse import TABLES


def main():
    t0 = time.time()
    jsonio.compact(config.RAW_NHL)
    for s in config.XG_TRAIN_SEASONS + config.BACKTEST_SEASONS + [config.LIVE_SEASON]:
        nhl_fetch.fetch_season(s)
    # past seasons never change: parse them once, then only re-parse this season each morning
    hist = TABLES / "hist"
    if not (hist / "games.parquet").exists():
        hist.mkdir(parents=True, exist_ok=True)
        g, sh, p = parse.build(config.XG_TRAIN_SEASONS + config.BACKTEST_SEASONS)
        for name, df in (("games", g), ("shots", sh), ("players", p)):
            df.to_parquet(hist / f"{name}.parquet")
    g, sh, p = parse.build([config.LIVE_SEASON])
    for name, df in (("games", g), ("shots", sh), ("players", p)):
        old = pd.read_parquet(hist / f"{name}.parquet")
        both = pd.concat([old, df], ignore_index=True)
        if name == "games":
            both = both.sort_values(["date", "game_id"]).reset_index(drop=True)
        both.to_parquet(TABLES / f"{name}.parquet")
    xg.run()
    tg, pr, _ = ratings.build()
    models = model.fit_live(tg, pr)
    config.MODEL_FILES.mkdir(exist_ok=True)
    with open(config.MODEL_FILES / "live_models.pkl", "wb") as f:
        pickle.dump(models, f)
    live = tg[tg.season == config.LIVE_SEASON]
    print(f"prep done in {time.time() - t0:.0f}s: models fit through {models['fit_through']}, "
          f"{len(live)} games of {config.LIVE_SEASON} rated")


if __name__ == "__main__":
    main()
