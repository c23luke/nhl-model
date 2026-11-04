"""Settings for the NHL shot-quality model. Everything tunable lives here.

The pass/fail gate at the bottom is fixed BEFORE any real results are seen.
Do not loosen it after the fact to make a result pass.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("NHL_MODEL_ROOT", Path(__file__).resolve().parent))
DATA = ROOT / "data"
RAW_NHL = DATA / "raw_nhl"        # cached NHL API json (one file per game per endpoint)
RAW_ODDS = DATA / "raw_odds"      # cached Odds API json
OUT = ROOT / "output"

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")   # never hard-code the key

# Seasons, written as the NHL writes them (start year + end year).
XG_TRAIN_SEASONS = ["20192020", "20202021", "20212022", "20222023"]   # xG fit only, never bet
BACKTEST_SEASONS = ["20232024", "20242025", "20252026"]               # props exist from May 2023
LIVE_SEASON = "20262027"                                              # the season being bet now
MODEL_FILES = ROOT / "model_files"                                    # small fitted files the live board needs
# No early-season blackout: ratings follow PLAYERS across seasons and teams, so a team is
# rateable from game 1. Early season is exactly where a market leaning on last year's team
# should be most wrong.

# Ratings: how fast old games fade (half-life in games played by that player).
PLAYER_HALF_LIFE_GAMES = 60
GOALIE_HALF_LIFE_GAMES = 80

# Odds pulls
ODDS_REGIONS = "us"
GAME_MARKETS = "h2h,totals"
PROP_MARKETS = "player_shots_on_goal"
SNAPSHOT_MINUTES_BEFORE = 15    # "closing-ish" price: snapshot this many minutes before puck drop

# ---------------------------------------------------------------- PASS / FAIL GATE
# A market passes only if ALL of these hold on bets the model picked, out of sample:
GATE = {
    "min_bets": 250,               # total bets across backtest seasons (was 300; lowered by Caden 2026-11-03)
    "min_roi": 0.0,                # flat-stake ROI at the price actually available
    "ci_level": 0.90,              # bootstrap CI on ROI; its LOWER bound must be > 0
    "min_seasons_positive": 2,     # of the 3 backtest seasons
    "must_beat_market_logloss": False,  # reported, not required (the market is very sharp)
}
EDGE_THRESHOLDS = [0.02, 0.03, 0.05]   # expected-value cutoffs to report; 0.03 is the headline
HEADLINE_EDGE = 0.03
