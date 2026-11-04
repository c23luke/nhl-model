#!/bin/bash
# One-time: puts the NHL model on your private GitHub repo c23luke/nhl-model and turns on the schedule.
# Run from ~/nhl-model after creating the empty private repo at github.com/new.
set -e
cd "$(dirname "$0")"
mkdir -p .github/workflows
cat > .github/workflows/daily.yml <<'YAML'
# Every morning: grade last night's picks, bring in last night's games, rebuild ratings and models.
name: daily prep
on:
  schedule:
    - cron: "15 11 * * *"        # 6:15 AM Eastern (11:15 UTC)
  workflow_dispatch: {}
concurrency: { group: nhl-data, cancel-in-progress: false }
jobs:
  prep:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    env:
      ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
      SUPABASE_SERVICE_KEY: ${{ secrets.SUPABASE_SERVICE_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: pip }
      - run: pip install -r requirements.txt
      - name: restore downloaded NHL data
        uses: actions/cache/restore@v4
        with:
          path: |
            data/raw_nhl
            data/tables/hist
          key: nhl-data-${{ github.run_id }}
          restore-keys: nhl-data-
      - name: grade last night
        run: python grade.py
      - name: prep (new games, ratings, models)
        run: python prep.py
      - name: save NHL data
        if: always()
        uses: actions/cache/save@v4
        with:
          path: |
            data/raw_nhl
            data/tables/hist
          key: nhl-data-${{ github.run_id }}
      - name: save today's ratings and models
        uses: actions/cache/save@v4
        with:
          path: |
            data/tables/rater_state.pkl
            model_files/live_models.pkl
          key: nhl-state-${{ github.run_id }}
YAML
cat > .github/workflows/board.yml <<'YAML'
# Through the day: re-pull lineups, goalies and prices, update tonight's picks.
name: board
on:
  schedule:
    - cron: "*/30 15-23 * * *"    # 10:00 AM - 6:30 PM Eastern
    - cron: "*/20 0-3 * * *"      # 7:00 PM - 10:40 PM Eastern, every 20 min around puck drops
  workflow_dispatch: {}
concurrency: { group: nhl-board, cancel-in-progress: true }
jobs:
  board:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    env:
      ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
      SUPABASE_SERVICE_KEY: ${{ secrets.SUPABASE_SERVICE_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: pip }
      - run: pip install -r requirements.txt
      - name: load this morning's ratings and models
        uses: actions/cache/restore@v4
        with:
          path: |
            data/tables/rater_state.pkl
            model_files/live_models.pkl
          key: nhl-state-${{ github.run_id }}
          restore-keys: nhl-state-
      - run: python board.py
YAML
[ -d .git ] || git init -q -b main
git add .gitignore *.py *.md *.sh requirements.txt model_files/blend.json model_files/flag_history.parquet \
        .github/workflows dashboard synthetic_test_results
echo "== files going to GitHub (data, samples and keys stay on your Mac):"
git status --short
git commit -qm "NHL model: live board, daily prep, grading" || true
git remote get-url origin >/dev/null 2>&1 || git remote add origin https://github.com/c23luke/nhl-model.git
git push -u origin main
echo "done: https://github.com/c23luke/nhl-model"
