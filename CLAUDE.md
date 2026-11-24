# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ML/statistical pipeline that predicts whether one of the top 50 European scorers will score in their next match, and backtests betting strategies on top of those predictions. All data comes from Understat via the `understatapi` library (EPL, La Liga, Bundesliga, Serie A, Ligue 1).

## Execution guardrails

- **Never run `python -m scripts.update_data`.** Understat has been down since
  early November 2026. A partial scrape would overwrite `data/raw/`, which
  cannot be regenerated right now and is the only working copy of the data.
- Never write to, delete or move anything under `data/raw/`.
- Before running any script that writes to `data/processed/` or
  `data/dictionaries/`, tell me what it will overwrite and wait for my approval.
- No git command that pushes or rewrites history (`push`, `rebase`,
  `reset --hard`, `filter-repo`) unless I explicitly ask for it.
- If a command may take more than a minute, ask me first.

## Environment

The repo lives in WSL (Ubuntu) and runs on the `.venv` virtualenv there (Python 3.10). There is no usable Python on the Windows side, so run commands through WSL:

```bash
wsl -d Ubuntu -- bash -lc 'cd ~/git/football-top-scorers-analysis && .venv/bin/python -m scripts.update_data'
```

Inside WSL: `source .venv/bin/activate`, then `pip install -r requirements.txt`.

## Commands

All of these must be run **from the repo root** — `src/core/config.py` defines data paths as relative strings (`data/raw`, …), and `src/dashboard/app.py` reads `data/processed/...` relatively too.

```bash
python -m scripts.update_data          # full scrape + rebuild of every data artifact (slow: heavy scraping)
python -m scripts.evaluate_models      # grid-search tuning + eval of the 3 models, writes params & metrics JSON
python -m scripts.evaluate_strategies  # backtests strategies 1-6, writes the two strategy result CSVs
python -m scripts.predict              # single-player prediction for their upcoming match
streamlit run src/dashboard/app.py     # Streamlit dashboard
```

- Use `python -m scripts.<name>`, not `python scripts/<name>.py`. There is **no `__init__.py` anywhere**; `src` and `scripts` resolve as implicit namespace packages only when the root is on `sys.path`.
- `scripts/predict.py` takes no CLI arguments: edit the `PLAYER_ID` and `MODEL_NAME` module constants at the top of the file.
- **There is no test suite.** `pytest` and `pytest-cov` are in `requirements.txt` but `pytest --collect-only` finds nothing. Note that `src/betting/test_strategy.py` is *not* a test file despite the name — it is the betting strategy implementation module.

## Data pipeline

The pipeline is a strictly sequential chain of files on disk; there is no in-memory orchestration. Each step reads what the previous one wrote, so order matters and a failed step silently poisons everything after it. `scripts/update_data.py` is the authoritative ordering:

| Step (function) | Writes |
|---|---|
| `understat_scraper.update_raw_data` | `data/raw/all_players_data.csv`, `goals_top50_last3seasons.csv`, `matches_top50_last3seasons.csv` |
| `dashboard_data_processing.build_clean_players_dataset` | `data/processed/all_players_data_clean.csv` (dashboard only) |
| `general_processing_functions.update_match_league_mapping` | `data/dictionaries/match_league_map.json` |
| `general_processing_functions.create_matches_goals_dataset` | `data/processed/matches_goals_top50_last3seasons.csv` |
| `delay_score_processing.create_delay_score_dataset` | `data/processed/top50_players_delay_score.csv` |
| `ml_features_processing.create_team_mappings` | `data/dictionaries/team_id_mapping.json`, `all_team_metrics_before_match.json` |
| `ml_features_processing.create_clean_features_dataset` | `data/processed/clean_features_dataset.csv` (ML-ready, target = `scored`) |

Two concepts carry the whole project and are computed across several files:

- **Delay** (`create_matches_goals_dataset`, then `create_delay_score_dataset`): playing minutes are accumulated per player across matches (`cumulative_time`), goals are placed on that absolute timeline (`last_goal_time`), giving `time_between_goals` and `minutes_since_last_goal_before_match` — the leak-free "how long has this player gone without scoring" feature.
- **`reliability_score`** (`create_delay_score_dataset`): weighted sum `0.50·avg_delay + 0.30·std_delay + 0.20·big_drought_freq`, each min-max scaled then inverted so higher = more regular scorer. `config.REL_THR` (0.73) filters players for **both** ML training and the betting backtests, so changing it moves every downstream number.

`all_team_metrics_before_match.json` is keyed by the composite string `"{match_id}_{team_id}"` and holds each team's pre-match cumulative form; the feature builder joins on that key for the player's team and again for the opponent.

`data/raw/` and `data/dictionaries/` are committed to git on purpose: the files
are 1-6 MB, the raw data cannot currently be re-scraped, and a fresh clone must
be able to run the whole pipeline. `data/processed/` is **no longer tracked**:
it is fully regenerable from the raw data and the code, and committing it
produced large, meaningless diffs. Never re-add it. The commit tagged
`baseline-before-refactor` holds the last fully committed state of
`data/processed/`, and is the reference for non-regression checks.

## ML layer

`src/ml/` is split by concern: `split.py` (random stratified split vs. chronological `train_test_split_by_date`), `tuning.py` (`GridSearchCV`, 5-fold stratified, refit on F-beta), `models.py` (model factory + `train_and_predict`), `evaluation.py` (metrics + threshold optimisation).

- `train_and_predict` **loads hyperparameters from `data/dictionaries/best_ml_params.json`**. Run `python -m scripts.evaluate_models` at least once before `scripts.predict` or strategy 6, otherwise the params are stale or missing.
- `config.BETA = 0.4` makes F-beta favour precision over recall — deliberate, because a false positive costs a real stake.
- Feature scaling (`StandardScaler`) is applied **only** for `logistic_regression`; tree models get raw features.

## Betting layer

`src/betting/test_strategy.py` holds strategies 1–6 plus the KPI computations. Strategies 1–5 are non-ML (fixed stake, global martingale, per-player martingale, and delay-conditioned variants) and backtest over the whole period. Strategy 6 is the only one that uses the models, and it is the only one with a chronological split (first 80% of matches train, last 20% evaluated). KPIs are intentionally different between the two families: `compute_kpis_for_ml_strat` additionally tracks `max_consecutive_losses`, which is what actually kills a martingale.

## Conventions and known rough edges

- **Never open the project's CSVs directly.** To inspect a data file, write a short pandas script that prints `df.shape`, `df.dtypes` and `df.head(5)`, and read its output. The raw/processed files are 1–6 MB.
- **Config-first**: tunable constants (`YEAR`, `SEASONS`, `LEAGUES`, `BETA`, `REL_THR`, `ODDS`, `STAKE`, data dirs) live in `src/core/config.py`. Do not hardcode them elsewhere. Logging goes through `src/core/logger.py`.
- `YEAR` is **one ahead** of the running season (2027 during the 2026-27 season); season filters are written as `season >= YEAR - nb_seasons`.
- Names lie about season counts: `LAST_3_SEASONS` holds **four** seasons, `update_raw_data` passes `nb_seasons=4`, and the filenames still say `last3seasons`. Filenames are hardcoded as string literals in several modules, so renaming requires touching every reader.
- **Feature engineering is duplicated** between `prepare_data_for_ml` (training) and `prepare_new_data_for_prediction` (inference) in `ml_features_processing.py`. Both compute the same eight ratios and the same encodings independently, and the inference path hardcodes the one-hot league columns (`league_Bundesliga`, …) that training derives from `OneHotEncoder`. Any new feature must be added to both, or inference will misalign silently.
- Scraping is the fragile part. Understat rate-limits by raising a spurious `InvalidMatch`; `create_matches_info_dataset_with_retries` retries with a random 2–5 s sleep, returns `None` on other exceptions, and raises `RuntimeError` after exhausting retries. In `create_team_metrics_dataset` that `RuntimeError` is only logged — `df` then still holds the *previous* league/season's frame, which gets processed a second time. Fix the loop rather than working around it if you touch that code.
Diagnosed on 2026-11-24: this path has never fired on the committed data (checked for duplicate keys and per-league/per-season row counts). The code is still wrong and must be fixed, but the current data is clean.
- `notebooks/` is exploratory only; nothing in the pipeline imports from it.

## Working with me

- Answer me in French. Code, identifiers and comments stay in English.
- No opportunistic refactoring: touch only what I asked for. If you spot
  something else worth fixing, mention it and move on.
- This project is also how I learn. Prefer solutions I can understand and
  explain back, and tell me when you rule out a more idiomatic but more opaque
  option.

@CONTEXTE-PERSO.md