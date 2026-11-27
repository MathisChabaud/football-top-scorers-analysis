# CLAUDE.md

ML/statistical pipeline predicting whether one of the top 50 European scorers will score in their next match, plus betting-strategy backtests on those predictions. Single source: Understat via `understatapi` (EPL, La Liga, Bundesliga, Serie A, Ligue 1).

## Execution guardrails

- **Never run `python -m scripts.update_data`.** Understat has been down since early November 2026; a partial scrape would overwrite `data/raw/`, the only working copy, which cannot be regenerated. To rebuild derived data use `python -m scripts.rebuild_processed`: offline, writes only to `data/processed/`.
- Never write to, delete or move anything under `data/raw/`.
- Before running anything that writes to `data/processed/` or `data/dictionaries/`, tell me what it will overwrite and wait for my approval. `scripts.rebuild_processed --dry-run` lists the targets without writing.
- `data/dictionaries/match_league_map.json`, `team_id_mapping.json` and `all_team_metrics_before_match.json` are **not** regenerable from `data/raw/`: they come from a separate scrape of ~2900 HTTP calls to a site that is down. The committed copies are the only ones. Never delete, truncate or regenerate them.
- No git command that pushes or rewrites history (`push`, `rebase`, `reset --hard`, `filter-repo`) unless I explicitly ask for it.

## Environment

The repo lives in WSL (Ubuntu) and runs on the `.venv` virtualenv there (Python 3.10). There is no usable Python on the Windows side:

```bash
wsl -d Ubuntu -- bash -lc 'cd ~/git/football-top-scorers-analysis && .venv/bin/python -m scripts.rebuild_processed --dry-run'
```

Inside WSL: `source .venv/bin/activate`, then `pip install -r requirements.txt`.

## Commands

Run everything **from the repo root**: `src/core/config.py` holds data paths as relative strings, and `src/dashboard/app.py` reads `data/processed/...` relatively too.

```bash
python -m scripts.rebuild_processed    # offline rebuild of data/processed/ (--dry-run lists targets)
python -m scripts.update_data          # full scrape + rebuild (NEVER run it, see guardrails)
python -m scripts.evaluate_models      # tuning + eval of the 3 models -> best_ml_params.json, results/
python -m scripts.evaluate_strategies  # backtests strategies 1-6 -> the 2 CSVs in results/
python -m scripts.predict              # single-player prediction (needs network)
streamlit run src/dashboard/app.py     # dashboard
```

- **A fresh clone must run `scripts.rebuild_processed` first**: `data/processed/` is untracked, so every other command fails on a missing file. Takes seconds, no network.
- Use `python -m scripts.<name>`, never `python scripts/<name>.py`. There is **no `__init__.py` anywhere**; `src` and `scripts` resolve only because the root is on `sys.path`.
- `scripts/predict.py` takes no CLI arguments: edit the `PLAYER_ID` and `MODEL_NAME` module constants.
- **There is no test suite.** pytest is installed but collects nothing. `src/betting/test_strategy.py` is *not* a test file despite the name — it is the strategy implementation.

## Data pipeline

A strictly sequential chain of files on disk, no in-memory orchestration: each step reads what the previous one wrote, so order matters and a failed step silently poisons everything downstream.

**`src/pipeline.py` is the authoritative ordering**, as two tables of `(label, callable, [files written])`. The scripts are thin wrappers over them.

- `SCRAPE_STEPS` / `scrape()` — 3 network steps: `update_raw_data` → the 3 raw CSVs; `refresh_match_league_mapping` → `match_league_map.json`; `create_team_mappings` → `team_id_mapping.json` + `all_team_metrics_before_match.json`.
- `REBUILD_STEPS` / `rebuild()` — 4 offline steps, in order: `build_clean_players_dataset` → `all_players_data_clean.csv` (dashboard only); `create_matches_goals_dataset` → `matches_goals_top50_last3seasons.csv`; `create_delay_score_dataset` → `top50_players_delay_score.csv`; `create_clean_features_dataset` → `clean_features_dataset.csv` (ML-ready, target = `scored`).

`update_data.main()` runs `scrape()` then `rebuild()`. That reorders the original interleaved sequence; it is safe because `create_team_mappings` reads nothing from disk and every other step still follows the one it consumes — verified by dependency analysis, not by execution.

Two concepts carry the whole project:

- **Delay** (`create_matches_goals_dataset`, then `create_delay_score_dataset`): playing minutes accumulate per player (`cumulative_time`), goals are placed on that absolute timeline (`last_goal_time`), giving `time_between_goals` and `minutes_since_last_goal_before_match` — the leak-free "how long has this player gone without scoring" feature.
- **`reliability_score`** (`create_delay_score_dataset`): `0.50·avg_delay + 0.30·std_delay + 0.20·big_drought_freq`, each min-max scaled then inverted so higher = more regular scorer. `config.REL_THR` (0.73) filters players for **both** ML training and the backtests, so changing it moves every downstream number.
- `all_team_metrics_before_match.json` is keyed by `"{match_id}_{team_id}"` and holds each team's pre-match cumulative form; the feature builder joins on it for the player's team and again for the opponent.

**Tracking.** `data/raw/` and `data/dictionaries/` are committed on purpose (1-6 MB, not re-scrapable). `data/processed/` is **not** tracked — fully regenerable by `scripts.rebuild_processed`; never re-add it. `results/` **is** tracked and holds the three published outputs (`model_evaluation_results.json`, the 2 strategy CSVs); `save_dict_json` writes to `DICTS_DIR` unless given `directory=RESULTS_DIR`. The tag `baseline-before-refactor` is the non-regression reference. The four offline artifacts are deterministic (rebuilt identically, 2026-11-26), so rebuild + diff works as a regression check.

## ML layer

`src/ml/` is split by concern: `split.py` (random stratified vs. chronological `train_test_split_by_date`), `tuning.py` (`GridSearchCV`, 5-fold stratified, refit on F-beta), `models.py` (factory + `train_and_predict`), `evaluation.py` (metrics + threshold search).

- `train_and_predict` **loads hyperparameters from `data/dictionaries/best_ml_params.json`**. Run `scripts.evaluate_models` at least once before `scripts.predict` or strategy 6, otherwise the params are stale or missing.
- `config.BETA = 0.4` makes F-beta favour precision over recall — deliberate: a false positive costs a real stake.
- `StandardScaler` is applied **only** for `logistic_regression`; tree models get raw features.

## Betting layer

`src/betting/test_strategy.py` holds strategies 1-6 plus the KPIs. Strategies 1-5 are non-ML (fixed stake, global and per-player martingale, delay-conditioned variants) and backtest the whole period. Strategy 6 is the only one that uses the models and the only one with a chronological split (first 80% train, last 20% evaluated). `compute_kpis_for_ml_strat` additionally tracks `max_consecutive_losses`, which is what actually kills a martingale.

## Conventions and known rough edges

- **Never open the project's CSVs directly.** To inspect one, write a short pandas script printing `df.shape`, `df.dtypes` and `df.head(5)`, and read its output. The files are 1-6 MB.
- **Config-first**: tunable constants (`YEAR`, `SEASONS`, `LEAGUES`, `BETA`, `REL_THR`, `ODDS`, `STAKE`, data dirs) live in `src/core/config.py`. Do not hardcode them elsewhere. Logging goes through `src/core/logger.py`.
- `YEAR` is **one ahead** of the running season (2027 during 2026-27); season filters read `season >= YEAR - nb_seasons`.
- Names lie about season counts: `LAST_3_SEASONS` holds **four** seasons, `update_raw_data` passes `nb_seasons=4`, and filenames still say `last3seasons`. Filenames are string literals in several modules, so renaming means touching every reader.
- **Feature engineering is duplicated** between `prepare_data_for_ml` (training) and `prepare_new_data_for_prediction` (inference) in `ml_features_processing.py`: the same eight ratios computed twice, and inference hardcodes the one-hot league columns that training derives from `OneHotEncoder`. Any new feature must go in both, or inference misaligns silently.
- Scraping is the fragile part. `create_matches_info_dataset_with_retries` absorbs Understat's spurious `InvalidMatch` rate-limit (3 attempts, 2-5 s sleep), returns `None` on other exceptions and raises `RuntimeError` when exhausted. In `create_team_metrics_dataset` that `RuntimeError` is only logged, so `df` keeps the *previous* league/season frame and gets processed twice. Fix the loop rather than working around it. (Diagnosed 2026-11-24: this path has never fired on the committed data, but the code is still wrong.)
- `notebooks/` is exploratory only; nothing in the pipeline imports from it.

## Answered — do not re-derive (put new open questions here)

- `scripts.evaluate_models` uses the random stratified split, not `train_test_split_by_date`.
- `date` is a plain string everywhere (no `pd.to_datetime` in the repo), so `train_test_split_by_date` sorts text lexicographically. With the ISO format in use, that matches true chronological order.
- Random seeds are **not** centralised: the literal `random_state=42` sits in `models.py:20`, `models.py:23`, `split.py:4` (signature default) and `tuning.py:16`. `LogisticRegression` gets none — harmless, `lbfgs` is deterministic. Nothing in `config.py`.
- **A full run is not known to be bit-for-bit reproducible** and nothing in the repo tests it: `GridSearchCV` and `XGBClassifier` both run with `n_jobs=-1`. Only the four offline artifacts are verified reproducible.
- `clean_features_dataset` has two producers — the CSV read by `evaluate_models`, and the same computation redone in memory by `create_results_dataset_for_ml_strat` (`test_strategy.py:498-499`). **They have not diverged**: measured 2026-11-24, both 4822 × 31, max absolute difference 1.8e-12. Latent risk, not a current bug — re-check after touching `create_features_dataset` or `prepare_data_for_ml`.

## Working with me

- Answer me in French. Code, identifiers and comments stay in English.
- No opportunistic refactoring: touch only what I asked for. If you spot something else worth fixing, mention it and move on.
- Scripts in `scripts/` are entry points: `main()` and argument parsing only. Logic, orchestration and anything worth testing or reusing belongs in `src/`.
- This project is also how I learn. Prefer solutions I can understand and explain back, and tell me when you rule out a more idiomatic but more opaque option.
- Keep this file under 100 lines.

@CONTEXTE-PERSO.md
