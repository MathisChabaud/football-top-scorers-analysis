# Football Top Scorers Analysis

Machine learning and statistical analysis project for predicting whether a football player will score in an upcoming match, using historical player/team statistics and betting-oriented evaluation metrics. The project focuses on the top 50 European scorers over the last 3 seasons.

Goals:
- Analyze scoring patterns and player performance trends
- Detect scoring droughts and rhythm anomalies
- Predict goal probabilities for upcoming matches
- Simulate and evaluate betting strategies based on ML predictions
- Dashboard for visualizing top scorers ranking and player delay analysis (time since last goal, average delay...)

## Data Sources

All data used in this project is sourced from [understat](https://understat.com/), a comprehensive football analytics platform providing detailed shot-based statistics for European leagues. Data was collected using the understatapi Python library, which provides a clean interface to scrape Understat's public data.

Leagues covered:
- Premier League (England)
- La Liga (Spain)
- Bundesliga (Germany)
- Ligue 1 (France)
- Serie A (Italy)

## Project Structure

```
football-top-scorers-analysis/
│
├── data/
│   ├── raw/                    # Raw data from Understat
│   ├── processed/              # Cleaned datasets (generated, not tracked)
│   └── dictionaries/           # Model parameters & mappings
│
├── results/                    # Model evaluation metrics & strategy backtest results
│
├── scripts/
│   ├── update_data.py
│   ├── rebuild_processed.py
│   ├── evaluate_models.py
│   ├── evaluate_strategies.py
│   └── predict.py
│
├── src/
│   ├── pipeline.py             # step tables + scrape() / rebuild()
│   ├── core/
│   ├── betting/
│   ├── dashboard/
│   ├── ml/
│   ├── processing/
│   └── scraping/
│
├── notebooks/                  # Jupyter notebooks for exploration
│
├── requirements.txt            # Python dependencies
├── .gitignore                  
└── README.md
```

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/MathisChabaud/football-top-scorers-analysis.git
cd football-top-scorers-analysis
```

2. **Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Rebuild the processed datasets**

`data/processed/` is generated and not tracked by git, so a fresh clone has none of
it. Build it once, from the repository root. No network access is required:

```bash
python3 -m scripts.rebuild_processed
```

Takes a few seconds. After this, the dashboard, the model evaluation and the
strategy backtests all run.

> All commands in this README must be run from the repository root: data paths in
> `src/core/config.py` are relative.

## Configuration

Global project constants are centralized in `src/core/config.py`. This file contains all configurable parameters, making it easy to modify behavior without digging through multiple files.

### Key Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `YEAR` | Current year used for data filtering and season calculations |
| `LEAGUES` | List of leagues to scrape from Understat (e.g., EPL, La_Liga, Serie_A) |
| `BETA` | F-beta score weight for evaluation and model tuning (β < 1 emphasizes precision over recall) |
| `REL_THR` | Minimum player reliability score. Only players above this threshold are kept for ML training and betting strategy simulation. Filters out inconsistent players. |

## Data Pipeline

The pipeline is a sequential chain of files on disk: each step reads what the
previous one wrote. It splits in two, and the split matters.

- **Offline steps** rebuild `data/processed/` from `data/raw/` and
  `data/dictionaries/`, both of which are committed. This is `scripts.rebuild_processed`.
- **Network steps** scrape Understat to refresh `data/raw/` and the three mapping
  dictionaries. This is `scripts.update_data`.

### Rebuild Processed Datasets (offline)

```bash
python3 -m scripts.rebuild_processed
python3 -m scripts.rebuild_processed --dry-run   # list the target files, write nothing
```

Four steps, in this order, each reading what the previous one wrote:

| Step | Writes |
|---|---|
| `build_clean_players_dataset` | `data/processed/all_players_data_clean.csv` |
| `create_matches_goals_dataset` | `data/processed/matches_goals_top50_last3seasons.csv` |
| `create_delay_score_dataset` | `data/processed/top50_players_delay_score.csv` |
| `create_clean_features_dataset` | `data/processed/clean_features_dataset.csv` |

Reads `data/raw/` and `data/dictionaries/`, writes **only** to `data/processed/`,
makes no HTTP call. Missing inputs are reported by name before any work starts.

This is the script to run after cloning, and the one to re-run after changing any
feature-engineering code.

### Create / Update Datasets (network)

Fetches the latest match data from Understat and rebuilds every artifact:

```bash
python3 -m scripts.update_data
```

This script runs both halves of the pipeline in turn:

1. `scrape()` — three network steps: refreshes the three raw CSVs, then
   `match_league_map.json`, then `team_id_mapping.json` and
   `all_team_metrics_before_match.json`.
2. `rebuild()` — the same four offline steps as `scripts.rebuild_processed`.

Both halves live in `src/pipeline.py`, which holds the single definition of the
step order (`SCRAPE_STEPS` and `REBUILD_STEPS`) plus `scrape()` and `rebuild()`.

> **This script overwrites `data/raw/`, which cannot currently be re-scraped.**
> Understat has been unreachable since early November 2026, so a partial run would
> replace the only working copy of the raw data with a truncated one. It also
> rewrites `match_league_map.json`, `team_id_mapping.json` and
> `all_team_metrics_before_match.json`, which come from a separate scrape of ~2900
> HTTP calls and have no other source. Use `scripts.rebuild_processed` unless you
> specifically intend to re-scrape.

### Machine Learning Workflow

Tuning, training and evaluating models:

```bash
python3 -m scripts.evaluate_models
```

This script:
1. **Loads and filters data** - Loads processed match data with engineered features, keeping only players with `reliability_score ≥ REL_THR`.
2. **Splits data** - Performs random train/test split.
3. **Tunes hyperparameters** - Finds and saves optimal parameters for Logistic Regression, Random Forest, and XGBoost to `data/dictionaries/best_ml_params.json`. Optimization uses F-beta score with 5-fold cross-validation on the training set only.
4. **Trains & evaluates models** - Trains each model with its optimal parameters and evaluates performance.
5. **Optimizes threshold** - Finds the optimal probability threshold for `y_pred` and computes corresponding metrics.
6. **Saves results** - Exports all evaluation metrics to `results/model_evaluation_results.json`.

### Making Predictions for Future Matches

Predict whether a specific player will score in their next match.

**Setup:**
1. Open `scripts/predict.py`
2. Set the following variables:
   - `PLAYER_ID` - Target player's unique identifier
   - `MODEL_NAME` - Model to use (`logistic_regression`, `random_forest`, or `xgboost`)

**Run prediction:**

```bash
python3 -m scripts.predict
```

- The model is trained on all available data (filtered by `reliability_score ≥ REL_THR`)
- Features for the upcoming match are saved to `data/processed/new_data.csv`
- Output includes prediction (1 (will score) or 0 (will not score)) and probability (confidence score between 0 and 1)

### Betting Strategies

Simulate and backtest betting strategies using historical data:

```bash
python3 -m scripts.evaluate_strategies
```

This script generates two result datasets:
- `results/strategy_1_5_results.csv` contains KPIs for strategies 1 to 5 (non-ML based) across various parameter configurations.
- `results/strategy_6_results.csv` contains KPIs for strategy 6 (progressive stake using ML predictions) across various parameter configurations. For Strategy 6, the ML model is trained on the first 80% of matches (chronologically) and applied on the most recent 20%.

## Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run src/dashboard/app.py
```

**Dashboard features:**
- "⚽ Top Scorers" - Filter by league, season, and cumulative periods to view player rankings
- "⏳ Delay Score" - Identify players in scoring droughts with customizable thresholds
