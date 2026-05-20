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
│   ├── processed/              # Cleaned datasets
│   └── dictionaries/           # Model parameters & mappings
│
├── scripts/
│   ├── update_data.py
│   ├── evaluate_models.py
│   ├── evaluate_strategies.py
│   └── predict.py
│
├── src/
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

### Create / Update Datasets

Run the data collection script to fetch latest match data from Understat:

```bash
python3 -m scripts.update_data
```

This script:
1. Updates and saves raw datasets
2. Rebuilds and saves features datasets
3. Updates and saves mappings/dictionaries

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
6. **Saves results** - Exports all evaluation metrics to `data/dictionaries/model_evaluation_results.json`.

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
- `data/processed/strategy_1_5_results.csv` contains KPIs for strategies 1 to 5 (non-ML based) across various parameter configurations.
- `data/processed/strategy_6_results.csv` contains KPIs for strategy 6 (progressive stake using ML predictions) across various parameter configurations. For Strategy 6, the ML model is trained on the first 80% of matches (chronologically) and applied on the most recent 20%.

## Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run src/dashboard/app.py
```

**Dashboard features:**
- "⚽ Top Scorers" - Filter by league, season, and cumulative periods to view player rankings
- "⏳ Delay Score" - Identify players in scoring droughts with customizable thresholds
