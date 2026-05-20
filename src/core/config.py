# =========================
# YEARS
# =========================

YEAR = 2026

SEASONS = [2025, 2024, 2023, 2022, 2021]

LAST_3_SEASONS = ["2025", "2024", "2023"]

# =========================
# LEAGUES
# =========================

LEAGUES = ["Bundesliga", "EPL", "La_Liga", "Ligue_1", "Serie_A"]

LEAGUE_MAPPING = {
        "EPL": "Premier League",
        "La_Liga": "La Liga",
        "Bundesliga": "Bundesliga",
        "Serie_A": "Serie A",
        "Ligue_1": "Ligue 1"
    }

# =========================
# PATHS
# =========================

RAW_DATA_DIR = "data/raw"

PROCESSED_DATA_DIR = "data/processed"

DICTS_DIR = "data/dictionaries"

# =========================
# ML
# =========================

BETA = 0.4 # weight for the F-beta score

REL_THR = 0.7 # reliability score threshold (selecting only the most reliable players)

# =========================
# BETTING
# =========================

ODDS = 2.0

STAKE = 1