import os
from functools import partial

import pandas as pd

from src.scraping.understat.understat_scraper import update_raw_data
from src.processing.dashboard_data_processing import build_clean_players_dataset
from src.processing.general_processing_functions import update_match_league_mapping, create_matches_goals_dataset
from src.processing.delay_score_processing import create_delay_score_dataset
from src.processing.ml_features_processing import create_clean_features_dataset, create_team_mappings
from src.core.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DICTS_DIR
from src.core.logger import logger


def refresh_match_league_mapping():
    """
    Refresh match_league_map.json from the raw matches file.

    Thin wrapper so the step fits SCRAPE_STEPS: update_match_league_mapping takes
    the matches DataFrame, not a path.
    """

    df_matches = pd.read_csv(f"{RAW_DATA_DIR}/matches_top50_last3seasons.csv")
    update_match_league_mapping(df_matches)


# Network half. Each entry is (label, callable, [files written]).
SCRAPE_STEPS = [
    (
        "Updating raw data",
        update_raw_data,
        [
            f"{RAW_DATA_DIR}/all_players_data.csv",
            f"{RAW_DATA_DIR}/goals_top50_last3seasons.csv",
            f"{RAW_DATA_DIR}/matches_top50_last3seasons.csv",
        ],
    ),
    (
        "Updating match-league mapping",
        refresh_match_league_mapping,
        [f"{DICTS_DIR}/match_league_map.json"],
    ),
    (
        "Creating team mappings",
        create_team_mappings,
        [
            f"{DICTS_DIR}/team_id_mapping.json",
            f"{DICTS_DIR}/all_team_metrics_before_match.json",
        ],
    ),
]

# Offline half. Reads data/raw/ and data/dictionaries/, writes only to
# data/processed/. Each step reads what the previous one wrote, so this list is
# the sequencing contract: do not reorder it.
REBUILD_STEPS = [
    (
        "Building clean players dataset",
        build_clean_players_dataset,
        [f"{PROCESSED_DATA_DIR}/all_players_data_clean.csv"],
    ),
    (
        "Creating matches-goals dataset",
        partial(create_matches_goals_dataset, save=True),
        [f"{PROCESSED_DATA_DIR}/matches_goals_top50_last3seasons.csv"],
    ),
    (
        "Creating delay score dataset",
        create_delay_score_dataset,
        [f"{PROCESSED_DATA_DIR}/top50_players_delay_score.csv"],
    ),
    (
        "Creating clean features dataset",
        create_clean_features_dataset,
        [f"{PROCESSED_DATA_DIR}/clean_features_dataset.csv"],
    ),
]


def run_steps(steps):
    """
    Run a list of (label, callable, outputs) in order, logging each step.
    """

    for position, (label, step, outputs) in enumerate(steps, start=1):
        logger.info("[%d/%d] %s", position, len(steps), label)
        step()
        for output in outputs:
            logger.info("        wrote %s", output)


def scrape():
    """
    Network half: refresh data/raw/ and the three mapping dictionaries.
    """

    run_steps(SCRAPE_STEPS)


def rebuild():
    """
    Offline half: rebuild data/processed/ from data/raw/ and the dictionaries.
    """

    # data/processed/ is gitignored, so it does not exist in a fresh clone.
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    run_steps(REBUILD_STEPS)


def main():

    logger.info("=== Scraping Understat (%d steps) ===", len(SCRAPE_STEPS))
    scrape()

    logger.info("=== Rebuilding data/processed/ (%d steps) ===", len(REBUILD_STEPS))
    rebuild()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
