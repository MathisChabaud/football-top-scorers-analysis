import pandas as pd
from src.scraping.understat.understat_scraper import update_raw_data
from src.processing.dashboard_data_processing import build_clean_players_dataset
from src.processing.general_processing_functions import update_match_league_mapping, create_matches_goals_dataset
from src.processing.delay_score_processing import create_delay_score_dataset
from src.processing.ml_features_processing import create_clean_features_dataset, create_team_mappings
from src.core.logger import logger
from src.core.config import RAW_DATA_DIR


def main():

    logger.info("Updating raw data...")
    update_raw_data() # all_players_data.csv, matches.csv, goals.csv

    logger.info("Building clean players dataset...")
    build_clean_players_dataset() # all_players_data_clean.csv

    logger.info("Updating match-league mapping...")
    df_matches = pd.read_csv(f"{RAW_DATA_DIR}/matches_top50_last3seasons.csv")
    update_match_league_mapping(df_matches) # match_league_map.json

    logger.info("Creating matches-goals dataset...")
    create_matches_goals_dataset(save=True) # matches_goals_last3seasons.csv

    logger.info("Creating delay score dataset...")
    create_delay_score_dataset() # top50_players_delay_score.csv

    logger.info("Creating team mappings...")
    create_team_mappings() # team_id_mapping.json, all_team_metrics_beore_match.json

    logger.info("Creating clean features dataset...")
    create_clean_features_dataset() # clean_features_dataset.csv


if __name__ == "__main__":
    main()