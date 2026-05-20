import pandas as pd
import numpy as np
import json
import os
from src.scraping.understat.understat_scraper import get_match_data
from src.core.config import DICTS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.core.logger import logger


def save_dict_json(dictionary, dict_name):
    """
    Save a dictionary as a JSON file with the specified name.
    """

    filepath = f"{DICTS_DIR}/{dict_name}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)
    logger.info(f"Dictionary saved to {filepath}")


def load_dict_json(dict_name):
    """
    Load a dictionary from a JSON file with the specified name.
    """

    filepath = f"{DICTS_DIR}/{dict_name}.json"
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
    logger.info(f"Dictionary loaded from {filepath}")


def update_match_league_mapping(df_matches, path=f"{DICTS_DIR}/match_league_map.json"):
    """
    Update the match_league_map dictionary based on the current matches data.
    """

    # Load existing dictionary or create a new one if it doesn't exist
    if os.path.exists(path):
        match_league_map = load_dict_json("match_league_map")
    else:
        match_league_map = {}

    current_match_ids = set(df_matches["id"].astype(str).unique())
    existing_match_ids = set(match_league_map.keys())

    # Add missing keys
    missing_ids = current_match_ids - existing_match_ids

    dict_league = {
        "EPL": "EPL",
        "La liga": "La_Liga",
        "Bundesliga": "Bundesliga",
        "Serie A": "Serie_A",
        "Ligue 1": "Ligue_1"
    }

    for match_id in missing_ids:

        match_data = get_match_data(match_id)

        if len(match_data) > 0 and "league" in match_data.columns:
            league = match_data["league"].iloc[0]
            match_league_map[match_id] = dict_league.get(league, league)

        else:
            match_league_map[match_id] = None

    # Remove extra keys

    extra_ids = existing_match_ids - current_match_ids

    for match_id in extra_ids:
        del match_league_map[match_id]

    # Save updated dictionary
    save_dict_json(match_league_map, "match_league_map")

    return match_league_map


def determine_h_a(row):
    """
    Determine if the player is home or away based on the player's team
    """

    # If h_a is already filled, keep it
    if pd.isna(row['h_a']):
        teams = []
        
        # Retrieve the player's team(s) from the "player_team" column (which may contain multiple teams separated by commas)
        if pd.notna(row['player_team']):
            teams = [t.strip() for t in str(row['player_team']).split(',')]
        
        # Check each team
        for team in teams:
            if team == row['h_team']:
                return 'h'
            elif team == row['a_team']:
                return 'a'
    
    return row['h_a']


def create_matches_goals_dataset(save=False):
    """
    Create a dataset combining matches and goals data, with features related to the time between goals.
    """

    # Load data
    df_matches = pd.read_csv(f"{RAW_DATA_DIR}/matches_top50_last3seasons.csv")
    df_goals = pd.read_csv(f"{RAW_DATA_DIR}/goals_top50_last3seasons.csv")
    df_players = pd.read_csv(f"{RAW_DATA_DIR}/all_players_data.csv")
    
    # Keep only relevant columns
    df_matches = df_matches[["id", "season", "player_id", "goals", "xG", "time", "date", "position", "h_team", "a_team"]]
    df_goals = df_goals[["minute", "player_id", "player", "match_id", "h_a", "situation"]]

    # Merge matches and goals : we keep all matches and add goal information when the player scored(left join)
    df_matches_goals = df_matches.merge(df_goals, left_on=["player_id", "id"], right_on=["player_id", "match_id"], how="left")
    df_matches_goals.drop(columns=["match_id"], inplace=True)
    df_matches_goals = df_matches_goals.sort_values(["player_id", "date", "minute"])

    # Create league column
    match_league_map = load_dict_json("match_league_map")
    df_matches_goals['league'] = df_matches_goals['id'].astype(str).map(match_league_map)

    # Create player team column
    player_team_map = df_players[['id', 'season', 'team_title', 'league']].drop_duplicates(
        subset=['id', 'season', 'league']
    ).copy()
    df_matches_goals = df_matches_goals.merge(
        player_team_map[['id', 'season', 'team_title', 'league']],
        left_on=['player_id', 'season', 'league'],
        right_on=['id', 'season', 'league'],
        how='left',
        suffixes=('', '_player')
    )
    df_matches_goals = df_matches_goals.rename(columns={'team_title': 'player_team'})
    if 'id_player' in df_matches_goals.columns:
        df_matches_goals = df_matches_goals.drop(columns=['id_player'])

    # Filling nan

    # "player name"
    df_matches_goals["player"] = df_matches_goals["player_id"].map(df_matches_goals.dropna(subset=["player"]).drop_duplicates(subset=["player_id"]).set_index("player_id")["player"])
    # "h_a"
    mask_h_a_nan = df_matches_goals["h_a"].isna()
    df_matches_goals.loc[mask_h_a_nan, "h_a"] = df_matches_goals.loc[mask_h_a_nan].apply(determine_h_a, axis=1)
    # "situation"
    df_matches_goals["situation"] = df_matches_goals["situation"].fillna("NoGoal")

    # Create features related to time between goals

    # "cumulative_time" : cumulated playing time for each player
    df_unique_matches = df_matches_goals.drop_duplicates(subset=["player_id", "id"]).copy() # we keep only one row per match and player to calculate the cumulative time correctly
    df_unique_matches["cumulative_time"] = df_unique_matches.groupby("player_id")["time"].cumsum()
    df_matches_goals = df_matches_goals.merge(
        df_unique_matches[["player_id", "id", "cumulative_time"]],
        on=["player_id", "id"],
        how="left",
    )
    # "match_duration" : more than 90 minutes if the player scores in the added time
    df_matches_goals["match_duration"] = np.where(
        df_matches_goals["minute"] > 90,
        df_matches_goals["minute"],
        90
    )
    # "start_time" : 0 if the player starts the match, otherwise the time he enters the match
    df_matches_goals["start_time"] = np.where(
        df_matches_goals["position"] == "Sub",
        df_matches_goals["match_duration"] - df_matches_goals["time"],
        0
    )
    # "last_goal_time" : absolute time of the last goal scored by the player
    df_matches_goals["last_goal_time"] = np.where(
        df_matches_goals["minute"].notna(),
        df_matches_goals["cumulative_time"] - df_matches_goals["time"] + (df_matches_goals["minute"]-df_matches_goals["start_time"]),
        np.nan
    )
    df_matches_goals["last_goal_time"] = df_matches_goals.groupby("player_id")["last_goal_time"].ffill()
    # "time_between_goals" : time between the current goal and the previous one (=0 if no goal scored in the match)
    df_matches_goals["time_between_goals"] = df_matches_goals.groupby("player_id")["last_goal_time"].diff()
    # "minutes_since_last_goal" : time between the current moment and the last goal scored by the player
    df_matches_goals["minutes_since_last_goal"] = df_matches_goals["cumulative_time"] - df_matches_goals["last_goal_time"]
    df_matches_goals["minutes_since_last_goal"] = np.where(df_matches_goals["minutes_since_last_goal"] < 0, 0, df_matches_goals["minutes_since_last_goal"])
    # "minutes_since_last_goal_before_match" : time since the last goal scored by the player before the match
    df_matches_goals["minutes_since_last_goal_before_match"] = df_matches_goals.groupby("player_id")["minutes_since_last_goal"].shift(1)
    df_matches_goals["minutes_since_last_goal_before_match"] = df_matches_goals.groupby(
        ["player_id", "id"]
    )["minutes_since_last_goal_before_match"].transform("first")

    # rename and delete columns
    df_matches_goals.rename(columns={"id": "match_id"}, inplace=True)
    df_matches_goals.drop(columns=["match_duration"], inplace=True)

    if save:
        df_matches_goals.to_csv(f"{PROCESSED_DATA_DIR}/matches_goals_top50_last3seasons.csv", index=False)

    return df_matches_goals


def create_dict_team_id_name(df_team_metrics):
    """
    Create a dictionary mapping team names to team IDs from the team metrics dataframe.
    """

    team_name_to_id = df_team_metrics.drop_duplicates(subset='team_name', keep='last').set_index('team_name')['team_id'].to_dict()
    
    save_dict_json(team_name_to_id, "team_id_mapping")

    return team_name_to_id


def create_dict_matchid_teamid_metrics(df_team_metrics_before_match):
    """
    Create a dictionary mapping a composite key of match_id and team_id to the corresponding team metrics before the match.
    """

    # Create composite key
    df_team_metrics_before_match['composite_key'] = df_team_metrics_before_match['match_id'].astype(str) + "_" + df_team_metrics_before_match['team_id'].astype(str)
    
    # Convert to dictionary
    dict_metrics = df_team_metrics_before_match.set_index('composite_key').to_dict(orient='index')
    
    save_dict_json(dict_metrics, "all_team_metrics_before_match")

    return dict_metrics
