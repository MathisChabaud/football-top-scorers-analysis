import random
import time
import pandas as pd
from understatapi import UnderstatClient
from understatapi.exceptions import InvalidMatch
from src.core.config import SEASONS, LEAGUES, YEAR, RAW_DATA_DIR
from src.core.logger import logger


def get_league_players(league, season):
    """
    Get player data for a specific league and season.
    """

    understat = UnderstatClient()

    league_data = understat.league(league=league)

    players = league_data.get_player_data(season=season)

    return pd.DataFrame(players)


def get_player_matches(player_id):
    """
    Get match data for a specific player.
    """

    understat = UnderstatClient()

    player = understat.player(player=player_id)

    matches = player.get_match_data()

    return pd.DataFrame(matches)


def get_player_shots(player_id):
    """
    Get shot data for a specific player.
    """

    understat = UnderstatClient()

    player = understat.player(player=player_id)

    shots = player.get_shot_data()

    return pd.DataFrame(shots)


def get_player_seasons(player_id):
    """
    Get season data for a specific player.
    """

    understat = UnderstatClient()

    player = understat.player(player=str(player_id))

    seasons = player.get_season_data()['season']

    return seasons


def get_match_data(match_id):
    """
    Get detailed match data for a specific match.
    """

    understat = UnderstatClient()

    match = understat.match(match=str(match_id))

    match_data = match.get_match_info()

    return pd.DataFrame([match_data])


def get_league_matches(league, season):
    """
    Get match data for a specific league and season.
    """

    understat = UnderstatClient()

    league_data = understat.league(league=league)

    matches = league_data.get_match_data(season=season)

    return pd.DataFrame(matches)


def get_team_matches(team_name, season):
    """
    Get match data for a specific team and season.
    """

    understat = UnderstatClient()

    team = understat.team(team=team_name)

    matches = team.get_match_data(season=season)

    return pd.DataFrame(matches)


def create_all_players_dataset():
    """
    Create a dataset containing all players from all leagues and seasons.
    """

    all_data = []
    for league in LEAGUES:
        for season in SEASONS:
            df = get_league_players(league, season)
            df["season"]=season
            df["league"]=league
            all_data.append(df)
        
    final_df = pd.concat(all_data)
    final_df.to_csv(f"{RAW_DATA_DIR}/all_players_data.csv", index=False)


def create_top_n_european_scorers_dataset(top_n=50, nb_seasons=3):
    """
    Create a dataset of the top N European scorers over the last M seasons.
    """

    df = pd.read_csv(f"{RAW_DATA_DIR}/all_players_data.csv")
    df_last_seasons = df[df["season"] >= (YEAR - nb_seasons)]
    df_last_seasons = df_last_seasons.sort_values("season")

    grouped = df_last_seasons.groupby(["id", "player_name"]).agg(
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        matches=("games", "sum"),
        minutes=("time", "sum"),
        team=("team_title", "last")   # last club
    ).reset_index()

    grouped = grouped.sort_values(
        ["goals", "assists"],
        ascending=[False, False]
    ).head(top_n).reset_index(drop=True)

    grouped["rank"] = range(1, len(grouped) + 1)
    grouped["min/goal"] = grouped["minutes"] / grouped["goals"]

    return grouped


def create_players_goals_dataset(df_players, nb_seasons):
    """
    Create a dataset of all goals scored by the players in the provided DataFrame over the last M seasons.
    """

    # ids in string format
    ids = [str(pid) for pid in df_players["id"].tolist()]

    goals_data = []
    for player_id in ids:
        df_shots = get_player_shots(player_id)
        df_shots = df_shots[df_shots["season"].astype(int) >= (YEAR - nb_seasons)]
        goals = df_shots[df_shots["result"] == "Goal"]
        goals_data.append(goals)
    
    final_df = pd.concat(goals_data)

    final_df.to_csv(f"{RAW_DATA_DIR}/goals_top{len(df_players)}_last{nb_seasons}seasons.csv", index=False)
    
    return final_df


def create_players_matches_dataset(df_players, nb_seasons):
    """
    Create a dataset of all matches played by the players in the provided DataFrame over the last M seasons.
    """

    # ids in string format
    ids = [str(pid) for pid in df_players["id"].tolist()]

    matches_data = []
    for player_id in ids:
        df_matches = get_player_matches(player_id)
        df_matches = df_matches[df_matches["season"].astype(int) >= (YEAR - nb_seasons)]
        df_matches["player_id"] = player_id
        matches_data.append(df_matches)
    
    final_df = pd.concat(matches_data)

    final_df.to_csv(f"{RAW_DATA_DIR}/matches_top{len(df_players)}_last{nb_seasons}seasons.csv", index=False)

    return final_df


def update_raw_data():
    """
    Update all raw datasets by scraping Understat.
    """

    create_all_players_dataset()

    top_scorers_df = create_top_n_european_scorers_dataset(top_n=50, nb_seasons=3)

    goals_df = create_players_goals_dataset(top_scorers_df, nb_seasons=3)

    matches_df = create_players_matches_dataset(top_scorers_df, nb_seasons=3)


def create_matches_info_dataset(league, season):
    """
    Create a dataset containing detailed information about all matches in a specific league and season.
    """

    df_league_matches = get_league_matches(league, season)
    df_league_matches = df_league_matches[df_league_matches['isResult'] == True]
    match_ids = df_league_matches['id'].tolist()
    match_data_list = []
    for match_id in match_ids:
        match_data = get_match_data(match_id)
        match_data_list.append(match_data)
    df_match_info = pd.concat(match_data_list, ignore_index=True)

    return df_match_info


def create_matches_info_dataset_with_retries(league, season, retries=5):
    """
    create_matches_info_dataset with retries to handle potential rate limits or temporary issues with the Understat API.
    """

    for attempt in range(retries):
        try:
            return create_matches_info_dataset(league, season)

        except InvalidMatch as e:
            logger.warning(
                f"Fake InvalidMatch error on attempt likely due to rate limit"
                f"Attempt {attempt + 1}/{retries} failed "
                f"for {league} {season}: {e}"
            )
    
        except Exception as e:
            logger.exception(
                f"Unexpected error for "
                f"{league} {season}"
            )

        sleep_time = random.uniform(2, 5)
        logger.info(f"Sleeping {sleep_time:.2f}s before retry")
        time.sleep(sleep_time)

    raise RuntimeError(
        f"Failed after {retries} retries "
        f"for {league} {season}"
    )
