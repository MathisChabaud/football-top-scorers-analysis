import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.scraping.understat.understat_scraper import create_matches_info_dataset_with_retries, get_team_matches
from src.processing.general_processing_functions import create_dict_matchid_teamid_metrics, create_dict_team_id_name, load_dict_json
from src.core.config import PROCESSED_DATA_DIR, LEAGUES, LAST_3_SEASONS
from src.core.logger import logger


def calculate_team_metrics_global(df_matches):
    """
    Compute global team metrics for each match and team, then cumulative metrics before each match
    Returns a dataframe with one row per team and match, containing both match-level and cumulative metrics
    """

    # Convert columns to numeric
    numeric_cols = ['h_goals', 'a_goals', 'h_xg', 'a_xg']
    for col in numeric_cols:
        df_matches[col] = pd.to_numeric(df_matches[col], errors='coerce')
    
    df_matches = df_matches.sort_values('date')
    all_team_metrics = []
    
    for idx, match in df_matches.iterrows():
        # Home metrics
        home_metrics = {
            'date': match['date'],
            'match_id': match['id'],
            'league': match['league'],
            'season': match['season'],
            'team_id': match['h'],
            'team_name': match['team_h'],
            'h_a': 'h',
            'opponent': match['team_a'],
            'opponent_id': match['a'],
            'goals_scored': match['h_goals'],
            'goals_conceded': match['a_goals'],
            'xg': match['h_xg'] if pd.notna(match['h_xg']) else 0,
            'xga': match['a_xg'] if pd.notna(match['a_xg']) else 0,
            'result': 'win' if match['h_goals'] > match['a_goals'] else 'draw' if match['h_goals'] == match['a_goals'] else 'loss'
        }
        
        # Away metrics
        away_metrics = {
            'date': match['date'],
            'match_id': match['id'],
            'league': match['league'],
            'season': match['season'],
            'team_id': match['a'],
            'team_name': match['team_a'],
            'h_a': 'a',
            'opponent': match['team_h'],
            'opponent_id': match['h'],
            'goals_scored': match['a_goals'],
            'goals_conceded': match['h_goals'],
            'xg': match['a_xg'] if pd.notna(match['a_xg']) else 0,
            'xga': match['h_xg'] if pd.notna(match['h_xg']) else 0,
            'result': 'win' if match['a_goals'] > match['h_goals'] else 'draw' if match['a_goals'] == match['h_goals'] else 'loss'
        }
        
        all_team_metrics.extend([home_metrics, away_metrics])
    
    df_team_metrics = pd.DataFrame(all_team_metrics)
    
    # Convert columns to numeric
    numeric_cols = ['goals_scored', 'goals_conceded', 'xg', 'xga']
    for col in numeric_cols:
        df_team_metrics[col] = pd.to_numeric(df_team_metrics[col], errors='coerce').fillna(0)
    
    df_team_metrics = df_team_metrics.sort_values(['team_id', 'date'])
    
    # === Global Metrics (Cumulative) ===
    df_team_metrics['total_matches'] = df_team_metrics.groupby('team_id').cumcount() + 1
    
    # Count results (using cumsum on booleans)
    df_team_metrics['is_win'] = (df_team_metrics['result'] == 'win').astype(int)
    df_team_metrics['is_draw'] = (df_team_metrics['result'] == 'draw').astype(int)
    df_team_metrics['is_loss'] = (df_team_metrics['result'] == 'loss').astype(int)
    
    df_team_metrics['total_wins'] = df_team_metrics.groupby('team_id')['is_win'].cumsum()
    df_team_metrics['total_draws'] = df_team_metrics.groupby('team_id')['is_draw'].cumsum()
    df_team_metrics['total_losses'] = df_team_metrics.groupby('team_id')['is_loss'].cumsum()
    
    df_team_metrics['pts'] = df_team_metrics['total_wins'] * 3 + df_team_metrics['total_draws']
    df_team_metrics['clean_sheet'] = np.where(df_team_metrics['goals_conceded'] == 0, 1, 0)

    # Cumulative sums for goals, xG, etc.
    df_team_metrics['total_goals_scored'] = df_team_metrics.groupby('team_id')['goals_scored'].cumsum()
    df_team_metrics['total_goals_conceded'] = df_team_metrics.groupby('team_id')['goals_conceded'].cumsum()
    df_team_metrics['total_xg'] = df_team_metrics.groupby('team_id')['xg'].cumsum()
    df_team_metrics['total_xga'] = df_team_metrics.groupby('team_id')['xga'].cumsum()
    df_team_metrics['nb_clean_sheet'] = df_team_metrics.groupby('team_id')['clean_sheet'].cumsum()
    
    # Drop intermediate columns
    drop_cols = ['is_win', 'is_draw', 'is_loss', 'clean_sheet']
    df_team_metrics = df_team_metrics.drop(columns=drop_cols)
    
    return df_team_metrics


def create_team_metrics_dataset():
    """
    For each league and season, create a dataset with team metrics for each match and team.
    """

    all_team_metrics = []
    for league in LEAGUES:
        for season in LAST_3_SEASONS:
            logger.info(f"Processing {league} {season}")

            try:
                df = create_matches_info_dataset_with_retries(league, season)
                df_team_metrics = calculate_team_metrics_global(df)
                all_team_metrics.append(df_team_metrics)

            except RuntimeError as e:
                logger.error(str(e))        
    
    df_all_team_metrics = pd.concat(all_team_metrics, ignore_index=True)

    return df_all_team_metrics


def create_team_features_before_match(df_all_team_metrics):
    """
    For each match and team, create features with cumulative metrics before the match (points, goals scored, xG, etc.)
    """

    df_all_team_metrics['pts_before_match'] = df_all_team_metrics.groupby(['team_id', 'season'])['pts'].shift(1).fillna(0).astype(int)
    df_all_team_metrics['total_matches_before_match'] = df_all_team_metrics.groupby(['team_id', 'season'])['total_matches'].shift(1).fillna(0).astype(int)
    df_all_team_metrics['total_goals_scored_before_match'] = df_all_team_metrics.groupby(['team_id', 'season'])['total_goals_scored'].shift(1).fillna(0).astype(int)
    df_all_team_metrics['total_goals_conceded_before_match'] = df_all_team_metrics.groupby(['team_id', 'season'])['total_goals_conceded'].shift(1).fillna(0).astype(int)
    df_all_team_metrics['total_xg_before_match'] = df_all_team_metrics.groupby(['team_id', 'season'])['total_xg'].shift(1).fillna(0)
    df_all_team_metrics['total_xga_before_match'] = df_all_team_metrics.groupby(['team_id', 'season'])['total_xga'].shift(1).fillna(0)
    df_all_team_metrics['clean_sheet_rate_before_match'] = df_all_team_metrics.groupby(['team_id', 'season'])['nb_clean_sheet'].shift(1).fillna(0) / df_all_team_metrics['total_matches_before_match'].replace(0, 1)

    df_all_team_metrics = df_all_team_metrics[['match_id', 'team_id', 'team_name', 'league', 'h_a', 'total_matches_before_match', 
                    'total_goals_scored_before_match', 'total_goals_conceded_before_match', 'total_xg_before_match', 'total_xga_before_match',
                    'clean_sheet_rate_before_match', 'pts_before_match', 'opponent_id']].copy()

    return df_all_team_metrics


def map_player_team_id(df_matches_goals, team_mapping):
    """
    Map player team names to team IDs using the provided team_mapping.
    Handles cases where player_team may contain multiple teams (e.g., 'team1,team2').
    """

    def get_team_id(row):
        player_team = row['player_team']
        
        # Handle missing player team
        if pd.isna(player_team):
            return None
        
        # Simple case: player_team is a single team name
        if ',' not in str(player_team):
            return team_mapping.get(player_team)
        
        # Case: 'team1,team2'
        teams = str(player_team).split(',')
        team1 = teams[0].strip()
        team2 = teams[1].strip() if len(teams) > 1 else None
        
        # Check if team1 matches either home or away team
        if team1 == row['h_team'] or team1 == row['a_team']:
            return team_mapping.get(team1)
        
        # Check if team2 matches either home or away team
        if team2 and (team2 == row['h_team'] or team2 == row['a_team']):
            return team_mapping.get(team2)
        
        # Fallback: take team1
        return team_mapping.get(team1)
    
    df_matches_goals['player_team_id'] = df_matches_goals.apply(get_team_id, axis=1)

    return df_matches_goals
    

def create_features_dataset():
    """
    Create a dataset with features for each top 50 player's match.
    """

    # load data
    df_matches_goals = pd.read_csv(f"{PROCESSED_DATA_DIR}/matches_goals_top50_last3seasons.csv")
    df_delay_analysis = pd.read_csv(f"{PROCESSED_DATA_DIR}/top50_players_delay_score.csv")

    # reliability score and delay score
    df_matches_goals["reliability_score"] = df_matches_goals.apply(lambda row: df_delay_analysis.loc[df_delay_analysis['player_id'] == row['player_id'], 'reliability_score'].values[0] if row['player_id'] in df_delay_analysis['player_id'].values else 0, axis=1)
    df_matches_goals["delay_score"] = df_matches_goals.apply(lambda row: (row["minutes_since_last_goal_before_match"] - df_delay_analysis.loc[df_delay_analysis['player_id'] == row['player_id'], 'avg_delay'].values[0]) / df_delay_analysis.loc[df_delay_analysis['player_id'] == row['player_id'], 'std_delay'].values[0] if row['player_id'] in df_delay_analysis['player_id'].values else 0, axis=1)

    # cumulated player minutes, goals and xG in last 3 matches
    df_match_totals = df_matches_goals.groupby(['player_id', 'match_id', 'date']).agg(total_time=('time', 'last'), total_goals=('goals', 'last'), total_xG=('xG', 'last')).reset_index()
    df_match_totals = df_match_totals.sort_values(['player_id', 'date'])
    df_match_totals['time_last_3_matches'] = (df_match_totals.groupby('player_id')['total_time'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())).fillna(0)
    df_match_totals['goals_last_3_matches'] = (df_match_totals.groupby('player_id')['total_goals'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())).fillna(0)
    df_match_totals['xG_last_3_matches'] = (df_match_totals.groupby('player_id')['total_xG'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())).fillna(0)
    df_matches_goals = df_matches_goals.merge(
        df_match_totals[['player_id', 'match_id', 'time_last_3_matches', 'goals_last_3_matches', 'xG_last_3_matches']],
        on=['player_id', 'match_id'],
        how='left'
    )

    # team_id
    team_mapping = load_dict_json("team_id_mapping")
    df_matches_goals = map_player_team_id(df_matches_goals, team_mapping)
    #TODO : handle cases where the two player's teams play in the same match.

    # add team features before match, using the dict with all team metrics.
    dict = load_dict_json("all_team_metrics_before_match")
    df_matches_goals['key'] = df_matches_goals['match_id'].astype(str) + '_' + df_matches_goals['player_team_id'].astype(str)

    # add other team features
    df_matches_goals['total_team_matches_before_match'] = df_matches_goals['key'].map(lambda x: dict.get(x, {}).get('total_matches_before_match', None)).astype(int)
    df_matches_goals['total_team_goals_scored_before_match'] = df_matches_goals['key'].map(lambda x: dict.get(x, {}).get('total_goals_scored_before_match', None)).astype(int)
    df_matches_goals['total_team_xg_before_match'] = df_matches_goals['key'].map(lambda x: dict.get(x, {}).get('total_xg_before_match', None))
    df_matches_goals['team_pts_before_match'] = df_matches_goals['key'].map(lambda x: dict.get(x, {}).get('pts_before_match', None)).astype(int)

    # add opponent team features
    df_matches_goals['opponent_team_id'] = df_matches_goals['key'].map(lambda x: dict.get(x, {}).get('opponent_id', None)).astype(int)
    df_matches_goals['opponent_key'] = df_matches_goals['match_id'].astype(str) + '_' + df_matches_goals['opponent_team_id'].astype(str)
    df_matches_goals['total_opponent_matches_before_match'] = df_matches_goals['opponent_key'].map(lambda x: dict.get(x, {}).get('total_matches_before_match', None)).astype(int)
    df_matches_goals['total_opponent_goals_conceded_before_match'] = df_matches_goals['opponent_key'].map(lambda x: dict.get(x, {}).get('total_goals_conceded_before_match', None)).astype(int)
    df_matches_goals['total_opponent_xga_before_match'] = df_matches_goals['opponent_key'].map(lambda x: dict.get(x, {}).get('total_xga_before_match', None))
    df_matches_goals['opponent_clean_sheet_rate_before_match'] = df_matches_goals['opponent_key'].map(lambda x: dict.get(x, {}).get('clean_sheet_rate_before_match', None))
    df_matches_goals['opponent_pts_before_match'] = df_matches_goals['opponent_key'].map(lambda x: dict.get(x, {}).get('pts_before_match', None)).astype(int)

    # target variable
    df_matches_goals['scored'] = np.where(df_matches_goals['goals'] > 0, 1, 0)

    # drop lines where minutes_since_last_goal_before_match is nan (very first matches of players)
    df_matches_goals = df_matches_goals.dropna(subset=['minutes_since_last_goal_before_match'])

    # keep only relevant columns for prediction
    df_matches_goals = df_matches_goals[['match_id', 'date', 'player_id', 'position', 'league', 'h_a', 'minutes_since_last_goal_before_match',
                                        'reliability_score', 'delay_score', 'time_last_3_matches',
                                        'goals_last_3_matches', 'xG_last_3_matches', 'total_team_matches_before_match', 
                                        'team_pts_before_match', 'total_team_goals_scored_before_match', 'total_team_xg_before_match', 
                                        'total_opponent_matches_before_match',  'opponent_pts_before_match', 'total_opponent_goals_conceded_before_match',
                                        'total_opponent_xga_before_match', 'opponent_clean_sheet_rate_before_match', 'scored']]

    # drop duplicates (where a player scores multiple goals in a match)
    df_matches_goals = df_matches_goals.drop_duplicates(ignore_index=True)

    return df_matches_goals    
                                                                    

def prepare_data_for_ml(features_data, skip_ids=True, save_features=False):
    """
    Prepare the features dataset for machine learning by converting categorical features and creating useful ratios.
    """

    # features preparation : convert categorical features
    features_data['starting'] = np.where(features_data['position'] == 'Sub', 0, 1)
    features_data['h_a'] = np.where(features_data['h_a']=='h', 1, 0)
    # encoding of league with one hot encoding : now one column per league
    enc = OneHotEncoder()
    league_encoded = enc.fit_transform(features_data[['league']])
    league_df = pd.DataFrame(league_encoded.toarray(), columns=enc.get_feature_names_out(['league']))
    df = pd.concat([features_data, league_df], axis=1)

    df = df.drop(columns=['position','league'])

    # useful ratios : 
    df['ratio_min_goals_last_3_matches'] = np.where(
        df['goals_last_3_matches'] > 0,
        df['time_last_3_matches'] / df['goals_last_3_matches'],
        300
    )

    df['ratio_min_xg_last_3_matches'] = np.where(
        df['xG_last_3_matches'] > 0,
        df['time_last_3_matches'] / df['xG_last_3_matches'],
        300
    )

    df['ratio_pts_match'] = np.where(
        df['total_team_matches_before_match'] > 0,
        df['team_pts_before_match'] / df['total_team_matches_before_match'],
        0
    )

    df['ratio_pts_match_opponent'] = np.where(
        df['total_opponent_matches_before_match'] > 0,
        df['opponent_pts_before_match'] / df['total_opponent_matches_before_match'],
        0
    )

    df['ratio_goals_scored'] = np.where(
        df['total_team_matches_before_match'] > 0,
        df['total_team_goals_scored_before_match'] / df['total_team_matches_before_match'],
        0
    )

    df['ratio_xG'] = np.where(
        df['total_team_matches_before_match'] > 0,
        df['total_team_xg_before_match'] / df['total_team_matches_before_match'],
        0
    )

    df['ratio_goals_conceded_opponent'] = np.where(
        df['total_opponent_matches_before_match'] > 0,
        df['total_opponent_goals_conceded_before_match'] / df['total_opponent_matches_before_match'],
        0
    )

    df['ratio_xga_opponent'] = np.where(
        df['total_opponent_matches_before_match'] > 0,
        df['total_opponent_xga_before_match'] / df['total_opponent_matches_before_match'],
        0
    )

    if skip_ids:
        df = df.drop(columns=['match_id', 'date', 'player_id'])

    if save_features:
        df.to_csv(f"{PROCESSED_DATA_DIR}/clean_features_dataset.csv", index=False)

    return df


def create_team_mappings():
    """
    Create mappings for team IDs and team metrics to facilitate feature engineering.
    """

    df_team_metrics = create_team_metrics_dataset()
    create_dict_team_id_name(df_team_metrics)

    df_team_metrics_before_match = create_team_features_before_match(df_team_metrics)
    create_dict_matchid_teamid_metrics(df_team_metrics_before_match)


def create_clean_features_dataset():
    """
    Create and save the clean features dataset for machine learning.
    """

    df_features = create_features_dataset()
    prepare_data_for_ml(features_data=df_features, skip_ids=True, save_features=True)


def load_new_data(player_id):
    """
    Prepare and load features for the next match of a given player.
    """

    # load all necessary data
    df_matches_goals = pd.read_csv(f"{PROCESSED_DATA_DIR}/matches_goals_top50_last3seasons.csv")
    df_delay_score = pd.read_csv(f"{PROCESSED_DATA_DIR}/top50_players_delay_score.csv")
    df_features = create_features_dataset()
    team_metrics = load_dict_json("all_team_metrics_before_match")

    # Player data
    player_matches = df_matches_goals[df_matches_goals["player_id"] == player_id].sort_values("date").copy()
    last_player_row = player_matches.iloc[-1]
    player_team = last_player_row["player_team"]

    # Next match
    team_matches = get_team_matches(player_team, LAST_3_SEASONS[0])
    
    try:
        next_match = team_matches[team_matches["isResult"] == False].sort_values("datetime").iloc[0]
    except IndexError:
        logger.warning("No upcoming match found for player %d", player_id)
        return None
    
    is_home = next_match["side"] == "h"
    player_team_id = (next_match["h"]["id"] if is_home else next_match["a"]["id"])
    opponent = next_match["a"] if is_home else next_match["h"]
    opponent_name = opponent["title"]
    opponent_id = opponent["id"]

    # Last completed matches

    def get_last_match(team_name, before_date):
        matches = get_team_matches(team_name, LAST_3_SEASONS[0])
        match_id = (matches[matches["datetime"] < before_date].iloc[-1]["id"])
        return matches[matches["id"] == match_id].iloc[0]

    last_team_match = get_last_match(player_team, next_match["datetime"])

    last_opponent_match = get_last_match(opponent_name, next_match["datetime"])

    # Metrics helpers

    def get_metrics(match_id, team_id):
        key = f"{match_id}_{team_id}"
        return team_metrics.get(key, {})

    team_metrics_before = get_metrics(last_team_match["id"], player_team_id)

    opponent_metrics_before = get_metrics(last_opponent_match["id"], opponent_id)

    # Utility helpers

    def points_from_result(result):
        return 3 if result == "w" else 1 if result == "d" else 0

    def side_key(side):
        return "h" if side == "h" else "a"

    def opponent_side_key(side):
        return "a" if side == "h" else "h"

    # Last 3 matches stats
    last_3 = player_matches.groupby("match_id").last().sort_values("date").tail(3)

    # Base row
    new_data = df_features[df_features["player_id"] == player_id].sort_values("date").iloc[-1:].copy()
    
    # Update features
    new_data["match_id"] = next_match["id"]
    new_data["date"] = next_match["datetime"]
    new_data["h_a"] = next_match["side"]
    new_data["minutes_since_last_goal_before_match"] = player_matches["minutes_since_last_goal"].iloc[-1]
    new_data["delay_score"] = df_delay_score.loc[df_delay_score["player_id"] == player_id, "delay_score"].iloc[0]
    new_data["time_last_3_matches"] = float(last_3["time"].sum())
    new_data["goals_last_3_matches"] = float(last_3["goals"].sum())
    new_data["xG_last_3_matches"] = last_3["xG"].sum()

    side = side_key(last_team_match["side"])
    new_data["total_team_matches_before_match"] = team_metrics_before.get("total_matches_before_match", 0) + 1
    new_data["team_pts_before_match"] = team_metrics_before.get("pts_before_match", 0) + points_from_result(last_team_match["result"])
    new_data["total_team_goals_scored_before_match"] = team_metrics_before.get("total_goals_scored_before_match", 0) + int(last_team_match["goals"][side])
    new_data["total_team_xg_before_match"] = team_metrics_before.get("total_xg_before_match", 0) + float(last_team_match["xG"][side])

    opp_side = opponent_side_key(last_opponent_match["side"])
    new_data["total_opponent_matches_before_match"] = opponent_metrics_before.get("total_matches_before_match", 0) + 1
    new_data["opponent_pts_before_match"] = opponent_metrics_before.get("pts_before_match", 0) + points_from_result(last_opponent_match["result"])
    new_data["total_opponent_goals_conceded_before_match"] = opponent_metrics_before.get("total_goals_conceded_before_match", 0) + int(last_opponent_match["goals"][opp_side])
    new_data["total_opponent_xga_before_match"] = opponent_metrics_before.get("total_xga_before_match", 0) + float(last_opponent_match["xG"][opp_side])
    

    conceded_clean_sheet = int(last_opponent_match["goals"][opp_side]) == 0
    prev_matches = opponent_metrics_before.get("total_matches_before_match", 0)
    prev_rate = opponent_metrics_before.get("clean_sheet_rate_before_match", 0)
    new_data["opponent_clean_sheet_rate_before_match"] = (prev_rate * prev_matches + conceded_clean_sheet) / new_data["total_opponent_matches_before_match"]

    # Final cleanup
    new_data = new_data.drop(columns=["scored"])

    return new_data


def prepare_new_data_for_prediction(new_data, skip_ids=True, save_features=False):
    """
    Prepare the new data for prediction by applying the same feature engineering steps as for the training data.
    """

    # same feature engineering as for training data
    new_data['starting'] = np.where(new_data['position'] == 'Sub', 0, 1)
    new_data['h_a'] = np.where(new_data['h_a']=='h', 1, 0)
    new_data[['league_Bundesliga', 'league_EPL', 'league_La_Liga', 'league_Ligue_1', 'league_Serie_A']] = 0.0
    new_data[f"league_{new_data['league'].values[0]}"] = 1.0
    new_data = new_data.drop(columns=['position','league'])

    # ratios
    new_data['ratio_min_goals_last_3_matches'] = np.where(
        new_data['goals_last_3_matches'] > 0,
        new_data['time_last_3_matches'] / new_data['goals_last_3_matches'],
        300
    )

    new_data['ratio_min_xg_last_3_matches'] = np.where(
        new_data['xG_last_3_matches'] > 0,
        new_data['time_last_3_matches'] / new_data['xG_last_3_matches'],
        300
    )

    new_data['ratio_pts_match'] = np.where(
        new_data['total_team_matches_before_match'] > 0,
        new_data['team_pts_before_match'] / new_data['total_team_matches_before_match'],
        0
    )

    new_data['ratio_pts_match_opponent'] = np.where(
        new_data['total_opponent_matches_before_match'] > 0,
        new_data['opponent_pts_before_match'] / new_data['total_opponent_matches_before_match'],
        0
    )

    new_data['ratio_goals_scored'] = np.where(
        new_data['total_team_matches_before_match'] > 0,
        new_data['total_team_goals_scored_before_match'] / new_data['total_team_matches_before_match'],
        0
    )

    new_data['ratio_xG'] = np.where(
        new_data['total_team_matches_before_match'] > 0,
        new_data['total_team_xg_before_match'] / new_data['total_team_matches_before_match'],
        0
    )

    new_data['ratio_goals_conceded_opponent'] = np.where(
        new_data['total_opponent_matches_before_match'] > 0,
        new_data['total_opponent_goals_conceded_before_match'] / new_data['total_opponent_matches_before_match'],
        0
    )

    new_data['ratio_xga_opponent'] = np.where(
        new_data['total_opponent_matches_before_match'] > 0,
        new_data['total_opponent_xga_before_match'] / new_data['total_opponent_matches_before_match'],
        0
    )

    if save_features:
        new_data.to_csv(f"{PROCESSED_DATA_DIR}/new_data.csv", index=False)

    if skip_ids:
        new_data = new_data.drop(columns=['match_id', 'date', 'player_id'])

    return new_data


def load_and_prepare_new_data(player_id):
    """
    Load and prepare the new data for prediction for a given player ID.
    """

    new_data = load_new_data(player_id)
    prepared_data = prepare_new_data_for_prediction(new_data, skip_ids=True, save_features=True)

    return prepared_data
