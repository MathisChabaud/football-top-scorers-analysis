import pandas as pd
import numpy as np
from src.ml.models import train_and_predict
from src.ml.split import train_test_split_by_date
from src.processing.ml_features_processing import create_features_dataset, prepare_data_for_ml
from src.core.config import PROCESSED_DATA_DIR, ODDS, RAW_DATA_DIR, STAKE, REL_THR


def select_reliable_player_lines(df_matches):
    """
    Select only the lines of players with a reliability score above the threshold.
    """

    df_score = pd.read_csv(f"{PROCESSED_DATA_DIR}/top50_players_delay_score.csv")

    df_score = df_score[df_score["reliability_score"] >= REL_THR].copy()
    ids_top_n = df_score["player_id"].tolist()
    df_matches = df_matches[df_matches["player_id"].isin(ids_top_n)].copy()

    return df_matches


def global_martingale(df_matches):
    """
    Global martingale strategy on starting players. Double the stake after each loss.
    """

    df_matches = df_matches.sort_values(["date", "player_id"]).reset_index(drop=True)

    # Initialize columns
    df_matches["stake"] = 0
    df_matches["gain"] = 0
    df_matches["consecutive_losses"] = 0
    
    # Global variable for martingale
    consecutive_losses = 0
    
    for idx in df_matches.index:
        # Check if we bet (starting player)
        if df_matches.loc[idx, "position"] != "Sub":
            stake = STAKE * (2 ** consecutive_losses)
            df_matches.loc[idx, "stake"] = stake

            # Check the result
            if df_matches.loc[idx, "goals"] >= 1:
                # Gain
                df_matches.loc[idx, "gain"] = ODDS * stake - stake
                df_matches.loc[idx, "consecutive_losses"] = consecutive_losses
                consecutive_losses = 0
            else:
                # Loss
                df_matches.loc[idx, "gain"] = -stake
                df_matches.loc[idx, "consecutive_losses"] = consecutive_losses
                consecutive_losses += 1
        else:
            # No bet
            df_matches.loc[idx, "stake"] = 0
            df_matches.loc[idx, "gain"] = 0
            df_matches.loc[idx, "consecutive_losses"] = consecutive_losses
    
    return df_matches


def by_player_martingale(df_matches):
    """
    By-player martingale strategy on starting players. 
    Double the stake after each loss for each player.
    """

    df_matches = df_matches.sort_values(["date", "player_id"]).reset_index(drop=True)

    # Initialize columns
    df_matches["consecutive_losses"] = 0
    df_matches["stake"] = 0
    df_matches["gain"] = 0
    
    # For each player, calculate their individual martingale strategy
    for player_id in df_matches["player_id"].unique():
        # Filter for a player
        mask = df_matches["player_id"] == player_id
        player_indices = df_matches[mask].index
        
        consecutive_losses = 0
        
        for idx in player_indices:
            # Check if we bet (starting player)
            if df_matches.loc[idx, "position"] != "Sub":
                stake = STAKE * (2 ** consecutive_losses)
                df_matches.loc[idx, "stake"] = stake
                # Check the result
                if df_matches.loc[idx, "goals"] >= 1:
                    # Gain
                    df_matches.loc[idx, "gain"] = ODDS * stake - stake
                    df_matches.loc[idx, "consecutive_losses"] = consecutive_losses
                    consecutive_losses = 0 
                else:
                    # Loss
                    df_matches.loc[idx, "gain"] = -stake
                    df_matches.loc[idx, "consecutive_losses"] = consecutive_losses
                    consecutive_losses += 1
            else:
                # No bet
                df_matches.loc[idx, "stake"] = 0
                df_matches.loc[idx, "gain"] = 0
                df_matches.loc[idx, "consecutive_losses"] = consecutive_losses

    return df_matches


def by_player_martingale_with_delay_condition(df_matches_goals, bet_after_avg_delay=True, delay_prop=1.0):
    """
    By-player martingale strategy with delay condition :
    - If bet_after_avg_delay is True, we bet on players who have exceeded their average minutes per goal by a factor of delay_prop.
    - If bet_after_avg_delay is False, we bet on players who have not exceeded their average minutes per goal by a factor of delay_prop.
    """

    df_matches_goals.sort_values(["date", "player_id"], inplace=True)

    # Initialize columns
    df_matches_goals["consecutive_losses"] = 0
    df_matches_goals["stake"] = 0
    df_matches_goals["gain"] = 0
    
    # For each player, calculate their individual martingale strategy
    for player_id in df_matches_goals["player_id"].unique():
        # Filter for a player
        mask = df_matches_goals["player_id"] == player_id
        player_indices = df_matches_goals[mask].index
        
        consecutive_losses = 0
        
        for idx in player_indices:
            # Verify if we bet (starting player + exceeded or not average minutes per goal)
            if (df_matches_goals.loc[idx, "position"] != "Sub") & ((df_matches_goals.loc[idx, "min/goal"] * delay_prop <= df_matches_goals.loc[idx, "minutes_since_last_goal_before_match"]) if bet_after_avg_delay else (df_matches_goals.loc[idx, "min/goal"] * delay_prop >= df_matches_goals.loc[idx, "minutes_since_last_goal_before_match"])):

                stake = STAKE * (2 ** consecutive_losses)
                df_matches_goals.loc[idx, "stake"] = stake
                # Verify the result
                if df_matches_goals.loc[idx, "goals"] >= 1:
                    # Gain
                    df_matches_goals.loc[idx, "gain"] = ODDS * stake - stake
                    df_matches_goals.loc[idx, "consecutive_losses"] = consecutive_losses
                    consecutive_losses = 0
                else:
                    # Loss
                    df_matches_goals.loc[idx, "gain"] = -stake
                    df_matches_goals.loc[idx, "consecutive_losses"] = consecutive_losses
                    consecutive_losses += 1
            else:
                # No bet
                df_matches_goals.loc[idx, "stake"] = 0
                df_matches_goals.loc[idx, "gain"] = 0
                df_matches_goals.loc[idx, "consecutive_losses"] = consecutive_losses

    # If a player scores multiple goals in a match, we win only one bet.
    df_matches_goals["stake"] = df_matches_goals.groupby(["player_id", "match_id"])["stake"].transform("first")
    df_matches_goals["gain"] = np.where(
            df_matches_goals["goals"]>=2,
            (ODDS * df_matches_goals["stake"] - df_matches_goals["stake"])/df_matches_goals["goals"],
            df_matches_goals["gain"]
        )

    return df_matches_goals


def global_martingale_with_predictions(df_matches):
    """
    Global martingale strategy on starting players with model predictions.
    We bet only if the model predicts a goal and the player is a starting player.
    """

    # Initialize columns
    df_matches["stake"] = 0
    df_matches["gain"] = 0
    df_matches["consecutive_losses"] = 0
        
    # Global variable for martingale
    consecutive_losses = 0
        
    for idx in df_matches.index:
    
        # Check if we bet (starting player + model prediction)
        if (df_matches.loc[idx, "starting"] == 1) and (df_matches.loc[idx, "y_custom"] == 1):
            stake = STAKE * (2 ** consecutive_losses)
            df_matches.loc[idx, "stake"] = stake

            # Check the result
            if df_matches.loc[idx, "scored"] == 1:
                # Gain
                df_matches.loc[idx, "gain"] = ODDS * stake - stake
                df_matches.loc[idx, "consecutive_losses"] = consecutive_losses
                consecutive_losses = 0
            else:
                # Loss
                df_matches.loc[idx, "gain"] = -stake
                df_matches.loc[idx, "consecutive_losses"] = consecutive_losses
                consecutive_losses += 1
        else:
            # No bet
            df_matches.loc[idx, "stake"] = 0
            df_matches.loc[idx, "gain"] = 0
            df_matches.loc[idx, "consecutive_losses"] = consecutive_losses

    return df_matches


def strategy_1(df_matches):
    """
    Strategy 1: Bet on all starting players, with a fixed stake.
    """

    df_matches = select_reliable_player_lines(df_matches)

    df_matches.sort_values(["date", "player_id"], inplace=True)
    df_matches["stake"] = np.where(df_matches["position"]!="Sub", STAKE, 0)
    df_matches["gain"] = np.where(df_matches["position"]!="Sub", ODDS * STAKE * (df_matches["goals"] > 0).astype(int) - STAKE, 0)
    df_matches["cumulative_gain"] = df_matches["gain"].cumsum()

    df_days = df_matches.groupby("date").agg(
        daily_gain=("gain", "sum"),
        daily_cumulative_gain=("cumulative_gain", "last")
    )

    return df_matches, df_days


def strategy_2(df_matches_goals, bet_after_avg_delay=True, delay_prop=1.0):
    """
    Strategy 2: Bet on starting players who 
    - have exceeded their average minutes per goal (if bet_after_avg_delay is True).
    - have not exceeded their average minutes per goal (if bet_after_avg_delay is False).
    Fixed stake.
    """

    df_delay_score = pd.read_csv(f"{PROCESSED_DATA_DIR}/top50_players_delay_score.csv")
    df_matches_goals = select_reliable_player_lines(df_matches_goals)

    df_matches_goals["min/goal"] = df_matches_goals["player_id"].map(
        df_delay_score.set_index("player_id")["avg_delay"]
    )

    df_matches_goals["bet"]= np.where(
        ((df_matches_goals["minutes_since_last_goal_before_match"] >= df_matches_goals["min/goal"] * delay_prop) if bet_after_avg_delay else (df_matches_goals["minutes_since_last_goal_before_match"] <= df_matches_goals["min/goal"] * delay_prop)) 
        & (df_matches_goals["position"]!="Sub"),
        1, 0
    )
    df_matches_goals["stake"] = np.where(df_matches_goals["bet"] == 1, STAKE, 0)
    df_matches_goals["gain"] = np.where(
        df_matches_goals["bet"] == 1,
        (ODDS * STAKE * (df_matches_goals["goals"] > 0).astype(int) - STAKE),
        0
    )
    # If a player scores multiple goals in a match, we win only one bet.
    df_matches_goals["gain"] = np.where(
        df_matches_goals["goals"]>=2,
        df_matches_goals["gain"]/df_matches_goals["goals"],
        df_matches_goals["gain"]
    )

    df_matches_goals.sort_values(["date", "player_id"], inplace=True)
    df_matches_goals["cumulative_gain"] = df_matches_goals["gain"].cumsum()
    df_days = df_matches_goals.groupby("date").agg(
        daily_gain=("gain", "sum"),
        daily_cumulative_gain=("cumulative_gain", "last")
    )

    return df_matches_goals, df_days


def strategy_3(df_matches):
    """
    Strategy 3: Global martingale strategy.
    """

    df_matches = select_reliable_player_lines(df_matches)

    # Apply global martingale strategy
    df_matches = global_martingale(df_matches)
    
    # Cumulative
    df_matches["cumulative_gain"] = df_matches["gain"].cumsum()
    
    # Aggregate by day
    df_days = df_matches.groupby("date").agg(
        daily_gain=("gain", "sum"),
        daily_cumulative_gain=("cumulative_gain", "last")
    )
    
    return df_matches, df_days


def strategy_4(df_matches):
    """
    Strategy 4: By-player martingale strategy.
    """

    df_matches = select_reliable_player_lines(df_matches)

    # Apply by-player martingale strategy
    df_matches = by_player_martingale(df_matches)

    # Cumulative
    df_matches["cumulative_gain"] = df_matches["gain"].cumsum()
    
    # Aggregate by day
    df_days = df_matches.groupby("date").agg(
        daily_gain=("gain", "sum"),
        daily_cumulative_gain=("cumulative_gain", "last")
    )
    
    return df_matches, df_days


def strategy_5(df_matches_goals, bet_after_avg_delay=True, delay_prop=1.0):
    """
    Strategy 5: By-player martingale strategy with delay condition.
    - If bet_after_avg_delay is True, we bet on players who have exceeded their average minutes per goal by a factor of delay_prop.
    - If bet_after_avg_delay is False, we bet on players who have not exceeded their average minutes per goal by a factor of delay_prop.
    """

    df_delay_score = pd.read_csv(f"{PROCESSED_DATA_DIR}/top50_players_delay_score.csv")
    df_matches_goals = select_reliable_player_lines(df_matches_goals)

    df_matches_goals["min/goal"] = df_matches_goals["player_id"].map(
        df_delay_score.set_index("player_id")["avg_delay"]
    )

    df_matches_goals = by_player_martingale_with_delay_condition(df_matches_goals, bet_after_avg_delay=bet_after_avg_delay, delay_prop=delay_prop)

    # Cumulative
    df_matches_goals["cumulative_gain"] = df_matches_goals["gain"].cumsum()
    
    # Aggregate by day
    df_days = df_matches_goals.groupby("date").agg(
        daily_gain=("gain", "sum"),
        daily_cumulative_gain=("cumulative_gain", "last")
    )
    
    return df_matches_goals, df_days


def strategy_6(df_features_with_ids, model_name, threshold=0.5):
    """
    Strategy 6: Global martingale strategy with model predictions.
    The model predicts a goal if the predicted probability of scoring is above the threshold.

    model_name can be "logistic_regression", "random_forest" or "xgboost".
    """

    # Select only reliable player lines
    df_features_with_ids = df_features_with_ids[df_features_with_ids["reliability_score"] >= REL_THR].reset_index(drop=True)
    
    # Train-test split by date (test set is the last 20% of the matches)
    X_train, X_test, y_train, y_test = train_test_split_by_date(df_features_with_ids, train_ratio=0.8)
    X_train_for_fit =  X_train.drop(columns=['match_id', 'date', 'player_id'])
    X_test_for_predict = X_test.drop(columns=['match_id', 'date', 'player_id'])

    y, y_proba = train_and_predict(X_train_for_fit, y_train, X_test_for_predict, model_name)

    # Create a dataframe with the matches, the true labels, the predicted probabilities and the predicted labels
    df_matches = pd.concat([X_test, y_test], axis=1)
    df_matches['y_proba'] = y_proba
    df_matches['y_custom'] = (df_matches['y_proba'] >= threshold).astype(int)

    # Apply global martingale strategy with predictions
    df_matches = global_martingale_with_predictions(df_matches)

    # Cumulative
    df_matches["cumulative_gain"] = df_matches["gain"].cumsum()

    # Aggregate by day
    df_days = df_matches.groupby("date").agg(
        daily_gain=("gain", "sum"),
        daily_cumulative_gain=("cumulative_gain", "last")
    )
    
    return df_matches, df_days


def compute_kpis(df, df_days):

    # General KPIs
    nb_bets = (df["stake"] > 0).sum()
    prop_winning_bets = (df["gain"] > 0).sum() / nb_bets if nb_bets > 0 else 0
    final_gain = df["cumulative_gain"].iloc[-1]
    max_cumulative_gain = df["cumulative_gain"].max()
    max_cumulative_loss = df["cumulative_gain"].min() if df["cumulative_gain"].min() <= 0 else 1

    # Daily KPIs
    max_gain_day = df_days["daily_gain"].max()
    max_loss_day = df_days["daily_gain"].min()

    # ROI
    roi = final_gain / df["stake"].sum() if df["stake"].sum() > 0 else 0

    # Strat score : cumulative gain * prop winning bets / max loss day
    loss_score = -max_loss_day if max_loss_day < 0 else 1
    strat_score = max_cumulative_gain * prop_winning_bets / loss_score

    kpis = {
        "nb_bets": nb_bets,
        "prop_winning_bets": prop_winning_bets,
        "final_gain": final_gain,
        "max_cumulative_gain": max_cumulative_gain,
        "max_cumulative_loss": max_cumulative_loss,
        "max_gain_day": max_gain_day,
        "max_loss_day": max_loss_day,
        "roi": roi,
        "strat_score": strat_score
    }

    return kpis


def compute_kpis_for_ml_strat(df, df_days):

    # General KPIs : we care about the number of bets, and the consecutive losses (martingale)
    nb_bets = (df["stake"] > 0).sum()
    prop_winning_bets = (df["gain"] > 0).sum() / nb_bets if nb_bets > 0 else 0
    final_gain = df["cumulative_gain"].iloc[-1]
    max_cumulative_gain = df["cumulative_gain"].max()
    max_cumulative_loss = df["cumulative_gain"].min() if df["cumulative_gain"].min() <= 0 else 1
    max_consecutive_losses = df["consecutive_losses"].max()
    nb_three_or_more_consecutive_losses = (df["consecutive_losses"] >= 3).sum()

    # Daily KPIs
    max_gain_day = df_days["daily_gain"].max()
    max_loss_day = df_days["daily_gain"].min()

    # ROI
    roi = final_gain / df["stake"].sum() if df["stake"].sum() > 0 else 0

    # Strat score : max cumulative gain * prop winning bets / max consecutive losses
    strat_score = max_cumulative_gain * prop_winning_bets / (max_consecutive_losses if max_consecutive_losses > 0 else 1)

    kpis = {
        "nb_bets": nb_bets,
        "prop_winning_bets": prop_winning_bets,
        "final_gain": final_gain,
        "max_cumulative_gain": max_cumulative_gain,
        "max_cumulative_loss": max_cumulative_loss,
        "max_consecutive_losses": max_consecutive_losses,
        "nb_three_or_more_consecutive_losses": nb_three_or_more_consecutive_losses,
        "max_gain_day": max_gain_day,
        "max_loss_day": max_loss_day,
        "roi": roi,
        "strat_score": strat_score
    }

    return kpis


def create_results_dataset():
    """
    Create a dataset with the results of the different strategies and parameters choices (no ML).
    """

    df_matches = pd.read_csv(f"{RAW_DATA_DIR}/matches_top50_last3seasons.csv")
    df_matches.rename(columns={"id":"match_id"}, inplace=True)
    df_matches_goals = pd.read_csv(f"{PROCESSED_DATA_DIR}/matches_goals_top50_last3seasons.csv")

    results = []
    df, df_days = strategy_1(df_matches)
    kpis = compute_kpis(df, df_days)
    results.append({"strategy": "strategy_1", **kpis})
    df, df_days = strategy_2(df_matches_goals, bet_after_avg_delay=True, delay_prop=1.0)
    kpis = compute_kpis(df, df_days)
    results.append({"strategy": "strategy_2_1", **kpis})
    df, df_days = strategy_2(df_matches_goals, bet_after_avg_delay=False, delay_prop=1.0)
    kpis = compute_kpis(df, df_days)
    results.append({"strategy": "strategy_2_2", **kpis})
    df, df_days = strategy_3(df_matches)
    kpis = compute_kpis(df, df_days)
    results.append({"strategy": "strategy_3", **kpis})
    df, df_days = strategy_4(df_matches)
    kpis = compute_kpis(df, df_days)
    results.append({"strategy": "strategy_4", **kpis})
    df, df_days = strategy_5(df_matches_goals, bet_after_avg_delay=True, delay_prop=1.0)
    kpis = compute_kpis(df, df_days)
    results.append({"strategy": "strategy_5_1", **kpis})
    df, df_days = strategy_5(df_matches_goals, bet_after_avg_delay=False, delay_prop=1.0)
    kpis = compute_kpis(df, df_days)
    results.append({"strategy": "strategy_5_2", **kpis})
    df, df_days = strategy_5(df_matches_goals, bet_after_avg_delay=False, delay_prop=1.5)
    kpis = compute_kpis(df, df_days)
    results.append({"strategy": "strategy_5_3", **kpis})

    df_results = pd.DataFrame(results)

    df_results.to_csv(f"{PROCESSED_DATA_DIR}/strategy_1_5_results.csv", index=False)


def create_results_dataset_for_ml_strat():
    """
    Create a dataset with the results of the different parameters choices for strategy 6 (ML).
    """

    df_features = create_features_dataset()
    df_features_with_ids = prepare_data_for_ml(features_data=df_features, skip_ids=False, save_features=False)

    results = []
    df, df_days = strategy_6(df_features_with_ids, model_name="logistic_regression", threshold=0)
    kpis = compute_kpis_for_ml_strat(df, df_days)
    results.append({"strategy": "strategy_6_noML", **kpis})
    df, df_days = strategy_6(df_features_with_ids, model_name="logistic_regression", threshold=0.5)
    kpis = compute_kpis_for_ml_strat(df, df_days)
    results.append({"strategy": "strategy_6_lr_0.5", **kpis})
    df, df_days = strategy_6(df_features_with_ids, model_name="logistic_regression", threshold=0.53)
    kpis = compute_kpis_for_ml_strat(df, df_days)
    results.append({"strategy": "strategy_6_lr_0.53", **kpis})
    df, df_days = strategy_6(df_features_with_ids, model_name="logistic_regression", threshold=0.55)
    kpis = compute_kpis_for_ml_strat(df, df_days)
    results.append({"strategy": "strategy_6_lr_0.55", **kpis})
    df, df_days = strategy_6(df_features_with_ids, model_name="logistic_regression", threshold=0.6)
    kpis = compute_kpis_for_ml_strat(df, df_days)
    results.append({"strategy": "strategy_6_lr_0.6", **kpis})
    df, df_days = strategy_6(df_features_with_ids, model_name="logistic_regression", threshold=0.65)
    kpis = compute_kpis_for_ml_strat(df, df_days)
    results.append({"strategy": "strategy_6_lr_0.65", **kpis})
    df, df_days = strategy_6(df_features_with_ids, model_name="xgboost", threshold=0.5)
    kpis = compute_kpis_for_ml_strat(df, df_days)
    results.append({"strategy": "strategy_6_xgb_0.5", **kpis})
    df, df_days = strategy_6(df_features_with_ids, model_name="xgboost", threshold=0.53)
    kpis = compute_kpis_for_ml_strat(df, df_days)
    results.append({"strategy": "strategy_6_xgb_0.53", **kpis})
    df, df_days = strategy_6(df_features_with_ids, model_name="xgboost", threshold=0.55)
    kpis = compute_kpis_for_ml_strat(df, df_days)
    results.append({"strategy": "strategy_6_xgb_0.55", **kpis})
    df, df_days = strategy_6(df_features_with_ids, model_name="xgboost", threshold=0.6)
    kpis = compute_kpis_for_ml_strat(df, df_days)
    results.append({"strategy": "strategy_6_xgb_0.6", **kpis})
    df, df_days = strategy_6(df_features_with_ids, model_name="xgboost", threshold=0.65)
    kpis = compute_kpis_for_ml_strat(df, df_days)
    results.append({"strategy": "strategy_6_xgb_0.65", **kpis})

    df_results = pd.DataFrame(results)
    
    df_results.to_csv(f"{PROCESSED_DATA_DIR}/strategy_6_results.csv", index=False)
