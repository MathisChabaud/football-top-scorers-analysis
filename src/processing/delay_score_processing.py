import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.core.config import PROCESSED_DATA_DIR


def create_delay_score_dataset():
    """
    Create a dataset with reliability and delay score for the top 50 players based on their goal scoring patterns.
    """

    df_matches_goals = pd.read_csv(f"{PROCESSED_DATA_DIR}/matches_goals_top50_last3seasons.csv")

    df_valid = df_matches_goals[df_matches_goals["time_between_goals"].notna() & (df_matches_goals["time_between_goals"] != 0)]

    # Computation of delay statistics for each player
    data_delay = df_valid.groupby(["player_id", "player"]).agg(
        avg_delay=("time_between_goals", "mean"),
        median_delay=("time_between_goals", "median"),
        std_delay=("time_between_goals", "std"),
        min_delay=("time_between_goals", "min"),
        max_delay=("time_between_goals", "max")
    ).reset_index()

    # Frequency of big droughts (time between goals > 2 * average delay)
    avg_by_player = data_delay.set_index(["player_id", "player"])["avg_delay"]
    data_delay["big_drought_freq"] = df_valid.groupby(["player_id", "player"])[["time_between_goals"]].apply(
        lambda g: (g["time_between_goals"] > 2 * avg_by_player[g.name]).mean()
    ).values

    # Actual delay since the last goal for each player
    data_delay["actual_delay"] = df_matches_goals.groupby("player_id")["minutes_since_last_goal"].last().values.astype(int)
    data_delay["delay_score"] = (data_delay["actual_delay"]-data_delay["avg_delay"])/data_delay["std_delay"]

    # Definition of the reliability score based on the delay statistics

    features = ["avg_delay", "std_delay", "big_drought_freq"]

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data_delay[features]), columns=features)

    # Inversion of the features so that higher values correspond to better reliability
    for col in features:
        df_scaled[col] = 1 - df_scaled[col]

    # Weighted sum of the features to compute the reliability score
    data_delay["reliability_score"] = (
        0.50 * df_scaled["avg_delay"]
        + 0.30 * df_scaled["std_delay"]
        + 0.20 * df_scaled["big_drought_freq"]
    )

    # Sorting the dataset by reliability score in descending order
    data_delay = data_delay.sort_values("reliability_score", ascending=False)

    data_delay.to_csv(f"{PROCESSED_DATA_DIR}/top50_players_delay_score.csv", index=False)

    return data_delay
