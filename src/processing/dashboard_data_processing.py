import pandas as pd
from src.core.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LEAGUE_MAPPING


def build_clean_players_dataset():
    """
    Cleans the raw player dataset for use in the dashboard.
    """

    # Load raw data
    df = pd.read_csv(f"{RAW_DATA_DIR}/all_players_data.csv")

    # Map league codes to full names
    df["league"] = df["league"].map(LEAGUE_MAPPING)
    # Delete unnecessary columns
    df = df.drop(["xG", "xA", "shots", "key_passes", "yellow_cards", "red_cards",
                   "position", "npxG", "xGChain", "xGBuildup"], axis=1)
    # Sort dataset
    df = df.sort_values(
        ["season", "goals"],
        ascending=[False, False]
    )
    # Rename columns
    df = df.rename(columns={
        "player_name": "player",
        "team_title": "team"
    })

    # Save processed dataset
    df.to_csv(f"{PROCESSED_DATA_DIR}/all_players_data_clean.csv", index=False)


def color_delay(val):
    """
    Colors the delay column in the dashboard based on the value.
    """

    if val <= -1:
        return "background-color: green"
    elif val <= 0:
        return "background-color: yellow"
    elif val <= 1:
        return "background-color: orange"
    else:
        return "background-color: red"
