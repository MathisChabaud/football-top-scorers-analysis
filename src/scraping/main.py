import pandas as pd
import time
from src.scraping.football_data_api import get_scorers, scorers_to_dataframe

def main():

    seasons = [2025, 2024, 2023]
    leagues = ["PL", "BL1", "SA", "PD", "FL1"]
    all_data = []

    for season in seasons:
        for league in leagues:
            scorers = get_scorers(league, season)
            df = scorers_to_dataframe(scorers)
            df["season"] = season
            df["league"] = league
            all_data.append(df)

            time.sleep(7)  # To respect API rate limits

    final_df = pd.concat(all_data)

    final_df.to_csv("data/raw/top_scorers.csv", index=False)


if __name__ == "__main__":
    main()