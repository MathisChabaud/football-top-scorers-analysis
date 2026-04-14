import requests
import pandas as pd
from src.scraping.config import BASE_URL, HEADERS
from src.scraping.endpoints import SCORERS_ENDPOINT


def get_scorers(competition, season):

    url = BASE_URL + SCORERS_ENDPOINT.format(competition=competition)
    params = {
        "limit": 50,
        "season": season
    }
    response = requests.get(url, headers=HEADERS, params=params)
    #print(response.status_code)
    data = response.json()

    return data["scorers"]


def scorers_to_dataframe(scorers):

    players = []

    for player in scorers:

        players.append({
            "player": player["player"]["name"],
            "team": player["team"]["name"],
            "goals": player["goals"],
            "assists": player["assists"],
            "matches": player["playedMatches"]
        })

    return pd.DataFrame(players)