import sys
from pathlib import Path

# Add the root directory to the Python path to allow imports from src
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

import streamlit as st
import numpy as np
import pandas as pd
from src.processing.dashboard_data_processing import color_delay
from src.core.config import REL_THR

# -----------------------------
# Config page
# -----------------------------

st.set_page_config(
    page_title="Top European Scorers",
    layout="wide"
)

# -----------------------------
# Load data
# -----------------------------

@st.cache_data
def load_data_1():
    df = pd.read_csv("data/processed/all_players_data_clean.csv")
    return df

@st.cache_data
def load_data_2():
    df = pd.read_csv("data/processed/top50_players_delay_score.csv")
    return df

df_players = load_data_1()
df_delay = load_data_2()

# -----------------------------
# Navigation
# -----------------------------

page = st.sidebar.radio("Browsing", ["⚽ Top Scorers","⏳ Delay Score"])

# -----------------------------
# Page: Top Scorers
# -----------------------------

if page == "⚽ Top Scorers":

    #-----------------------------
    # Filters
    #-----------------------------

    st.sidebar.header("Filters")

    # Season
    seasons = sorted(df_players["season"].unique(), reverse=True)
    selected_season = st.sidebar.selectbox(
        "Season",
        seasons,
    )

    # League
    leagues = sorted(df_players["league"].unique())
    leagues.insert(0, "Europe")
    selected_league = st.sidebar.selectbox(
        "League",
        leagues
    )

    # Number of seasons (cumulative)
    max_seasons = selected_season - 2020
    n_seasons = st.sidebar.selectbox(
        "Number of seasons (cumulative)",
        options=range(1, max_seasons + 1),
        format_func=lambda x: f"Cumulative {x} season{'s' if x > 1 else ''} ({selected_season if x == 1 else f'{selected_season - x + 1} - {selected_season}'})",
        index=0
    )

    # Number of players
    top_n = st.sidebar.slider(
        "Top players",
        min_value=10,
        max_value=50,
        value=20
    )

    st.title("⚽ Top Scorers Ranking")

    # -----------------------------
    # Apply filters
    # -----------------------------

    # Selection cumulative seasons
    season_index = seasons.index(selected_season)
    selected_seasons = seasons[season_index : season_index + n_seasons]
    filtered = df_players[df_players["season"].isin(selected_seasons)]

    # Selection league
    if selected_league != "Europe":
        filtered = filtered[filtered["league"] == selected_league]
    
    # Multiple seasons aggregation

    filtered = filtered.sort_values("season")
    grouped = filtered.groupby(["id", "player"]).agg(
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        matches=("games", "sum"),
        minutes=("time", "sum"),
        team=("team", "last") # last club
    ).reset_index()

    # When a player has played for multiple clubs in the same season, we keep the last one
    grouped["team"] = grouped["team"].apply(lambda x: x.split(",")[1].strip() if "," in x else x)

    # -----------------------------
    # Ranking
    # -----------------------------

    grouped = grouped.sort_values(["goals", "assists"],ascending=[False, False]).head(top_n)
    grouped["rank"] = range(1, len(grouped) + 1)
    grouped["min/goal"] = np.ceil(grouped["minutes"] / grouped["goals"])

    # -----------------------------
    # Table
    # -----------------------------

    st.dataframe(
        grouped[
            ["rank", "player", "team", "goals", "assists", "matches", "min/goal"]
        ],
        hide_index=True,
        width="stretch"
    )

# -----------------------------
# Page: Delay Score
# -----------------------------

elif page == "⏳ Delay Score":

    #-----------------------------
    # Filters
    #-----------------------------

    st.sidebar.header("Filters")

    min_rel_score = st.sidebar.slider(
        "Minimum reliability score",
        min_value=0.1,
        max_value=1.0,
        step=0.1,
        value=REL_THR
    )

    top_n_delay = st.sidebar.slider(
        "Top delayed players",
        min_value=10,
        max_value=50,
        value=20
    )

    st.title("⏳ Scoring Delay Analysis")

    # -----------------------------
    # Apply filters
    # -----------------------------

    filtered_rel_score = df_delay[df_delay["reliability_score"] >= min_rel_score]
    filtered_delay = filtered_rel_score.sort_values("delay_score", ascending=False).head(top_n_delay)
    filtered_delay["avg_delay"] = filtered_delay["avg_delay"].round(0).astype(int)
    filtered_delay["actual_delay"] = filtered_delay["actual_delay"].round(0).astype(int)

    # -----------------------------
    # Table
    # -----------------------------

    st.subheader("Most delayed scorers among the top 50 in Europe over the last 3 seasons")

    # Rename columns for better display
    filtered_delay = filtered_delay.rename(columns={
        "avg_delay": "average delay between goals (minutes)",
        "actual_delay": "actual delay (minutes)"
    })

    st.dataframe(
        filtered_delay[["player", "average delay between goals (minutes)", "actual delay (minutes)", "reliability_score", "delay_score"]]
        .style
        .map(
            color_delay,
            subset=["delay_score"]
        ),
        width="stretch",
        hide_index=True
    )
