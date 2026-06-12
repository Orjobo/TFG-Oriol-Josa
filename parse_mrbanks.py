"""
TFG - Entropy analysis of the Mr. Banks experiment
Step 1: parse the XML and explore the data
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# --- Parse the XML ---
XML_PATH = "/Users/racelabs/Downloads/MrBanks2013.xml"

tree = ET.parse(XML_PATH)
root = tree.getroot()
db = root.find("database")

def parse_table(db, table_name):
    rows = []
    for table in db.findall("table"):
        if table.get("name") == table_name:
            row = {}
            for col in table.findall("column"):
                val = col.text
                row[col.get("name")] = val
            rows.append(row)
    return pd.DataFrame(rows)

print("Parsing XML...")
df_series = parse_table(db, "Series")
df_users = parse_table(db, "Users")
df_games = parse_table(db, "Games")
df_rounds = parse_table(db, "Rounds")

# --- Cast columns to numeric types ---
# Series
for c in ["id", "series", "round", "result", "expert"]:
    df_series[c] = pd.to_numeric(df_series[c])
df_series["price"] = pd.to_numeric(df_series["price"])
df_series["diff"] = pd.to_numeric(df_series["diff"])

# Users
df_users["id"] = pd.to_numeric(df_users["id"])
df_users["score"] = pd.to_numeric(df_users["score"])
df_users["finished"] = pd.to_numeric(df_users["finished"])

# Games
for c in ["id", "completed", "correct_answers", "errors"]:
    df_games[c] = pd.to_numeric(df_games[c])

# Rounds
for c in ["id", "game", "round", "user", "decision", "result",
          "information_consulted", "clicks"]:
    df_rounds[c] = pd.to_numeric(df_rounds[c])
for c in ["round_time", "info_daily_price_time", "info_5days_average_time",
          "info_30days_average_time", "info_intraday_time", "info_expert_time",
          "info_arrows_time", "info_world_markets_time"]:
    df_rounds[c] = pd.to_numeric(df_rounds[c])

# --- Basic summary ---
print("\n" + "="*60)
print("DATASET SUMMARY")
print("="*60)

print(f"\nSeries:  {len(df_series)} rows")
print(f"Users:   {len(df_users)} rows")
print(f"Games:   {len(df_games)} rows")
print(f"Rounds:  {len(df_rounds)} rows")

print(f"\nUsers who finished:    {df_users['finished'].sum()}")
print(f"Completed games:       {df_games[df_games['completed']==1].shape[0]}")

# Valid decisions only (decision=0 means "no answer", so drop it)
valid = df_rounds[df_rounds["decision"] != 0]
print(f"Valid decisions:       {len(valid)} ({len(df_rounds)-len(valid)} dropped, no answer)")

print("\n--- Decision distribution ---")
print(df_rounds["decision"].value_counts().sort_index())
print(f"\np(up) = {(valid['decision']==1).mean():.3f}")
print(f"p(down) = {(valid['decision']==-1).mean():.3f}")

print("\n--- Results ---")
print(valid["result"].value_counts().sort_index())
print(f"\nSuccess rate = {(valid['result']==1).mean():.3f}")

print("\n--- Rounds per user (statistics) ---")
rounds_per_user = valid.groupby("user").size()
print(rounds_per_user.describe())

print("\n--- Scenarios (via Games) ---")
print(df_games["scenario"].value_counts().sort_index())

# --- Merge rounds with games to attach the scenario ---
df_rounds_full = df_rounds.merge(
    df_games[["id", "scenario", "series"]].rename(columns={"id": "game", "series": "game_series"}),
    on="game",
    how="left"
)

print("\n--- First rows of Rounds (with scenario) ---")
print(df_rounds_full[["user", "game", "round", "scenario", "decision", "result"]].head(20).to_string())

# --- Save CSVs for easy access ---
OUT_DIR = "/Users/racelabs/Desktop/TFG/data"
import os
os.makedirs(OUT_DIR, exist_ok=True)

df_rounds_full.to_csv(f"{OUT_DIR}/rounds.csv", index=False)
df_users.to_csv(f"{OUT_DIR}/users.csv", index=False)
df_games.to_csv(f"{OUT_DIR}/games.csv", index=False)
df_series.to_csv(f"{OUT_DIR}/series.csv", index=False)

print(f"\nCSVs saved to {OUT_DIR}/")
print("Done. Data parsed and exported.")
