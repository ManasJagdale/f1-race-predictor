"""
data/build_features.py

Loads the cached Jolpica race results and runs the driver ratings pipeline
on real F1 data. Saves the output to data/processed/driver_ratings.parquet.

Usage (from project root):
    python -m data.build_features
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_jolpica import load_race_results
from feature_engineering.driver_ratings import build_driver_ratings
from config import DRIVER_RATINGS_CACHE


def main():
    print("\nLoading cached race results...")
    race_df = load_race_results()
    print(f"  {len(race_df)} rows, {race_df['season'].nunique()} seasons, "
          f"{race_df['driverId'].nunique()} unique drivers\n")

    print("Building driver ratings...")
    ratings = build_driver_ratings(race_df)

    print(f"\nSaving to {DRIVER_RATINGS_CACHE}...")
    ratings.to_parquet(DRIVER_RATINGS_CACHE, index=False)

    # Print the final Elo standings after the last race in the dataset
    last_season = ratings["season"].max()
    last_round  = ratings[ratings["season"] == last_season]["round"].max()
    final = (
        ratings[
            (ratings["season"] == last_season) &
            (ratings["round"] == last_round)
        ]
        .sort_values("elo_pre_race", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n--- Elo standings going into {last_season} Round {last_round} ---")
    print(f"{'Pos':<5} {'Driver':<25} {'Elo':>8} {'Form':>8} {'Teammate Δ':>12}")
    print("-" * 62)
    for i, row in final.iterrows():
        form = f"{row['recent_form']:.2f}" if pd.notna(row['recent_form']) else "  —"
        delta = f"{row['teammate_delta']:+.2f}" if pd.notna(row['teammate_delta']) else "  —"
        print(f"{i+1:<5} {row['driverId']:<25} {row['elo_pre_race']:>8.1f} "
              f"{form:>8} {delta:>12}")

    print(f"\nDone. {len(ratings)} total rows saved.\n")


if __name__ == "__main__":
    main()
