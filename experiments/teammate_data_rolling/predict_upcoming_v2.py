"""
experiments/teammate_delta_rolling/predict_upcoming_v2.py

Same live-prediction flow as the real project's predict_upcoming.py,
but using driver_ratings_v2.py's rolling teammate_delta and the v2
trained ensemble (race_model_v2.joblib) instead of the originals.

Reuses resolve_circuit(), get_lineup(), _align_dtypes(), and
_build_placeholder_rows() DIRECTLY from the real predict_upcoming.py
unchanged -- none of those touch driver_ratings. Only
compute_rolling_features() and build_prediction_row() are redefined
locally here, with the driver_ratings import swapped to driver_ratings_v2.

WRITES ONLY WITHIN THIS FOLDER:
    experiments/teammate_delta_rolling/output_v2/*.csv
    experiments/teammate_delta_rolling/output_v2/*.html

Usage (same CLI shape as the real predict_upcoming.py):
    python predict_upcoming_v2.py --season 2026 --round 9 --grid-csv ../../silverstone_2026_grid.csv --runs 10000
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _THIS_DIR)

from data.fetch_jolpica import load_race_results, load_qualifying
from feature_engineering.car_performance import build_car_performance
from feature_engineering.track_features import load_track_features
from feature_engineering.track_car_interaction import add_track_car_interaction
from data.build_master_features import impute, CIRCUIT_PROFILES_PATH
from models.race_model import FEATURE_COLS
from simulation.simulation import run_monte_carlo, summarize, N_RUNS_DEFAULT
from oracle import print_terminal_table, build_heatmap, CIRCUIT_TO_RACE_NAME

# Reused UNCHANGED from the real project -- these don't touch driver_ratings
from predict_upcoming import (
    resolve_circuit,
    get_lineup,
    _align_dtypes,
    _build_placeholder_rows,
    TRACK_FEATURE_COLS,
)

# The ONE swapped import
from driver_ratings_v2 import build_driver_ratings

MODEL_V2_PATH = os.path.join(_THIS_DIR, "race_model_v2.joblib")
OUTPUT_V2_DIR = os.path.join(_THIS_DIR, "output_v2")


def _load_model_v2() -> list:
    if not os.path.exists(MODEL_V2_PATH):
        raise FileNotFoundError(
            f"No trained v2 model at {MODEL_V2_PATH} -- run `python train_v2.py` first"
        )
    ensemble = joblib.load(MODEL_V2_PATH)
    if not isinstance(ensemble, list):
        ensemble = [ensemble]
    return ensemble


def compute_rolling_features(season: int, round_num: int, circuit_id: str,
                              lineup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identical to the real predict_upcoming.py's version, except
    build_driver_ratings() comes from driver_ratings_v2 (rolling
    teammate_delta) instead of the original.
    """
    real_race_df = load_race_results()
    real_quali_df = load_qualifying()

    real_race_df = real_race_df[
        ~((real_race_df["season"] == season) & (real_race_df["round"] == round_num))
    ]
    real_quali_df = real_quali_df[
        ~((real_quali_df["season"] == season) & (real_quali_df["round"] == round_num))
    ]

    placeholder_race_df, placeholder_quali_df = _build_placeholder_rows(
        season, round_num, circuit_id, lineup
    )
    placeholder_race_df = _align_dtypes(placeholder_race_df, real_race_df)
    placeholder_quali_df = _align_dtypes(placeholder_quali_df, real_quali_df)

    combined_race_df = pd.concat([real_race_df, placeholder_race_df], ignore_index=True)
    combined_quali_df = pd.concat([real_quali_df, placeholder_quali_df], ignore_index=True)

    print("  [v2] Computing driver ratings (rolling teammate_delta)...")
    driver_df = build_driver_ratings(combined_race_df)
    driver_target = driver_df[
        (driver_df["season"] == season) & (driver_df["round"] == round_num)
    ][["driverId", "elo_pre_race", "recent_form", "teammate_delta"]]

    print("  Computing car performance (pace delta, DNF rate) -- unchanged...")
    car_df = build_car_performance(combined_quali_df, combined_race_df)
    car_target = car_df[
        (car_df["season"] == season) & (car_df["round"] == round_num)
    ][["constructorId", "pace_delta_vs_pole", "dnf_rate"]]

    return driver_target, car_target


def build_prediction_row(season: int, round_num: int, circuit_id: str,
                          lineup: pd.DataFrame, wet_override: bool) -> pd.DataFrame:
    driver_target, car_target = compute_rolling_features(season, round_num, circuit_id, lineup)

    race_df = lineup.copy()
    race_df["season"] = season
    race_df["round"] = round_num
    race_df["circuitId"] = circuit_id
    race_df["raceId"] = circuit_id

    race_df = race_df.merge(driver_target, on="driverId", how="left")
    race_df = race_df.merge(car_target, on="constructorId", how="left")

    print("  Adding track × car interaction features (unchanged)...")
    race_df = add_track_car_interaction(race_df, CIRCUIT_PROFILES_PATH)

    print("  Loading track features (unchanged)...")
    track_df = load_track_features()
    this_circuit_track = (
        track_df[track_df["circuitId"] == circuit_id].sort_values("season", ascending=False)
    )
    if not this_circuit_track.empty:
        proxy_row = this_circuit_track.iloc[0]
        for col in TRACK_FEATURE_COLS:
            race_df[col] = proxy_row[col]
        print(f"    Using {int(proxy_row['season'])} telemetry for circuit "
              f"'{circuit_id}' as a proxy")
    else:
        for col in TRACK_FEATURE_COLS:
            race_df[col] = np.nan

    if wet_override:
        race_df["wet_flag"] = 1

    print("  Imputing any remaining missing values (unchanged logic)...")
    race_df = impute(race_df)
    race_df["grid_position"] = race_df["grid_position"].astype(float)

    missing = [c for c in FEATURE_COLS if c not in race_df.columns or race_df[c].isna().any()]
    if missing:
        raise RuntimeError(f"Feature row incomplete -- missing/NaN in: {missing}.")

    return race_df


def export_csv_v2(prob_matrix: pd.DataFrame, season: int, round_num: int, circuit_id: str) -> str:
    os.makedirs(OUTPUT_V2_DIR, exist_ok=True)
    filename = f"v2_{season}_r{round_num:02d}_{circuit_id}_probabilities.csv"
    path = os.path.join(OUTPUT_V2_DIR, filename)
    prob_matrix.reset_index().to_csv(path, index=False)
    return path


def export_heatmap_v2(fig: go.Figure, season: int, round_num: int, circuit_id: str) -> str:
    os.makedirs(OUTPUT_V2_DIR, exist_ok=True)
    filename = f"v2_{season}_r{round_num:02d}_{circuit_id}_heatmap.html"
    path = os.path.join(OUTPUT_V2_DIR, filename)
    fig.write_html(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict an upcoming F1 race -- v2 (rolling teammate_delta)")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_num")
    parser.add_argument("--grid-csv", type=str, default=None)
    parser.add_argument("--wet", action="store_true")
    parser.add_argument("--runs", type=int, default=N_RUNS_DEFAULT)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    print(f"Resolving round {args.round_num} of season {args.season}...")
    circuit_id, race_name = resolve_circuit(args.season, args.round_num)
    print(f"  {race_name} ({circuit_id})")

    print("\nDetermining grid / lineup (unchanged)...")
    lineup, source = get_lineup(args.season, args.round_num, args.grid_csv)
    print(f"  ✓ {len(lineup)} drivers, source: {source}")

    print("\n[v2] Building prediction features (rolling teammate_delta)...")
    race_df = build_prediction_row(args.season, args.round_num, circuit_id, lineup, args.wet)

    print("\nLoading v2 trained model ensemble...")
    ensemble = _load_model_v2()

    print(f"Running simulation ({args.runs:,} Monte Carlo runs)...")
    prob_matrix, dnf_prob = run_monte_carlo(race_df, ensemble, n_runs=args.runs)

    summary = summarize(prob_matrix, dnf_prob, race_df)
    print_terminal_table(summary, args.season, args.round_num, circuit_id)

    csv_path = export_csv_v2(prob_matrix, args.season, args.round_num, circuit_id)
    print(f"✓ [v2] CSV saved → {csv_path}")

    fig = build_heatmap(prob_matrix, args.season, args.round_num, circuit_id)
    heatmap_path = export_heatmap_v2(fig, args.season, args.round_num, circuit_id)
    print(f"✓ [v2] Heatmap saved → {heatmap_path}")

    if args.show:
        fig.show()

    print("\nCompare this table's P(Win)/P(Top 3) for antonelli against the real")
    print("project's original Silverstone run to see whether the rolling")
    print("teammate_delta change moved his probability, and in which direction.")
