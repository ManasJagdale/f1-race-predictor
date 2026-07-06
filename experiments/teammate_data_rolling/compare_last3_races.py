"""
experiments/teammate_delta_rolling/compare_last3_races.py

Runs predict_vs_actual.py-style diagnostics -- predicted vs. actual,
Brier scored -- on the last N already-raced rounds in the dataset,
using BOTH the real project's original model and this experiment's v2
(rolling teammate_delta) model side by side. This is the most direct
test available: not an aggregate statistic across 78 races, but "on
races that actually happened recently, which model's predictions were
closer to what really occurred?"

Reuses evaluate_race() from the real project's backtest.py DIRECTLY,
unmodified -- identical scoring logic for both models, just fed
different trained ensembles and different feature rows (original
master_features.parquet vs. this folder's master_features_v2.parquet).

WRITES NOTHING -- purely a read/print diagnostic. Safe to re-run as
many times as you like.

Usage:
    python compare_last3_races.py              # last 3 races (default)
    python compare_last3_races.py --n 5        # last 5 races instead
    python compare_last3_races.py --runs 5000  # more MC runs per race
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _THIS_DIR)

from backtest import evaluate_race, N_BACKTEST_RUNS_DEFAULT
from simulation.simulation import _load_model as _load_model_original
from models.race_model import MASTER_FEATURES_PATH as REAL_MASTER_FEATURES_PATH
from oracle import CIRCUIT_TO_RACE_NAME

MASTER_FEATURES_V2_PATH = os.path.join(_THIS_DIR, "master_features_v2.parquet")
MODEL_V2_PATH = os.path.join(_THIS_DIR, "race_model_v2.joblib")


def _load_model_v2() -> list:
    if not os.path.exists(MODEL_V2_PATH):
        raise FileNotFoundError(f"No trained v2 model at {MODEL_V2_PATH} -- run train_v2.py first")
    ensemble = joblib.load(MODEL_V2_PATH)
    if not isinstance(ensemble, list):
        ensemble = [ensemble]
    return ensemble


def get_race_features(path: str, season: int, round_num: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    race = df[(df["season"] == season) & (df["round"] == round_num)].reset_index(drop=True)
    if race.empty:
        raise ValueError(f"No data for season={season}, round={round_num} in {path}")
    return race


def find_last_n_races(path: str, n: int) -> list[tuple[int, int]]:
    df = pd.read_parquet(path)
    races = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    return list(races.tail(n).itertuples(index=False, name=None))


def print_race_comparison(season, round_num, circuit_id, result_orig, result_v2) -> tuple[float, float]:
    race_name = CIRCUIT_TO_RACE_NAME.get(circuit_id, circuit_id)
    n = len(result_orig["driverId"])
    order = sorted(range(n), key=lambda i: result_orig["actual_position"][i])
    v2_driver_list = list(result_v2["driverId"])

    print(f"\n{'='*100}")
    print(f"{race_name} {season} (Round {round_num}) -- ORIGINAL vs V2 (rolling teammate_delta)")
    print(f"{'='*100}")
    print(f"{'Driver':<16}{'Grid':>5}{'Actual':>7}   "
          f"{'Orig Win':>9}{'V2 Win':>8}   {'Orig Brier':>11}{'V2 Brier':>10}")
    print("-" * 100)
    for i in order:
        driver = result_orig["driverId"][i]
        v2_idx = v2_driver_list.index(driver)
        actual = int(result_orig["actual_position"][i])
        grid = int(result_orig["grid_position"][i])
        orig_win = result_orig["model_win_prob"][i]
        v2_win = result_v2["model_win_prob"][v2_idx]
        orig_brier = result_orig["model_full_brier"][i]
        v2_brier = result_v2["model_full_brier"][v2_idx]

        flag = ""
        if v2_brier < orig_brier - 0.02:
            flag = "  <- v2 better here"
        elif v2_brier > orig_brier + 0.02:
            flag = "  <- v2 worse here"

        print(f"{driver:<16}{grid:>5}{actual:>7}   "
              f"{orig_win:>8.1%}{v2_win:>8.1%}   {orig_brier:>11.3f}{v2_brier:>10.3f}{flag}")

    orig_mean = float(result_orig["model_full_brier"].mean())
    v2_mean = float(result_v2["model_full_brier"].mean())
    verdict = "v2 better" if v2_mean < orig_mean else ("original better" if orig_mean < v2_mean else "tie")
    print(f"\nRace-level mean Brier (full distribution): "
          f"original={orig_mean:.3f}  v2={v2_mean:.3f}  ({verdict})")
    print(f"{'='*100}\n")

    return orig_mean, v2_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare original vs v2 model on the last N raced rounds")
    parser.add_argument("--n", type=int, default=3,
                         help="Number of most recent races to compare (default: 3)")
    parser.add_argument("--runs", type=int, default=N_BACKTEST_RUNS_DEFAULT,
                         help=f"Monte Carlo runs per race (default: {N_BACKTEST_RUNS_DEFAULT})")
    args = parser.parse_args()

    if not os.path.exists(MASTER_FEATURES_V2_PATH):
        print(f"Error: {MASTER_FEATURES_V2_PATH} not found. Run build_master_features_v2.py first.")
        sys.exit(1)
    if not os.path.exists(MODEL_V2_PATH):
        print(f"Error: {MODEL_V2_PATH} not found. Run train_v2.py first.")
        sys.exit(1)

    print("Loading original (real project) model ensemble...")
    ensemble_orig = _load_model_original()

    print("Loading v2 (rolling teammate_delta) model ensemble...")
    ensemble_v2 = _load_model_v2()

    races = find_last_n_races(MASTER_FEATURES_V2_PATH, args.n)
    print(f"\nComparing last {len(races)} raced rounds: {races}\n")

    orig_briers, v2_briers = [], []

    for season, round_num in races:
        print(f"Running season={season}, round={round_num} "
              f"({args.runs:,} Monte Carlo runs, twice -- original + v2)...")
        race_df_orig = get_race_features(REAL_MASTER_FEATURES_PATH, season, round_num)
        race_df_v2 = get_race_features(MASTER_FEATURES_V2_PATH, season, round_num)
        circuit_id = race_df_orig["circuitId"].iloc[0]

        result_orig = evaluate_race(race_df_orig, ensemble_orig, n_runs=args.runs)
        result_v2 = evaluate_race(race_df_v2, ensemble_v2, n_runs=args.runs)

        orig_mean, v2_mean = print_race_comparison(
            season, round_num, circuit_id, result_orig, result_v2
        )
        orig_briers.append(orig_mean)
        v2_briers.append(v2_mean)

    print(f"\n{'='*70}")
    print(f"SUMMARY -- last {len(races)} races")
    print(f"{'='*70}")
    print(f"{'Race':<20}{'Original Brier':>16}{'V2 Brier':>12}")
    print("-" * 70)
    for (season, round_num), ob, vb in zip(races, orig_briers, v2_briers):
        label = f"{season} R{round_num}"
        print(f"{label:<20}{ob:>16.3f}{vb:>12.3f}")
    print("-" * 70)
    avg_orig = float(np.mean(orig_briers))
    avg_v2 = float(np.mean(v2_briers))
    print(f"{'Average':<20}{avg_orig:>16.3f}{avg_v2:>12.3f}")
    print(f"{'='*70}\n")

    if avg_v2 < avg_orig:
        pct = (avg_orig - avg_v2) / avg_orig * 100
        print(f"v2 (rolling teammate_delta) scored better on average across "
              f"these {len(races)} races -- {pct:.1f}% lower mean Brier.")
    elif avg_v2 > avg_orig:
        pct = (avg_v2 - avg_orig) / avg_orig * 100
        print(f"Original (career-average teammate_delta) scored better on average "
              f"across these {len(races)} races -- v2 was {pct:.1f}% higher mean Brier.")
    else:
        print("Tied on average across these races.")

    print("\nNote: 3 races is a very small sample -- treat this as a spot check")
    print("alongside the 78-race backtest_v2.py result, not a replacement for it.")
