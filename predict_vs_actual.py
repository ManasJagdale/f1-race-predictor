"""
predict_vs_actual.py

Single-race diagnostic: runs the simulation for one PAST race (one that
already has recorded results) and prints predicted probabilities next
to what actually happened — driver by driver, plus per-race Brier
scores for both model and naive baseline.

This exists because backtest.py only reports AGGREGATE Brier scores
across ~78 races. That's the right way to judge whether the model
adds value overall, but it can't tell you why one specific race (e.g.
"why was Antonelli's predicted win probability so low when he actually
podiumed?") went the way it did. This tool answers that second kind
of question.

Reuses backtest.py's evaluate_race() directly — not a reimplementation
— so the numbers here are guaranteed consistent with what the full
backtest already reported.

NOTE ON TRAINING: the ranking model (models/race_model.py) is trained
ONLY on 2020-2022 pairwise comparisons (TRAIN_SEASONS). Any race in
2023 onwards — including recent 2026 races — was NOT used to train the
model's learned weights, regardless of whether it "already happened."
Rolling features (Elo, recent form, pace delta, DNF rate) DO use real
history up to the race in question, which is correct and leak-free —
only the model's decision boundary itself is frozen to 2020-2022.

Usage (from project root):
    python predict_vs_actual.py --season 2026 --round 7
    python predict_vs_actual.py --race "Canadian Grand Prix 2026"
    python predict_vs_actual.py --season 2026 --round 7 --runs 20000
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.race_model import MASTER_FEATURES_PATH
from simulation.simulation import _load_model, get_race_features, N_RUNS_DEFAULT
from backtest import evaluate_race
from oracle import resolve_race, CIRCUIT_TO_RACE_NAME


def print_comparison(result: dict, season: int, round_num: int, circuit_id: str) -> None:
    race_name = CIRCUIT_TO_RACE_NAME.get(circuit_id, circuit_id)
    n = len(result["driverId"])

    print(f"\n{'='*90}")
    print(f"{race_name} {season} (Round {round_num}) — Predicted vs Actual")
    print(f"{'='*90}")

    # Sort by actual finishing position — read top-down like a real result sheet
    order = sorted(range(n), key=lambda i: result["actual_position"][i])

    print(f"\n{'Driver':<18} {'Grid':>5} {'Actual':>7} {'M.Win':>7} {'N.Win':>7} "
          f"{'M.Top3':>7} {'M.Top6':>7} {'M.Brier':>9} {'N.Brier':>9}")
    print("-" * 90)
    for i in order:
        actual = int(result["actual_position"][i])
        grid = int(result["grid_position"][i])
        model_w = result["model_win_prob"][i]
        naive_w = result["naive_win_prob"][i]
        model_t3 = result["model_top3_prob"][i]
        model_t6 = result["model_top6_prob"][i]
        model_b = result["model_full_brier"][i]
        naive_b = result["naive_full_brier"][i]

        # Flag notable mismatches. Uses P(Top 3) — not P(Win) — to judge
        # whether a podium was surprising: a low win probability from
        # outside the front row is normal in F1 and not itself a miss;
        # what matters is whether the model rated a TOP-3 finish unlikely.
        flag = ""
        if actual <= 3 and model_t3 < 0.15:
            flag = "  ← podium, model gave <15% top-3 chance"
        elif actual == 1 and model_w < result["model_win_prob"].max() * 0.5:
            flag = "  ← won, wasn't the model's top pick"

        print(f"{result['driverId'][i]:<18} {grid:>5} {actual:>7} "
              f"{model_w:>6.1%} {naive_w:>6.1%} "
              f"{model_t3:>6.1%} {model_t6:>6.1%} "
              f"{model_b:>9.3f} {naive_b:>9.3f}{flag}")

    print(f"\n{'-'*90}")
    print(f"Race-level mean Brier (full distribution): "
          f"model={result['model_full_brier'].mean():.3f}  "
          f"naive={result['naive_full_brier'].mean():.3f}  "
          f"{'✓ model better' if result['model_full_brier'].mean() < result['naive_full_brier'].mean() else '✗ naive better'} "
          f"(single-race result — not a statement about overall model quality; "
          f"see backtest.py for the aggregate picture across many races)")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare predicted vs actual outcome for one past race")
    parser.add_argument("--race", type=str, default=None,
                         help='Race query, e.g. "Canadian Grand Prix 2026"')
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--round", type=int, default=None, dest="round_num")
    parser.add_argument("--runs", type=int, default=N_RUNS_DEFAULT,
                         help=f"Monte Carlo runs (default: {N_RUNS_DEFAULT:,})")
    args = parser.parse_args()

    print("Loading master features...")
    master_df = pd.read_parquet(MASTER_FEATURES_PATH)

    if args.season is not None and args.round_num is not None:
        season, round_num = args.season, args.round_num
        match = master_df[(master_df["season"] == season) & (master_df["round"] == round_num)]
        if match.empty:
            print(f"No data for season={season}, round={round_num}.")
            sys.exit(1)
        circuit_id = match["circuitId"].iloc[0]
    elif args.race is not None:
        try:
            season, round_num, circuit_id = resolve_race(args.race, master_df)
        except ValueError as e:
            print(f"\nError: {e}\n")
            sys.exit(1)
    else:
        print('Provide either --race "<name> <year>" or both --season and --round.')
        sys.exit(1)

    print("Loading trained model ensemble...")
    ensemble = _load_model()

    print(f"Loading race features (season={season}, round={round_num}, circuit={circuit_id})...")
    race_df = get_race_features(season, round_num)
    print(f"  {len(race_df)} drivers found")

    print(f"Running simulation ({args.runs:,} Monte Carlo runs)...")
    result = evaluate_race(race_df, ensemble, n_runs=args.runs)

    print_comparison(result, season, round_num, circuit_id)
