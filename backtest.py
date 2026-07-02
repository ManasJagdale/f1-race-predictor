"""
backtest.py

Evaluates the full pipeline (ensemble ranking model -> Monte Carlo
simulation -> P1-Pn probability distribution) against a naive baseline
across every race in the test set, using Brier score — the metric
specified in f1_predictor_spec.md.

WHY THIS REUSES simulation.py's run_monte_carlo() DIRECTLY:
This backtest calls the exact same function used for live single-race
predictions, not a separate reimplementation. That matters — if backtest
used its own copy of the simulation logic, it could silently drift from
what actually runs when you predict a real upcoming race, and a good
backtest score wouldn't mean much. Scoring the real pipeline is the
whole point of a backtest.

THE NAIVE BASELINE, PROBABILISTICALLY:
The spec's naive baseline is "qualifying position = finishing position"
— a point prediction. To score it with Brier (which needs probabilities,
not point predictions), it's expressed as a one-hot distribution: 100%
probability on the driver's actual grid position, 0% everywhere else.
This is a deliberately strong, slightly unforgiving baseline — Brier
score punishes overconfidence hard when wrong, so the naive baseline
"pays" for its 100% confidence every time a driver doesn't finish
exactly where they qualified. The model's smoother distribution can
still lose to it if the model isn't actually adding signal, which is
the honest test this is supposed to be.

METRICS REPORTED (lower Brier = better, 0 = perfect):
  - Full distribution Brier — every position class, P1 through Pn
  - P(Win) Brier          — binary event: did this driver win?
  - P(Top 3) Brier        — binary event: podium finish?
  - P(Top 6) Brier        — binary event: points-adjacent finish?
  - Brier Skill Score     — 1 - (model_brier / naive_brier); positive
                            means the model beats naive, the minimum
                            bar set in the spec.

Usage (from project root):
    python backtest.py
    python backtest.py --runs 2000          # more MC runs per race (slower, more stable)
    python backtest.py --seasons 2023 2024  # restrict to a subset of test seasons
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROCESSED_DIR
from models.race_model import MODEL_PATH, MASTER_FEATURES_PATH, TEST_SEASONS
from simulation.simulation import run_monte_carlo, _load_model

BACKTEST_RESULTS_PATH = os.path.join(PROCESSED_DIR, "backtest_results.parquet")

# Fewer Monte Carlo runs per race than the live CLI default (10,000) —
# this runs across ~70-90 races instead of one, so runtime adds up fast.
# 1,000 runs is enough for the position-range probabilities (Win, Top3,
# Top6) to be reasonably stable; increase via --runs if you want tighter
# estimates and can afford the extra time.
N_BACKTEST_RUNS_DEFAULT = 1000


# ---------------------------------------------------------------------------
# Section 1: Build one-hot distributions (actual outcome, naive prediction)
# ---------------------------------------------------------------------------

def _build_onehot(positions: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convert an array of 1-indexed positions into a one-hot matrix,
    shape (len(positions), n_classes). Values outside [1, n_classes]
    (shouldn't normally happen) are clamped into range rather than
    silently dropped, so every row always sums to 1.
    """
    onehot = np.zeros((len(positions), n_classes))
    for i, pos in enumerate(positions):
        idx = int(np.clip(pos, 1, n_classes)) - 1
        onehot[i, idx] = 1.0
    return onehot


# ---------------------------------------------------------------------------
# Section 2: Brier score helpers
# ---------------------------------------------------------------------------

def _multiclass_brier_per_driver(pred_probs: np.ndarray, actual_onehot: np.ndarray) -> np.ndarray:
    """
    Per-driver multiclass Brier score: sum of squared errors across all
    position classes for that driver. Returns shape (n_drivers,) so
    results from races of different field sizes can be pooled together
    and averaged fairly (every driver-race observation counts once).
    """
    return np.sum((pred_probs - actual_onehot) ** 2, axis=1)


def _event_brier_per_driver(pred_prob: np.ndarray, actual_indicator: np.ndarray) -> np.ndarray:
    """Per-driver binary-event Brier score, e.g. for P(Win) or P(Top 3)."""
    return (pred_prob - actual_indicator) ** 2


# ---------------------------------------------------------------------------
# Section 3: Per-race evaluation
# ---------------------------------------------------------------------------

def evaluate_race(race_df: pd.DataFrame, ensemble: list, n_runs: int) -> dict:
    """
    Run the simulation for one race and score model vs naive on Brier
    metrics. Returns a dict of pooled per-driver arrays plus the raw
    per-driver results (for the saved backtest_results.parquet).
    """
    n = len(race_df)

    prob_matrix, _dnf_prob = run_monte_carlo(race_df, ensemble, n_runs=n_runs, verbose=False)

    # Align race_df to the driverId order in prob_matrix (Monte Carlo
    # aggregation doesn't guarantee the same row order as race_df).
    race_ordered = race_df.set_index("driverId").loc[prob_matrix.index].reset_index()

    actual_positions = race_ordered["finishing_position"].astype(int).values
    grid_positions    = race_ordered["grid_position"].astype(int).values

    actual_onehot = _build_onehot(actual_positions, n)
    naive_onehot  = _build_onehot(grid_positions, n)
    model_probs   = prob_matrix.values  # already (n, n), aligned to race_ordered

    # --- Full distribution ---
    model_full_brier = _multiclass_brier_per_driver(model_probs, actual_onehot)
    naive_full_brier = _multiclass_brier_per_driver(naive_onehot, actual_onehot)

    # --- Event-specific: Win, Top 3, Top 6 ---
    k_top3 = min(3, n)
    k_top6 = min(6, n)

    actual_win  = actual_onehot[:, 0]
    actual_top3 = actual_onehot[:, :k_top3].sum(axis=1)
    actual_top6 = actual_onehot[:, :k_top6].sum(axis=1)

    model_win  = model_probs[:, 0]
    model_top3 = model_probs[:, :k_top3].sum(axis=1)
    model_top6 = model_probs[:, :k_top6].sum(axis=1)

    naive_win  = naive_onehot[:, 0]
    naive_top3 = naive_onehot[:, :k_top3].sum(axis=1)
    naive_top6 = naive_onehot[:, :k_top6].sum(axis=1)

    return {
        "driverId":         race_ordered["driverId"].values,
        "season":           race_ordered["season"].values,
        "round":            race_ordered["round"].values,
        "actual_position":  actual_positions,
        "grid_position":    grid_positions,
        "model_win_prob":   model_win,
        "naive_win_prob":   naive_win,
        "model_top3_prob":  model_top3,
        "naive_top3_prob":  naive_top3,
        "model_top6_prob":  model_top6,
        "naive_top6_prob":  naive_top6,
        "model_full_brier": model_full_brier,
        "naive_full_brier": naive_full_brier,
        "model_win_brier":  _event_brier_per_driver(model_win, actual_win),
        "naive_win_brier":  _event_brier_per_driver(naive_win, actual_win),
        "model_top3_brier": _event_brier_per_driver(model_top3, actual_top3),
        "naive_top3_brier": _event_brier_per_driver(naive_top3, actual_top3),
        "model_top6_brier": _event_brier_per_driver(model_top6, actual_top6),
        "naive_top6_brier": _event_brier_per_driver(naive_top6, actual_top6),
    }


# ---------------------------------------------------------------------------
# Section 4: Full backtest loop
# ---------------------------------------------------------------------------

def run_backtest(test_df: pd.DataFrame, ensemble: list, n_runs: int) -> pd.DataFrame:
    """
    Run evaluate_race() across every (season, round) in test_df and pool
    all per-driver results into a single DataFrame.
    """
    races = list(test_df.groupby(["season", "round"]))
    n_races = len(races)

    print(f"\nRunning backtest: {n_races} races, {n_runs:,} Monte Carlo runs each")
    print(f"(this reuses the exact simulation.py pipeline — expect this to take a while)\n")

    all_rows = []
    start = time.time()

    for i, ((season, rnd), race) in enumerate(races):
        if len(race) < 2:
            continue  # can't rank a "race" with fewer than 2 drivers

        result = evaluate_race(race, ensemble, n_runs)
        n_drivers = len(result["driverId"])
        for j in range(n_drivers):
            all_rows.append({k: v[j] for k, v in result.items()})

        if (i + 1) % 10 == 0 or (i + 1) == n_races:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n_races - (i + 1)) / rate if rate > 0 else 0
            print(f"  {i + 1}/{n_races} races evaluated "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Section 5: Summary report
# ---------------------------------------------------------------------------

def _brier_skill_score(model_brier: float, naive_brier: float) -> float:
    """
    1 - (model / naive). Positive = model better than naive.
    0 = exactly as good as naive. Negative = worse than naive.
    """
    if naive_brier == 0:
        return float("nan")
    return 1 - (model_brier / naive_brier)


def print_report(results: pd.DataFrame) -> None:
    n_races = results.groupby(["season", "round"]).ngroups
    n_obs = len(results)

    print(f"\n{'='*70}")
    print(f"BACKTEST REPORT — {n_races} races, {n_obs} driver-race observations")
    print(f"Test seasons: {sorted(results['season'].unique())}")
    print(f"{'='*70}")

    metrics = [
        ("Full P1-Pn distribution", "full_brier"),
        ("P(Win)",                  "win_brier"),
        ("P(Top 3)",                "top3_brier"),
        ("P(Top 6)",                "top6_brier"),
    ]

    print(f"\n{'Metric':<28} {'Model Brier':>12} {'Naive Brier':>12} {'Skill Score':>12}  Result")
    print("-" * 82)

    any_fail = False
    for label, col_suffix in metrics:
        model_col = f"model_{col_suffix}"
        naive_col = f"naive_{col_suffix}"
        model_mean = results[model_col].mean()
        naive_mean = results[naive_col].mean()
        skill = _brier_skill_score(model_mean, naive_mean)
        beats = model_mean < naive_mean
        any_fail = any_fail or not beats
        flag = "✓ beats naive" if beats else "✗ loses to naive"
        print(f"{label:<28} {model_mean:>12.4f} {naive_mean:>12.4f} {skill:>+12.3f}  {flag}")

    print(f"\n{'='*70}")
    primary_model = results["model_full_brier"].mean()
    primary_naive = results["naive_full_brier"].mean()
    if primary_model < primary_naive:
        print("PRIMARY METRIC (full distribution): model beats naive baseline.")
        print("This is the minimum bar set in the spec — the model is adding")
        print("real signal, not noise, once run through the full simulation.")
    else:
        print("PRIMARY METRIC (full distribution): model does NOT beat naive.")
        print("The ranking model's edge over naive on MAE doesn't necessarily")
        print("translate into better-calibrated probabilities once Monte Carlo")
        print("noise, DNF sampling, and safety car/pit stop perturbation are")
        print("folded in. Worth investigating which of those adds the most")
        print("uncertainty relative to the signal the ranking model provides.")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brier score backtest vs naive baseline")
    parser.add_argument("--runs", type=int, default=N_BACKTEST_RUNS_DEFAULT,
                         help="Monte Carlo runs per race (default: 1000)")
    parser.add_argument("--seasons", type=int, nargs="+", default=None,
                         help="Restrict to specific seasons, e.g. --seasons 2023 2024")
    args = parser.parse_args()

    print("Loading trained model ensemble...")
    ensemble = _load_model()

    print("Loading master features...")
    df = pd.read_parquet(MASTER_FEATURES_PATH)

    seasons = args.seasons if args.seasons else list(TEST_SEASONS)
    test_df = df[df["season"].isin(seasons)].copy()
    print(f"  Test set: {len(test_df)} rows, seasons {sorted(test_df['season'].unique())}")

    results = run_backtest(test_df, ensemble, n_runs=args.runs)

    print(f"\nSaving detailed results to {BACKTEST_RESULTS_PATH}...")
    results.to_parquet(BACKTEST_RESULTS_PATH, index=False)
    print("  ✓ Saved")

    print_report(results)
