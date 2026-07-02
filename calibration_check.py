"""
calibration_check.py

Reliability/calibration check: for Win, Top-3, and Top-6 predictions,
buckets the model's predicted probabilities into ranges and compares
the AVERAGE PREDICTED probability in each bucket against the ACTUAL
observed frequency of that event happening, across every driver-race
observation in the backtest set.

WHY THIS EXISTS:
predict_vs_actual.py flagged Antonelli's Round 8 podium as surprising
(model gave 7.8% top-3 chance, he finished P3) — and a second data
point (Hadjar, grid P8, also ~7.2% top-3, despite starting 4 places
behind Antonelli's P4) raised the question of whether the model
systematically under-rates certain drivers' top-3 chances, or whether
this is normal variance from two cherry-picked races. This script
answers that with data across the full test set rather than more
eyeballing.

A well-calibrated model should show predicted ≈ actual in every bucket.
If the 0-10% bucket, say, actually converts to top-3 finishes 20% of
the time, that's real evidence of under-confidence in that range —
exactly the kind of finding that would justify Phase 2 backlog item #9
(probability calibration layer / Platt scaling / isotonic regression).

REQUIRES: backtest_results.parquet must include model_top3_prob and
model_top6_prob columns (added to evaluate_race() alongside the
existing brier columns). If you generated backtest_results.parquet
before that change, re-run `python backtest.py` first.

Usage (from project root):
    python calibration_check.py
    python calibration_check.py --bins 5
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROCESSED_DIR

BACKTEST_RESULTS_PATH = os.path.join(PROCESSED_DIR, "backtest_results.parquet")

REQUIRED_COLS = ["model_win_prob", "model_top3_prob", "model_top6_prob",
                  "actual_position"]


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"\nError: backtest_results.parquet is missing columns: {missing}")
        print("This file was likely generated before model_top3_prob/model_top6_prob")
        print("were added to evaluate_race(). Re-run `python backtest.py` first, then")
        print("try this script again.\n")
        sys.exit(1)


def _build_actual_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["actual_win"]  = (df["actual_position"] == 1).astype(int)
    df["actual_top3"] = (df["actual_position"] <= 3).astype(int)
    df["actual_top6"] = (df["actual_position"] <= 6).astype(int)
    return df


def calibration_table(pred_probs: np.ndarray, actual: np.ndarray, n_bins: int) -> pd.DataFrame:
    """
    Bucket pred_probs into n_bins equal-width ranges [0,1], and for each
    bucket compute: n observations, mean predicted probability, actual
    observed frequency, and the gap between them.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(pred_probs, bin_edges[1:-1]), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = mask.sum()
        if n == 0:
            continue
        mean_pred = pred_probs[mask].mean()
        actual_freq = actual[mask].mean()
        rows.append({
            "bucket": f"{bin_edges[b]:.0%}-{bin_edges[b+1]:.0%}",
            "n": int(n),
            "mean_predicted": mean_pred,
            "actual_frequency": actual_freq,
            "gap": actual_freq - mean_pred,
        })

    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame, total_n: int) -> float:
    """Weighted average of |predicted - actual| across buckets, weighted by n."""
    return (table["n"] / total_n * (table["mean_predicted"] - table["actual_frequency"]).abs()).sum()


def print_calibration_report(df: pd.DataFrame, n_bins: int) -> None:
    events = [
        ("P(Win)",   "model_win_prob",  "actual_win"),
        ("P(Top 3)", "model_top3_prob", "actual_top3"),
        ("P(Top 6)", "model_top6_prob", "actual_top6"),
    ]

    print(f"\n{'='*70}")
    print(f"CALIBRATION CHECK — {len(df)} driver-race observations, {n_bins} buckets")
    print(f"{'='*70}")

    for label, pred_col, actual_col in events:
        table = calibration_table(df[pred_col].values, df[actual_col].values, n_bins)
        ece = expected_calibration_error(table, len(df))

        print(f"\n--- {label} (Expected Calibration Error: {ece:.4f}) ---")
        print(f"{'Bucket':<12} {'n':>6} {'Predicted':>11} {'Actual':>9} {'Gap':>8}  Flag")
        print("-" * 60)
        for _, row in table.iterrows():
            flag = ""
            if abs(row["gap"]) > 0.05 and row["n"] >= 10:
                direction = "under-confident" if row["gap"] > 0 else "over-confident"
                flag = f"  ← {direction} (model {direction.split('-')[0]}-predicts by {abs(row['gap']):.1%})"
            print(f"{row['bucket']:<12} {row['n']:>6} {row['mean_predicted']:>10.1%} "
                  f"{row['actual_frequency']:>8.1%} {row['gap']:>+7.1%}{flag}")

    print(f"\n{'='*70}")
    print("Reading this: a well-calibrated model has 'Actual' close to 'Predicted'")
    print("in every bucket. A consistent positive gap (actual > predicted) in the")
    print("low-probability buckets specifically would mean the model systematically")
    print("under-rates longshot-ish drivers — the pattern the Antonelli/Hadjar")
    print("comparison raised. Buckets with small n are noisy — weight conclusions")
    print("toward buckets with more observations.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check probability calibration against backtest results")
    parser.add_argument("--bins", type=int, default=10,
                         help="Number of probability buckets (default: 10)")
    args = parser.parse_args()

    if not os.path.exists(BACKTEST_RESULTS_PATH):
        print(f"\nError: {BACKTEST_RESULTS_PATH} not found. Run `python backtest.py` first.\n")
        sys.exit(1)

    print(f"Loading {BACKTEST_RESULTS_PATH}...")
    df = pd.read_parquet(BACKTEST_RESULTS_PATH)
    _check_columns(df)
    df = _build_actual_indicators(df)

    print_calibration_report(df, n_bins=args.bins)
