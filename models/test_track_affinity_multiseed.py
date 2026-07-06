"""
models/test_track_affinity_multiseed.py

Follow-up to test_track_affinity.py. That script found track_affinity
made MAE worse on a SINGLE seed (random_state=42) — but this project's
own prior work (the v1 seed-sensitivity finding: only 5/8 seeds beat the
naive baseline) already established that single-seed comparisons here are
unreliable, since seed noise alone is roughly the same size as the signal
being measured. A one-seed "worse" result could just as easily be an
unlucky seed as a real effect.

This script re-runs the same A/B comparison (baseline vs baseline +
track_affinity) across all 8 ensemble seeds (0-7, matching
race_model.py's ENSEMBLE_SEEDS / stability_check.py's SEEDS), and reports
average MAE and how many seeds each config wins — not just one number.

Usage (from project root):
    python -m models.test_track_affinity_multiseed
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DIR, TRAIN_SEASONS
from models.compare_feature_sets import (
    ALL_FEATURE_COLS,
    TEST_SEASONS,
    build_pairwise,
    evaluate_on_test,
)

MASTER_FEATURES_PATH = os.path.join(PROCESSED_DIR, "master_features.parquet")
SEEDS = list(range(8))  # matches race_model.py's ENSEMBLE_SEEDS exactly

CONFIGS = {
    "A_baseline_no_affinity": ALL_FEATURE_COLS,
    "B_with_track_affinity":  ALL_FEATURE_COLS + ["track_affinity"],
}


def train_with_seed(X_train, y_train, seed: int) -> Pipeline:
    """Same as compare_feature_sets.train(), but seed is a parameter
    instead of hardcoded, so we can sweep it."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=seed,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


if __name__ == "__main__":
    print("\nLoading master features...")
    df = pd.read_parquet(MASTER_FEATURES_PATH)
    print(f"  {len(df)} rows, seasons {df['season'].min()}-{df['season'].max()}")

    if "track_affinity" not in df.columns:
        print("\n✗ track_affinity column not found. Run "
              "`python -m data.build_master_features` first.")
        sys.exit(1)

    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test_df = df[df["season"].isin(TEST_SEASONS)].copy()

    all_results = {name: [] for name in CONFIGS}

    for name, feature_cols in CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Config: {name}  ({len(feature_cols)} features)")
        print(f"{'='*70}")
        X_train, y_train = build_pairwise(train_df, feature_cols)

        for seed in SEEDS:
            pipeline = train_with_seed(X_train, y_train, seed)
            model_mae, naive_mae = evaluate_on_test(pipeline, test_df, feature_cols)
            beats_naive = model_mae < naive_mae
            all_results[name].append(model_mae)
            print(f"  seed {seed}: model MAE = {model_mae:.4f}  "
                  f"(naive = {naive_mae:.4f})  beats_naive = {beats_naive}")

    print(f"\n{'='*70}")
    print("SUMMARY — averaged across all 8 seeds")
    print(f"{'='*70}")

    baseline_maes = all_results["A_baseline_no_affinity"]
    affinity_maes = all_results["B_with_track_affinity"]

    baseline_mean, baseline_std = np.mean(baseline_maes), np.std(baseline_maes)
    affinity_mean, affinity_std = np.mean(affinity_maes), np.std(affinity_maes)

    print(f"  A_baseline_no_affinity:  mean MAE = {baseline_mean:.4f}  (std {baseline_std:.4f})")
    print(f"  B_with_track_affinity:   mean MAE = {affinity_mean:.4f}  (std {affinity_std:.4f})")
    print(f"  Delta (positive = affinity better): {baseline_mean - affinity_mean:+.4f}")

    wins = sum(1 for b, a in zip(baseline_maes, affinity_maes) if a < b)
    print(f"\n  track_affinity beat baseline on {wins}/8 seeds")

    print("\n  Interpretation:")
    delta = baseline_mean - affinity_mean
    # Use the std of the difference across seeds as a rough noise floor —
    # if the mean delta is smaller than seed-to-seed variability, this
    # isn't a real effect either way, same logic as the v1 stability check.
    diffs = [b - a for b, a in zip(baseline_maes, affinity_maes)]
    diff_std = np.std(diffs)
    if abs(delta) < diff_std:
        print(f"    Mean delta ({delta:+.4f}) is smaller than seed-to-seed noise")
        print(f"    (std of per-seed differences: {diff_std:.4f}). This is NOT a")
        print("    reliable signal in either direction — track_affinity is roughly")
        print("    neutral at current shrinkage settings, not clearly good or bad.")
    elif delta > 0:
        print(f"    track_affinity improved MAE by {delta:.4f} on average, and this")
        print("    exceeds seed-to-seed noise — plausibly a real (if modest) gain.")
    else:
        print(f"    track_affinity worsened MAE by {abs(delta):.4f} on average, and")
        print("    this exceeds seed-to-seed noise — plausibly a real regression.")
