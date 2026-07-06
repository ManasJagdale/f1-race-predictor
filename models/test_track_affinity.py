"""
models/test_track_affinity.py

Answers the question: does driver_track_affinity actually help, or does
empirical Bayes shrinkage collapse it to near-zero noise for most drivers?

Runs 2 configs on the SAME train/test split as race_model.py and
compare_feature_sets.py (train 2020-2022, test 2023-2026), so results are
directly comparable to everything else already backtested in this project:

    A. baseline       — the current 14 features, no track_affinity
    B. with_affinity  — baseline + track_affinity (15 features)

Reuses build_pairwise / train / evaluate_on_test / get_importances /
run_config directly from compare_feature_sets.py — same training logic,
same evaluation logic, no duplicated code to drift out of sync.

Usage (from project root):
    python -m models.test_track_affinity
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DIR, TRAIN_SEASONS
from models.compare_feature_sets import (
    ALL_FEATURE_COLS,
    run_config,
)

MASTER_FEATURES_PATH = os.path.join(PROCESSED_DIR, "master_features.parquet")

CONFIGS = [
    {
        "name": "A_baseline_no_affinity",
        "train_seasons": TRAIN_SEASONS,
        "feature_cols": ALL_FEATURE_COLS,
    },
    {
        "name": "B_with_track_affinity",
        "train_seasons": TRAIN_SEASONS,
        "feature_cols": ALL_FEATURE_COLS + ["track_affinity"],
    },
]


if __name__ == "__main__":
    print("\nLoading master features...")
    df = pd.read_parquet(MASTER_FEATURES_PATH)
    print(f"  {len(df)} rows, seasons {df['season'].min()}-{df['season'].max()}")

    if "track_affinity" not in df.columns:
        print("\n✗ track_affinity column not found in master_features.parquet.")
        print("  Run `python -m data.build_master_features` first (with the")
        print("  driver_track_affinity.py module in place) before running this.")
        sys.exit(1)

    results = []
    for cfg in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Running {cfg['name']}")
        print(f"{'='*70}")
        result = run_config(df, cfg)
        results.append(result)
        print(f"  Train rows:       {result['n_train_rows']}")
        print(f"  Pairwise rows:    {result['n_pairwise_rows']}")
        print(f"  Model MAE:        {result['model_mae']:.4f}")
        print(f"  Naive MAE:        {result['naive_mae']:.4f}")
        print(f"  Improvement:      {result['improvement_pct']:.2f}%")
        print(f"  Beats naive:      {result['beats_naive']}")

    baseline, with_affinity = results

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    delta_mae = baseline["model_mae"] - with_affinity["model_mae"]
    print(f"  Baseline MAE:               {baseline['model_mae']:.4f}")
    print(f"  With track_affinity MAE:    {with_affinity['model_mae']:.4f}")
    print(f"  Delta (positive = better):  {delta_mae:+.4f}")

    print("\n  Feature importance for track_affinity in the 'with' config:")
    affinity_importance = next(
        (imp for imp in with_affinity["importances"] if imp[0] == "track_affinity"),
        None,
    )
    if affinity_importance:
        print(f"    track_affinity importance: {affinity_importance[1]:.4f}")
        rank = sorted(with_affinity["importances"], key=lambda x: -x[1]).index(affinity_importance) + 1
        print(f"    Rank among {len(with_affinity['importances'])} features: #{rank}")
    else:
        print("    (not found in importances — check feature_cols wiring)")

    print("\n  Interpretation:")
    if delta_mae > 0.01:
        print("    MAE improved meaningfully — track_affinity appears to add real signal.")
    elif delta_mae > -0.01:
        print("    MAE roughly unchanged — shrinkage likely collapsed this feature")
        print("    close to zero for most drivers. Marginal or no value as-is.")
    else:
        print("    MAE got WORSE — track_affinity is adding noise, not signal, at")
        print("    current shrinkage settings. Consider increasing SHRINKAGE_STRENGTH")
        print("    in driver_track_affinity.py, or dropping the feature.")
