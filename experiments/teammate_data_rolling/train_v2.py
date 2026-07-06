"""
experiments/teammate_delta_rolling/train_v2.py

Trains the same 8-seed ensemble architecture as models/race_model.py,
but on master_features_v2.parquet (rolling teammate_delta) instead of
the real project's master_features.parquet.

Reuses build_pairwise(), train_ensemble(), evaluate_on_test(), and
print_importances() DIRECTLY from the real models/race_model.py --
none of those functions are modified or reimplemented, since none of
them depend on how teammate_delta was computed upstream; they just
process whatever's in the FEATURE_COLS columns of whatever DataFrame
they're given.

WRITES ONLY WITHIN THIS FOLDER:
    experiments/teammate_delta_rolling/race_model_v2.joblib

Usage:
    python train_v2.py
"""

import os
import sys
import joblib
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from models.race_model import (
    build_pairwise,
    train_ensemble,
    evaluate_on_test,
    print_importances,
    TRAIN_SEASONS,
    TEST_SEASONS,
    ENSEMBLE_SEEDS,
)

MASTER_FEATURES_V2_PATH = os.path.join(_THIS_DIR, "master_features_v2.parquet")
MODEL_V2_PATH = os.path.join(_THIS_DIR, "race_model_v2.joblib")


if __name__ == "__main__":
    if not os.path.exists(MASTER_FEATURES_V2_PATH):
        print(f"Error: {MASTER_FEATURES_V2_PATH} not found. "
              f"Run `python build_master_features_v2.py` first.")
        sys.exit(1)

    print("\n[v2] Loading master features (rolling teammate_delta)...")
    df = pd.read_parquet(MASTER_FEATURES_V2_PATH)

    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test_df = df[df["season"].isin(TEST_SEASONS)].copy()

    print(f"  Train: {len(train_df)} rows → building pairwise comparisons...")
    X_train, y_train = build_pairwise(train_df)
    print(f"  Pairwise training rows: {len(X_train):,}")
    print(f"  Test:  {len(test_df)} rows ({test_df['season'].min()}–{test_df['season'].max()})")

    print(f"\nTraining ensemble of {len(ENSEMBLE_SEEDS)} models "
          f"(seeds {ENSEMBLE_SEEDS}) on v2 features...")
    ensemble = train_ensemble(X_train, y_train)
    print("  ✓ [v2] Ensemble training complete")

    print("\nEvaluating v2 ensemble on test set...")
    model_mae, naive_mae = evaluate_on_test(ensemble, test_df)

    print(f"\n  [v2] Ensemble MAE:  {model_mae:.3f} positions")
    print(f"  Naive baseline MAE: {naive_mae:.3f} positions")
    if model_mae < naive_mae:
        improvement = (naive_mae - model_mae) / naive_mae * 100
        print(f"  ✓ [v2] Ensemble beats naive baseline by {improvement:.1f}%")
    else:
        print(f"  ✗ [v2] Still worse than naive by {model_mae - naive_mae:.3f}")

    print("\nNote: this raw ranking MAE is known to be noisy/seed-sensitive")
    print("(see stability_check.py in the real project) -- the number that")
    print("actually matters for comparison is the Brier skill score from")
    print("backtest_v2.py, not this MAE figure. Run that next.")

    print_importances(ensemble)

    print(f"\nSaving v2 ensemble to {MODEL_V2_PATH}...")
    joblib.dump(ensemble, MODEL_V2_PATH)
    print("  ✓ Saved\n")
