"""
models/race_model.py

Trains a learning-to-rank ENSEMBLE for F1 finishing position prediction.

Core insight: F1 prediction is a ranking problem, not a regression problem.
We want to know "who finishes ahead of whom in this race", not "what
absolute position does this driver get".

Approach: for each race, generate all pairwise driver comparisons.
Train a classifier on "does driver A finish ahead of driver B?" given
their feature differences. At prediction time, score each driver by
how many other drivers they're predicted to beat — this produces a
ranking within the race.

This is the standard LTR (Learning to Rank) approach used in search
and sports prediction. It beats global regression because:
  - The model explicitly learns relative ordering
  - Features are compared in-race context (same track, same conditions)
  - Field size variation doesn't affect the model

WHY AN ENSEMBLE (added after stability_check.py diagnostic):
A single GradientBoostingClassifier trained on one random_state produced
wildly different results depending on the seed — 5 of 8 tested seeds beat
the naive grid-position baseline, 3 lost, with MAE ranging 3.270-3.645
(std dev 0.110) against a "signal" (model vs naive gap) of only ~0.06.
Root cause: only 61 actual races (2020-2022) back the 23,180 pairwise
training rows — driver pairs within a race are correlated, so the true
effective sample size is much smaller than the row count suggests. One
seed's random initialization can easily latch onto a handful of chaotic
test-set races and swing MAE substantially.

Fix: bagging. Train N models with different seeds, average their pairwise
win-probabilities at prediction time. This is standard variance reduction
— it won't fix the underlying data-thinness problem, but it stops a
single unlucky seed from determining whether the model "works" or not,
and produces steadier P1-P20 probabilities for simulation.py to build on.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DIR, TRAIN_SEASONS, TEST_SEASONS

MASTER_FEATURES_PATH = os.path.join(PROCESSED_DIR, "master_features.parquet")
MODEL_PATH           = os.path.join(PROCESSED_DIR, "race_model.joblib")

# Same 8 seeds used in stability_check.py's diagnostic run, for continuity
# with the numbers you already saw (5/8 beating naive, MAE 3.270-3.645).
ENSEMBLE_SEEDS = list(range(8))

FEATURE_COLS = [
    "elo_pre_race",
    "recent_form",
    "teammate_delta",
    "pace_delta_vs_pole",
    "dnf_rate",
    "straight_line_exposure",
    "cornering_exposure",
    "full_throttle_pct",
    "avg_corner_speed",
    "lap_length_km",
    "drs_zones",
    "is_street_circuit",
    "wet_flag",
    "grid_position",
]


# ---------------------------------------------------------------------------
# Section 1: Build pairwise training data
# ---------------------------------------------------------------------------
# For each race, take every pair of drivers (A, B).
# Features = difference vector: features_A - features_B
# Label = 1 if A finishes ahead of B (lower position number), else 0.
#
# This doubles the data symmetrically — we also add (B-A, 0) for each
# (A-B, 1) pair, which helps the model learn antisymmetry.
#
# With 20 drivers per race: 20×19/2 = 190 pairs per race × 2 (symmetric)
# = 380 pairwise rows per race. ~60 races training → ~23,000 pairs.
#
# Built ONCE and shared across all ensemble members — only the model's
# random_state differs between members, not the training data itself.

def build_pairwise(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert per-driver race rows into pairwise comparison training data.

    Returns:
        X: array of feature difference vectors, shape (n_pairs, n_features)
        y: array of labels (1 = first driver beats second), shape (n_pairs,)
    """
    X_rows = []
    y_rows = []

    for (season, rnd), race in df.groupby(["season", "round"]):
        drivers = race.reset_index(drop=True)
        feats   = drivers[FEATURE_COLS].values
        pos     = drivers["finishing_position"].values

        for i, j in combinations(range(len(drivers)), 2):
            diff_ij = feats[i] - feats[j]
            diff_ji = feats[j] - feats[i]
            label_ij = 1 if pos[i] < pos[j] else 0  # lower pos = better
            label_ji = 1 - label_ij

            X_rows.append(diff_ij)
            y_rows.append(label_ij)
            X_rows.append(diff_ji)
            y_rows.append(label_ji)

    return np.array(X_rows), np.array(y_rows)


# ---------------------------------------------------------------------------
# Section 2: Score drivers within a race — ENSEMBLE AVERAGED
# ---------------------------------------------------------------------------
# At prediction time, for each race:
#   1. Get all driver features
#   2. For each pair, average P(i beats j) across all ensemble members
#   3. For each driver, sum averaged win-probabilities against every
#      other driver → predicted rank (higher score = better)

def predict_race_rankings(ensemble: list, race_df: pd.DataFrame) -> np.ndarray:
    """
    Predict finishing order for a single race using pairwise comparisons,
    averaged across every model in the ensemble.

    Args:
        ensemble: list of trained Pipeline objects (different random_state)
        race_df: driver rows for one race

    Returns:
        array indexed like race_df with predicted rank (1 = best)
    """
    drivers = race_df.reset_index(drop=True)
    n       = len(drivers)
    feats   = drivers[FEATURE_COLS].values
    scores  = np.zeros(n)

    for i, j in combinations(range(n), 2):
        diff = feats[i] - feats[j]
        # Average P(i beats j) across every ensemble member rather than
        # trusting a single model's (possibly seed-biased) estimate.
        probs = [pipeline.predict_proba([diff])[0][1] for pipeline in ensemble]
        prob  = float(np.mean(probs))
        scores[i] += prob
        scores[j] += (1 - prob)

    # Rank: highest score = P1 (rank 1)
    ranks = pd.Series(scores).rank(ascending=False, method="min")
    return ranks.values


# ---------------------------------------------------------------------------
# Section 3: Evaluate on test set
# ---------------------------------------------------------------------------

def evaluate_on_test(ensemble: list, test_df: pd.DataFrame):
    """
    Evaluate by predicting full race rankings (ensemble-averaged) and
    comparing to actual. MAE in position units.
    """
    all_pred_ranks = []
    all_true_pos   = []

    for (season, rnd), race in test_df.groupby(["season", "round"]):
        pred_ranks = predict_race_rankings(ensemble, race)
        all_pred_ranks.extend(pred_ranks)
        all_true_pos.extend(race["finishing_position"].values)

    model_mae = mean_absolute_error(all_true_pos, all_pred_ranks)

    # Naive baseline: predicted rank = grid_position
    naive_preds = test_df["grid_position"].values
    naive_mae   = mean_absolute_error(
        test_df["finishing_position"].values,
        naive_preds
    )

    return model_mae, naive_mae


# ---------------------------------------------------------------------------
# Section 4: Train ensemble
# ---------------------------------------------------------------------------

def _train_single(X_train, y_train, seed: int) -> Pipeline:
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


def train_ensemble(X_train, y_train, seeds: list = ENSEMBLE_SEEDS) -> list:
    """Train one model per seed on the SAME pairwise training data."""
    ensemble = []
    for seed in seeds:
        print(f"    Training member seed={seed}...")
        ensemble.append(_train_single(X_train, y_train, seed))
    return ensemble


# ---------------------------------------------------------------------------
# Section 5: Feature importance — averaged across ensemble
# ---------------------------------------------------------------------------

def print_importances(ensemble: list):
    """
    Average feature_importances_ across every ensemble member, and show
    the spread (std dev) so it's visible which features are consistently
    important vs which ones only mattered to a subset of seeds.
    """
    all_importances = np.array([
        pipeline.named_steps["model"].feature_importances_ for pipeline in ensemble
    ])  # shape: (n_seeds, n_features)

    mean_imp = all_importances.mean(axis=0)
    std_imp  = all_importances.std(axis=0)

    ranked = sorted(zip(FEATURE_COLS, mean_imp, std_imp), key=lambda x: x[1], reverse=True)

    print(f"\n  Feature importances (mean ± std across {len(ensemble)} ensemble members):")
    for feat, mean_val, std_val in ranked:
        bar = "█" * int(mean_val * 40)
        print(f"    {feat:<25} {mean_val:.4f} ± {std_val:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nLoading master features...")
    df = pd.read_parquet(MASTER_FEATURES_PATH)

    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test_df  = df[df["season"].isin(TEST_SEASONS)].copy()

    print(f"  Train: {len(train_df)} rows → building pairwise comparisons...")
    X_train, y_train = build_pairwise(train_df)
    print(f"  Pairwise training rows: {len(X_train):,}")

    print(f"  Test:  {len(test_df)} rows ({test_df['season'].min()}–{test_df['season'].max()})")

    print(f"\nTraining ensemble of {len(ENSEMBLE_SEEDS)} models (seeds {ENSEMBLE_SEEDS})...")
    ensemble = train_ensemble(X_train, y_train)
    print("  ✓ Ensemble training complete")

    print("\nEvaluating ensemble on test set (2023–2026)...")
    model_mae, naive_mae = evaluate_on_test(ensemble, test_df)

    print(f"\n  Ensemble MAE:       {model_mae:.3f} positions")
    print(f"  Naive baseline MAE: {naive_mae:.3f} positions")

    if model_mae < naive_mae:
        improvement = (naive_mae - model_mae) / naive_mae * 100
        print(f"  ✓ Ensemble beats naive baseline by {improvement:.1f}%")
    else:
        print(f"  ✗ Still worse than naive by {model_mae - naive_mae:.3f}")

    print_importances(ensemble)

    print(f"\nSaving ensemble ({len(ensemble)} models) to {MODEL_PATH}...")
    joblib.dump(ensemble, MODEL_PATH)
    print("  ✓ Saved\n")
