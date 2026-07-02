"""
models/stability_check.py

Diagnostic: is the model's "beats naive baseline" result a stable finding,
or noise from GradientBoostingClassifier's random_state=42?

Context: the DNF shrinkage fix in car_performance.py changed dnf_rate
values (a feature with importance ~0.0124-0.0149, one of the weakest in
the model) and MAE moved from 3.299 (beats naive by 1.7%) to 3.476
(loses to naive by 3.5%). That's a large swing for a low-importance
feature to cause on its own — this script checks whether that swing is
within the model's normal seed-to-seed variance, or whether something
else is going on.

Retrains the SAME architecture (same features, same window, same
hyperparameters except random_state) across N different seeds and
reports the spread of MAE / naive-beat outcomes. Uses the CURRENT
master_features.parquet on disk — run this AFTER regenerating features,
so it reflects the shrinkage-adjusted dnf_rate.

Usage (from project root):
    python -m models.stability_check
    python -m models.stability_check --seeds 10
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DIR
from models.race_model import FEATURE_COLS, TRAIN_SEASONS, TEST_SEASONS

MASTER_FEATURES_PATH = os.path.join(PROCESSED_DIR, "master_features.parquet")

DEFAULT_N_SEEDS = 8
SEEDS = list(range(DEFAULT_N_SEEDS))  # 0, 1, 2, ... — arbitrary, just need variety


# ---------------------------------------------------------------------------
# Same logic as race_model.py, parametrized by random_state
# ---------------------------------------------------------------------------

def build_pairwise(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X_rows, y_rows = [], []
    for (season, rnd), race in df.groupby(["season", "round"]):
        drivers = race.reset_index(drop=True)
        feats = drivers[FEATURE_COLS].values
        pos = drivers["finishing_position"].values

        for i, j in combinations(range(len(drivers)), 2):
            diff_ij = feats[i] - feats[j]
            diff_ji = feats[j] - feats[i]
            label_ij = 1 if pos[i] < pos[j] else 0
            label_ji = 1 - label_ij

            X_rows.append(diff_ij)
            y_rows.append(label_ij)
            X_rows.append(diff_ji)
            y_rows.append(label_ji)

    return np.array(X_rows), np.array(y_rows)


def predict_race_rankings(pipeline, race_df: pd.DataFrame) -> np.ndarray:
    drivers = race_df.reset_index(drop=True)
    n = len(drivers)
    feats = drivers[FEATURE_COLS].values
    scores = np.zeros(n)

    for i, j in combinations(range(n), 2):
        diff = feats[i] - feats[j]
        prob = pipeline.predict_proba([diff])[0][1]
        scores[i] += prob
        scores[j] += (1 - prob)

    ranks = pd.Series(scores).rank(ascending=False, method="min")
    return ranks.values


def evaluate_on_test(pipeline, test_df: pd.DataFrame):
    all_pred_ranks, all_true_pos = [], []
    for (season, rnd), race in test_df.groupby(["season", "round"]):
        pred_ranks = predict_race_rankings(pipeline, race)
        all_pred_ranks.extend(pred_ranks)
        all_true_pos.extend(race["finishing_position"].values)

    model_mae = mean_absolute_error(all_true_pos, all_pred_ranks)
    naive_preds = test_df["grid_position"].values
    naive_mae = mean_absolute_error(test_df["finishing_position"].values, naive_preds)
    return model_mae, naive_mae


def train(X_train, y_train, seed: int) -> Pipeline:
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS,
                         help="Number of random seeds to test")
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    print("\nLoading master features...")
    df = pd.read_parquet(MASTER_FEATURES_PATH)

    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test_df = df[df["season"].isin(TEST_SEASONS)].copy()

    print(f"  Train: {len(train_df)} rows → building pairwise comparisons "
          f"(shared across all seeds — only the model init changes)...")
    X_train, y_train = build_pairwise(train_df)
    print(f"  Pairwise training rows: {len(X_train):,}")
    print(f"  Test:  {len(test_df)} rows ({test_df['season'].min()}-{test_df['season'].max()})")

    print(f"\nTraining {len(seeds)} models with different random_state values...")
    print(f"{'Seed':<8} {'Model MAE':>10} {'Naive MAE':>10} {'Result':>18}")
    print("-" * 50)

    results = []
    for seed in seeds:
        pipeline = train(X_train, y_train, seed)
        model_mae, naive_mae = evaluate_on_test(pipeline, test_df)
        beats = model_mae < naive_mae
        pct = (naive_mae - model_mae) / naive_mae * 100
        results.append({"seed": seed, "model_mae": model_mae, "naive_mae": naive_mae,
                         "beats_naive": beats, "pct": pct})
        tag = f"beats by {pct:.1f}%" if beats else f"loses by {-pct:.1f}%"
        print(f"{seed:<8} {model_mae:>10.3f} {naive_mae:>10.3f} {tag:>18}")

    maes = [r["model_mae"] for r in results]
    n_beats = sum(r["beats_naive"] for r in results)

    print(f"\n{'='*50}")
    print("STABILITY SUMMARY")
    print(f"{'='*50}")
    print(f"  MAE range:    {min(maes):.3f} - {max(maes):.3f}  (spread: {max(maes)-min(maes):.3f})")
    print(f"  MAE mean:     {np.mean(maes):.3f}")
    print(f"  MAE std dev:  {np.std(maes):.3f}")
    print(f"  Naive MAE:    {results[0]['naive_mae']:.3f}  (constant across seeds)")
    print(f"  Seeds beating naive: {n_beats} / {len(seeds)}")

    print(f"\n{'='*50}")
    if n_beats == len(seeds):
        print("VERDICT: Model reliably beats naive across all tested seeds.")
        print("The single seed=42 regression to 3.476 was likely an unlucky")
        print("outlier, not representative of the model's typical performance.")
    elif n_beats == 0:
        print("VERDICT: Model does NOT reliably beat naive on any tested seed.")
        print("The earlier 'beats naive' result may have been a lucky seed,")
        print("not a robust finding. Worth reconsidering model architecture")
        print("or feature set rather than chasing a specific seed.")
    else:
        print(f"VERDICT: Mixed — beats naive in {n_beats}/{len(seeds)} seeds.")
        print("The model's edge over naive baseline is real but fragile —")
        print("small perturbations (seed, or minor feature changes like the")
        print("DNF shrinkage fix) can flip the result either way. Treat the")
        print("1.7% margin as noisy rather than a settled result.")
    print(f"{'='*50}\n")
