"""
models/compare_feature_sets.py

Answers the open question from the June 30 session: does widening the
training window (2018-2022 vs 2020-2022) rescue the 6 zero-importance
features (full_throttle_pct, avg_corner_speed, lap_length_km, drs_zones,
is_street_circuit, wet_flag), or should they just be dropped?

Runs 4 configs back to back on the SAME train/test split logic as
race_model.py, so results are directly comparable:

    A. baseline        — 2020-2022, all 14 features   (current race_model.py)
    B. wide_window      — 2018-2022, all 14 features
    C. dropped_feats     — 2020-2022, 8 features (6 zero-importance dropped)
    D. wide_dropped     — 2018-2022, 8 features

For each config: model MAE, naive MAE, % improvement, and feature
importances. Logic is copied (not imported) from race_model.py and
parametrized by feature_cols / train_seasons, so this script doesn't
mutate or depend on race_model.py's module-level state.

Usage (from project root):
    python -m models.compare_feature_sets
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DIR, TEST_SEASONS

MASTER_FEATURES_PATH = os.path.join(PROCESSED_DIR, "master_features.parquet")

ALL_FEATURE_COLS = [
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

# The 6 features that sat at 0.0000 importance in the June 30 run
ZERO_IMPORTANCE_FEATS = [
    "full_throttle_pct",
    "avg_corner_speed",
    "lap_length_km",
    "drs_zones",
    "is_street_circuit",
    "wet_flag",
]

REDUCED_FEATURE_COLS = [c for c in ALL_FEATURE_COLS if c not in ZERO_IMPORTANCE_FEATS]

CONFIGS = [
    {"name": "A_baseline      (2020-2022, 14 feats)", "train_seasons": range(2020, 2023), "feature_cols": ALL_FEATURE_COLS},
    {"name": "B_wide_window   (2018-2022, 14 feats)", "train_seasons": range(2018, 2023), "feature_cols": ALL_FEATURE_COLS},
    {"name": "C_dropped_feats (2020-2022,  8 feats)", "train_seasons": range(2020, 2023), "feature_cols": REDUCED_FEATURE_COLS},
    {"name": "D_wide_dropped  (2018-2022,  8 feats)", "train_seasons": range(2018, 2023), "feature_cols": REDUCED_FEATURE_COLS},
]


# ---------------------------------------------------------------------------
# Core logic — same approach as race_model.py, parametrized by feature_cols
# ---------------------------------------------------------------------------

def build_pairwise(df: pd.DataFrame, feature_cols: list) -> tuple[np.ndarray, np.ndarray]:
    X_rows, y_rows = [], []
    for (season, rnd), race in df.groupby(["season", "round"]):
        drivers = race.reset_index(drop=True)
        feats = drivers[feature_cols].values
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


def predict_race_rankings(pipeline, race_df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    drivers = race_df.reset_index(drop=True)
    n = len(drivers)
    feats = drivers[feature_cols].values
    scores = np.zeros(n)

    for i, j in combinations(range(n), 2):
        diff = feats[i] - feats[j]
        prob = pipeline.predict_proba([diff])[0][1]
        scores[i] += prob
        scores[j] += (1 - prob)

    ranks = pd.Series(scores).rank(ascending=False, method="min")
    return ranks.values


def evaluate_on_test(pipeline, test_df: pd.DataFrame, feature_cols: list):
    all_pred_ranks, all_true_pos = [], []

    for (season, rnd), race in test_df.groupby(["season", "round"]):
        pred_ranks = predict_race_rankings(pipeline, race, feature_cols)
        all_pred_ranks.extend(pred_ranks)
        all_true_pos.extend(race["finishing_position"].values)

    model_mae = mean_absolute_error(all_true_pos, all_pred_ranks)

    naive_preds = test_df["grid_position"].values
    naive_mae = mean_absolute_error(test_df["finishing_position"].values, naive_preds)

    return model_mae, naive_mae


def train(X_train, y_train) -> Pipeline:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def get_importances(pipeline, feature_cols: list) -> list:
    model = pipeline.named_steps["model"]
    return sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Run a single config
# ---------------------------------------------------------------------------

def run_config(df: pd.DataFrame, cfg: dict) -> dict:
    train_seasons = cfg["train_seasons"]
    feature_cols = cfg["feature_cols"]

    train_df = df[df["season"].isin(train_seasons)].copy()
    test_df = df[df["season"].isin(TEST_SEASONS)].copy()

    X_train, y_train = build_pairwise(train_df, feature_cols)
    pipeline = train(X_train, y_train)
    model_mae, naive_mae = evaluate_on_test(pipeline, test_df, feature_cols)
    importances = get_importances(pipeline, feature_cols)

    improvement_pct = (naive_mae - model_mae) / naive_mae * 100

    return {
        "name": cfg["name"],
        "n_train_rows": len(train_df),
        "n_pairwise_rows": len(X_train),
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "improvement_pct": improvement_pct,
        "beats_naive": model_mae < naive_mae,
        "importances": importances,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nLoading master features...")
    df = pd.read_parquet(MASTER_FEATURES_PATH)
    print(f"  {len(df)} rows, seasons {df['season'].min()}-{df['season'].max()}")

    results = []
    for cfg in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Running {cfg['name']}")
        print(f"{'='*70}")
        result = run_config(df, cfg)
        results.append(result)
        print(f"  Train rows: {result['n_train_rows']} ({result['n_pairwise_rows']:,} pairwise)")
        print(f"  Model MAE:  {result['model_mae']:.3f}")
        print(f"  Naive MAE:  {result['naive_mae']:.3f}")
        if result["beats_naive"]:
            print(f"  ✓ Beats naive by {result['improvement_pct']:.1f}%")
        else:
            print(f"  ✗ Loses to naive by {-result['improvement_pct']:.1f}%")

    # --- Side-by-side summary table ---
    print(f"\n\n{'='*70}")
    print("SUMMARY — Model MAE vs Naive Baseline")
    print(f"{'='*70}")
    print(f"{'Config':<40} {'Model MAE':>10} {'Naive MAE':>10} {'vs Naive':>10}")
    print("-" * 72)
    for r in results:
        sign = "✓" if r["beats_naive"] else "✗"
        print(f"{r['name']:<40} {r['model_mae']:>10.3f} {r['naive_mae']:>10.3f} "
              f"{sign} {r['improvement_pct']:>+.1f}%")

    # --- Did the 6 dropped features ever earn non-zero importance? ---
    print(f"\n{'='*70}")
    print("ZERO-IMPORTANCE FEATURES — did widening the window help?")
    print(f"{'='*70}")
    for r in results:
        if r["name"].startswith(("A_", "B_")):  # only configs that include them
            print(f"\n  {r['name']}:")
            imp_dict = dict(r["importances"])
            for feat in ZERO_IMPORTANCE_FEATS:
                if feat in imp_dict:
                    val = imp_dict[feat]
                    flag = "" if val > 0.005 else "  (still ~zero)"
                    print(f"    {feat:<22} {val:.4f}{flag}")

    # --- Recommendation ---
    best = min(results, key=lambda r: r["model_mae"])
    print(f"\n{'='*70}")
    print(f"BEST CONFIG BY MAE: {best['name']}")
    print(f"  Model MAE {best['model_mae']:.3f} vs naive {best['naive_mae']:.3f} "
          f"({best['improvement_pct']:+.1f}%)")
    print(f"{'='*70}\n")
