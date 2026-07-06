"""
experiments/teammate_delta_rolling/diagnose_field_scores_v2.py

Same diagnostic as the real project's diagnose_field_scores.py, but
against the v2 model + v2 rolling teammate_delta computation. Re-run
this after train_v2.py to see whether Antonelli's total strength score
and rank moved relative to the field, compared to the original run
where he was rank 6/22 with a 9.6% score gap from the leader.

Reuses compute_strength_scores() DIRECTLY from the real project's
simulation.py -- unmodified. Only the feature-building and model-
loading steps come from this folder's own predict_upcoming_v2.py.

Usage:
    python diagnose_field_scores_v2.py --season 2026 --round 9 \
        --grid-csv ../../silverstone_2026_grid.csv --focus antonelli
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _THIS_DIR)

from predict_upcoming_v2 import resolve_circuit, get_lineup, build_prediction_row, _load_model_v2
from models.race_model import FEATURE_COLS
from simulation.simulation import compute_strength_scores


def full_field_scores(ensemble: list, race_df: pd.DataFrame) -> pd.DataFrame:
    scores = compute_strength_scores(ensemble, race_df)
    out = race_df[["driverId", "grid_position"]].copy().reset_index(drop=True)
    out["grid_position"] = out["grid_position"].astype(int)
    out["strength_score"] = scores
    out["rank_by_score"] = out["strength_score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("strength_score", ascending=False).reset_index(drop=True)


def per_opponent_breakdown(ensemble: list, race_df: pd.DataFrame, focus_driver: str) -> pd.DataFrame:
    drivers = race_df.reset_index(drop=True)
    n = len(drivers)
    feats = drivers[FEATURE_COLS].values
    driver_ids = drivers["driverId"].values

    if focus_driver not in driver_ids:
        raise ValueError(f"'{focus_driver}' not in this round's lineup: {sorted(driver_ids)}")

    focus_idx = list(driver_ids).index(focus_driver)
    results = []
    for j in range(n):
        if j == focus_idx:
            continue
        diff = feats[focus_idx] - feats[j]
        probs = [pipeline.predict_proba([diff])[0][1] for pipeline in ensemble]
        prob = float(np.mean(probs))
        results.append({
            "opponent": driver_ids[j],
            "grid_position": int(drivers.loc[j, "grid_position"]),
            "p_focus_beats_opponent": prob,
        })
    return pd.DataFrame(results).sort_values("grid_position").reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose v2 full-field strength scores")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_num")
    parser.add_argument("--grid-csv", type=str, default=None)
    parser.add_argument("--focus", type=str, required=True)
    parser.add_argument("--wet", action="store_true")
    args = parser.parse_args()

    print(f"Resolving round {args.round_num} of season {args.season}...")
    circuit_id, race_name = resolve_circuit(args.season, args.round_num)
    print(f"  {race_name} ({circuit_id})")

    print("\nDetermining grid / lineup...")
    lineup, source = get_lineup(args.season, args.round_num, args.grid_csv)
    print(f"  {len(lineup)} drivers, source: {source}")

    print("\n[v2] Building prediction features (rolling teammate_delta)...")
    race_df = build_prediction_row(args.season, args.round_num, circuit_id, lineup, args.wet)

    print("Loading v2 trained model ensemble...")
    ensemble = _load_model_v2()

    print("\nComputing full-field round-robin strength scores (v2)...")
    scores_df = full_field_scores(ensemble, race_df)
    max_score = scores_df["strength_score"].max()

    print(f"\n{'='*72}")
    print("[v2] FULL FIELD -- total strength score")
    print(f"{'='*72}")
    print(f"{'Rank':<6}{'Driver':<18}{'Grid':>6}{'Score':>10}   Relative to leader")
    print("-" * 72)
    for _, row in scores_df.iterrows():
        bar = "#" * int(row["strength_score"] / max_score * 30)
        print(f"{row['rank_by_score']:<6}{row['driverId']:<18}{row['grid_position']:>6}"
              f"{row['strength_score']:>10.3f}   {bar}")
    print(f"{'='*72}\n")

    focus_row = scores_df[scores_df["driverId"] == args.focus]
    if focus_row.empty:
        print(f"'{args.focus}' not found in this round's lineup.")
        sys.exit(1)

    focus_score = focus_row["strength_score"].values[0]
    focus_rank = int(focus_row["rank_by_score"].values[0])
    gap_pct = (max_score - focus_score) / max_score * 100

    print(f"[v2] {args.focus}: rank {focus_rank}/{len(scores_df)} by total score "
          f"({focus_score:.3f} vs field-leading {max_score:.3f} -- a {gap_pct:.1f}% gap)")
    print(f"\nCOMPARE AGAINST THE ORIGINAL RUN: antonelli was rank 6/22, 9.6% gap.")
    print(f"If the v2 gap above is smaller and/or the rank improved, the rolling")
    print(f"teammate_delta change moved his score in the expected direction.\n")

    breakdown = per_opponent_breakdown(ensemble, race_df, args.focus)
    print(f"{'='*72}")
    print(f"[v2] {args.focus.upper()} -- per-opponent win probability")
    print(f"{'='*72}")
    for _, row in breakdown.iterrows():
        print(f"  vs {row['opponent']:<18} (grid {row['grid_position']:>2}): "
              f"{row['p_focus_beats_opponent']:.1%}")
    print(f"{'='*72}\n")

    front_pack = breakdown[breakdown["grid_position"] <= 8]
    back_pack = breakdown[breakdown["grid_position"] > 8]
    print(f"[v2] Average P({args.focus} beats front-pack, grid 1-8):  "
          f"{front_pack['p_focus_beats_opponent'].mean():.1%}  (was 52.5% originally)")
    print(f"[v2] Average P({args.focus} beats back-pack, grid 9+):    "
          f"{back_pack['p_focus_beats_opponent'].mean():.1%}  (was 76.7% originally)")
