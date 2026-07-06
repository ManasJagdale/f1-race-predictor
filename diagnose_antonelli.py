"""
diagnose_antonelli.py

Diagnostic: prints the raw rolling features (elo_pre_race, recent_form,
teammate_delta, pace_delta_vs_pole, dnf_rate) computed by
predict_upcoming.py's placeholder-row technique for every driver in a
given round, then decomposes the pairwise strength-score comparison
between two specific drivers so you can see WHICH individual feature(s)
are actually pushing the model's prediction one way or the other.

WHY THIS EXISTS:
predict_upcoming.py and oracle.py only ever show you the FINAL P1-Pn
probabilities after 10,000 Monte Carlo runs — by that point the raw
per-feature signal that produced the underlying strength score is
completely obscured. This script stops one layer earlier: it shows the
actual elo_pre_race / recent_form / teammate_delta / grid_position
values the model saw, and the exact pairwise win probability the
ensemble assigned to one driver over another, per seed and averaged.

Built to investigate a specific anomaly: Antonelli started P1 (pole) at
Silverstone 2026 (round 9) but was predicted 6th-most-likely to win,
behind Russell (P4), Hamilton (P3), Verstappen (P7), Leclerc (P2), and
Hadjar (P5) — despite grid_position being ~70% of the model's average
feature importance. Feature importance is a POPULATION-AVERAGE
statistic across ~23,000 training pairs; it doesn't guarantee grid
dominates every individual pairwise comparison. This script surfaces
the actual numbers behind that one comparison instead of inferring it
from importance averages.

Reuses resolve_circuit(), get_lineup(), and build_prediction_row()
directly from predict_upcoming.py — not a reimplementation — so the
feature values shown here are guaranteed identical to what actually
drove your live prediction, not a separate approximation of it.

Usage (from project root, same conventions as predict_upcoming.py):
    python diagnose_antonelli.py --season 2026 --round 9 \
        --grid-csv silverstone_2026_grid.csv \
        --driver-a antonelli --driver-b russell

    # Any two drivers in the field can be compared this way, not just
    # this specific pair:
    python diagnose_antonelli.py --season 2026 --round 9 \
        --grid-csv silverstone_2026_grid.csv \
        --driver-a leclerc --driver-b hamilton
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict_upcoming import (
    resolve_circuit,
    get_lineup,
    build_prediction_row,
)
from models.race_model import FEATURE_COLS
from simulation.simulation import _load_model


# ---------------------------------------------------------------------------
# Section 1: Full-field feature dump
# ---------------------------------------------------------------------------

def print_full_field_features(race_df: pd.DataFrame) -> None:
    """
    Print every driver's raw rolling feature values, sorted by grid
    position. This is the same race_df that gets handed to
    run_monte_carlo() — nothing hidden or recomputed differently.
    """
    cols = ["driverId", "constructorId", "grid_position", "elo_pre_race",
            "recent_form", "teammate_delta", "pace_delta_vs_pole", "dnf_rate"]
    cols = [c for c in cols if c in race_df.columns]
    display = race_df[cols].sort_values("grid_position").reset_index(drop=True)

    print(f"\n{'='*95}")
    print("FULL FIELD — rolling features going into this race (sorted by grid position)")
    print(f"{'='*95}")
    print(display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"{'='*95}\n")


# ---------------------------------------------------------------------------
# Section 2: Pairwise decomposition between two named drivers
# ---------------------------------------------------------------------------

def decompose_pairwise(ensemble: list, race_df: pd.DataFrame,
                        driver_a: str, driver_b: str) -> float:
    """
    Show, feature by feature, the raw values and the diff (a - b) for
    two drivers, then the actual ensemble-averaged pairwise win
    probability the trained model assigns to driver_a beating driver_b.

    This is the EXACT same diff vector and prediction call used inside
    compute_strength_scores() in simulation.py — just surfaced for one
    pair instead of summed silently across the whole field.
    """
    row_a = race_df[race_df["driverId"] == driver_a]
    row_b = race_df[race_df["driverId"] == driver_b]

    if row_a.empty or row_b.empty:
        missing = driver_a if row_a.empty else driver_b
        available = sorted(race_df["driverId"].tolist())
        raise ValueError(
            f"Driver '{missing}' not found in this round's lineup.\n"
            f"Available drivers: {available}"
        )

    feats_a = row_a[FEATURE_COLS].values[0].astype(float)
    feats_b = row_b[FEATURE_COLS].values[0].astype(float)
    diff = feats_a - feats_b

    print(f"\n{'='*84}")
    print(f"FEATURE-BY-FEATURE: {driver_a} vs {driver_b}")
    print(f"{'='*84}")
    print(f"{'Feature':<25} {driver_a:>16} {driver_b:>16} {'diff (a - b)':>16}")
    print("-" * 84)
    for name, va, vb in zip(FEATURE_COLS, feats_a, feats_b):
        d = va - vb
        flag = ""
        if name == "elo_pre_race" and abs(d) > 50:
            flag = "  ← large Elo gap"
        if name == "grid_position" and d < 0:
            flag = "  ← a started ahead of b"
        print(f"{name:<25} {va:>16.3f} {vb:>16.3f} {d:>16.3f}{flag}")

    # Per-ensemble-member pairwise probability that driver_a beats driver_b —
    # identical computation to the inner loop of compute_strength_scores()
    # in simulation/simulation.py, just isolated to one pair.
    probs = [pipeline.predict_proba([diff])[0][1] for pipeline in ensemble]
    mean_prob = float(np.mean(probs))
    spread = float(np.std(probs))

    print(f"\n{'-'*84}")
    print(f"Per-seed P({driver_a} beats {driver_b}) — {len(ensemble)}-member ensemble:")
    for seed, p in enumerate(probs):
        print(f"  seed {seed}: {p:.1%}")
    print(f"\nEnsemble-averaged P({driver_a} beats {driver_b}): {mean_prob:.1%}  "
          f"(std across seeds: {spread:.1%})")
    print(f"{'='*84}\n")

    if mean_prob < 0.5:
        print(f"⚠ Despite whatever grid/feature advantage {driver_a} holds on paper,\n"
              f"  the trained model still favors {driver_b} in this specific head-to-head.\n"
              f"  Check which feature(s) above have the largest |diff| relative to their\n"
              f"  typical scale in the training data — a large elo_pre_race gap is a\n"
              f"  common culprit, since Elo is a much slower-moving, longer-history\n"
              f"  signal than a single-race grid position gap, and this project's own\n"
              f"  training data may contain a learned pattern like 'a big Elo deficit\n"
              f"  outweighs a modest grid advantage' from 2020-2022 races where that\n"
              f"  held true. Whether that pattern SHOULD generalize to a 2026 rookie\n"
              f"  on a genuine hot streak is exactly the open question this surfaces —\n"
              f"  not something this script can answer on its own.\n")
    else:
        print(f"✓ In this specific head-to-head, the model does favor {driver_a} — "
              f"the surprising\n  aggregate result must be coming from other drivers "
              f"in the field, not this pair.\n  Try comparing {driver_a} against "
              f"whichever driver had the single biggest jump\n  in the final "
              f"probability table instead.\n")

    return mean_prob


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decompose a specific pairwise prediction to diagnose a surprising result"
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_num")
    parser.add_argument("--grid-csv", type=str, default=None,
                         help="Same format as predict_upcoming.py's --grid-csv")
    parser.add_argument("--driver-a", type=str, required=True,
                         help="e.g. antonelli — the driver you expected to be favored")
    parser.add_argument("--driver-b", type=str, required=True,
                         help="e.g. russell — the driver the model actually favors more")
    parser.add_argument("--wet", action="store_true")
    args = parser.parse_args()

    print(f"Resolving round {args.round_num} of season {args.season}...")
    circuit_id, race_name = resolve_circuit(args.season, args.round_num)
    print(f"  {race_name} ({circuit_id})")

    print("\nDetermining grid / lineup...")
    lineup, source = get_lineup(args.season, args.round_num, args.grid_csv)
    print(f"  {len(lineup)} drivers, source: {source}")

    print("\nBuilding prediction features (identical to predict_upcoming.py)...")
    race_df = build_prediction_row(args.season, args.round_num, circuit_id, lineup, args.wet)

    print_full_field_features(race_df)

    print("Loading trained model ensemble...")
    ensemble = _load_model()

    decompose_pairwise(ensemble, race_df, args.driver_a, args.driver_b)
