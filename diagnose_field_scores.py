"""
diagnose_field_scores.py

Computes the FULL round-robin strength score for every driver in a
race (identical to compute_strength_scores() in simulation.py), ranks
them, and then breaks down ONE focus driver's score into its individual
per-opponent components -- so you can see whether a low final win
probability is coming from being weak against everyone, or from a
tightly bunched group at the front of the grid combined with how
Monte Carlo handles "who's the single best out of 22" draws.

WHY THIS EXISTS:
diagnose_antonelli.py showed Antonelli close to 50/50 against every
individual front-runner (41-55% depending on opponent) -- not a
blowout loss to any one of them. That's consistent with two very
different explanations:

  (a) His TOTAL summed score is close to the other front-runners'
      totals, and the huge gap in final P(Win) comes from Monte Carlo
      noise acting on a tightly clustered field -- an order-statistics
      effect ("winning requires beating everyone simultaneously, not
      just being competitive with each one"), not a modeling flaw.

  (b) His total score really IS meaningfully lower than the others,
      accumulated in small increments across many opponents, and the
      individual pairwise checks just happened not to reveal it.

This script settles which one it actually is by computing the real
total scores directly -- reuses compute_strength_scores() from
simulation.py exactly as run_monte_carlo() calls it, so nothing here
can drift from what actually drives the live prediction.

HOW TO READ THE OUTPUT:
  - If the focus driver's total score sits within a few percent of the
    field leader's, but their P(Win) in the final Monte Carlo output
    was drastically lower -- that confirms explanation (a). The fix,
    if you want one, is a simulation/aggregation change (e.g. reducing
    SCORE_NOISE_STD_FRAC, or accepting this as realistic -- being
    genuinely mid-pack among several near-equal rivals SHOULD produce
    a low single-race win probability even with a small score gap).
  - If the focus driver's total score is substantially below the
    leader's (not just a few percent), that confirms explanation (b)
    -- look at the per-opponent breakdown below the ranking to see
    which matchups are actually dragging the total down, and consider
    whether elo_pre_race's influence on those specific matchups needs
    revisiting (e.g. faster decay for young/hot-streak drivers).

Usage:
    python diagnose_field_scores.py --season 2026 --round 9 \
        --grid-csv silverstone_2026_grid.csv --focus antonelli
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict_upcoming import resolve_circuit, get_lineup, build_prediction_row
from models.race_model import FEATURE_COLS
from simulation.simulation import _load_model, compute_strength_scores


# ---------------------------------------------------------------------------
# Section 1: Full-field score ranking
# ---------------------------------------------------------------------------

def full_field_scores(ensemble: list, race_df: pd.DataFrame) -> pd.DataFrame:
    """
    Total round-robin strength score per driver -- IDENTICAL computation
    to compute_strength_scores() in simulation.py (imported directly,
    not reimplemented). Higher = stronger relative to the whole field.
    """
    scores = compute_strength_scores(ensemble, race_df)
    out = race_df[["driverId", "grid_position"]].copy().reset_index(drop=True)
    out["grid_position"] = out["grid_position"].astype(int)
    out["strength_score"] = scores
    out["rank_by_score"] = out["strength_score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("strength_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 2: Per-opponent decomposition for one focus driver
# ---------------------------------------------------------------------------

def per_opponent_breakdown(ensemble: list, race_df: pd.DataFrame, focus_driver: str) -> pd.DataFrame:
    """
    Same inner loop as compute_strength_scores(), but instead of only
    accumulating the total, records each individual pairwise win
    probability the focus driver has against every other driver in
    the field -- so the SHAPE of their score is visible, not just the
    final sum.
    """
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose full-field strength scores")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_num")
    parser.add_argument("--grid-csv", type=str, default=None)
    parser.add_argument("--focus", type=str, required=True,
                         help="Driver whose score you want to decompose, e.g. antonelli")
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

    print("Loading trained model ensemble...")
    ensemble = _load_model()

    print("\nComputing full-field round-robin strength scores "
          "(identical to simulation.py's compute_strength_scores)...")
    scores_df = full_field_scores(ensemble, race_df)

    max_score = scores_df["strength_score"].max()

    print(f"\n{'='*72}")
    print("FULL FIELD -- total strength score (sum of pairwise win-probs vs all 21 others)")
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
    gap_from_top = max_score - focus_score
    gap_pct = gap_from_top / max_score * 100

    print(f"{args.focus}: rank {focus_rank}/{len(scores_df)} by total score "
          f"({focus_score:.3f} vs field-leading {max_score:.3f} -- a {gap_pct:.1f}% gap)\n")

    if gap_pct < 5:
        print("  -> Total score gap is SMALL (<5%). If the final Monte Carlo P(Win) gap\n"
              "     was much larger than this, that's the order-statistics effect described\n"
              "     in the module docstring -- explanation (a), not a modeling flaw.\n")
    else:
        print("  -> Total score gap is NOT small. This driver's score is meaningfully\n"
              "     below the leader's before Monte Carlo noise is even applied --\n"
              "     explanation (b). See the per-opponent breakdown below for which\n"
              "     specific matchups are driving it.\n")

    print("Decomposing per-opponent win probabilities...")
    breakdown = per_opponent_breakdown(ensemble, race_df, args.focus)

    print(f"\n{'='*72}")
    print(f"{args.focus.upper()} -- per-opponent win probability (sorted by opponent's grid position)")
    print(f"{'='*72}")
    for _, row in breakdown.iterrows():
        print(f"  vs {row['opponent']:<18} (grid {row['grid_position']:>2}): "
              f"{row['p_focus_beats_opponent']:.1%}")
    print(f"{'='*72}\n")

    front_pack = breakdown[breakdown["grid_position"] <= 8]
    back_pack = breakdown[breakdown["grid_position"] > 8]
    print(f"Average P({args.focus} beats front-pack, grid 1-8):  "
          f"{front_pack['p_focus_beats_opponent'].mean():.1%}  "
          f"(n={len(front_pack)})")
    print(f"Average P({args.focus} beats back-pack, grid 9+):    "
          f"{back_pack['p_focus_beats_opponent'].mean():.1%}  "
          f"(n={len(back_pack)})")
    print()
