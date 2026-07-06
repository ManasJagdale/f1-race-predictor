"""
feature_engineering/driver_ratings.py

Computes three driver-level features for the F1 predictor:

  1. Elo rating       — long-term driver quality, updated after every race
  2. Recent form      — average finishing position over the last N races
  3. Teammate delta   — gap to same-car teammate at the same circuit,
                        averaged over multiple events (driver skill proxy)

All features use only data that is available BEFORE the target race starts.
No post-race information leaks into the feature set.

Usage:
    from feature_engineering.driver_ratings import build_driver_ratings
    ratings = build_driver_ratings(results_df)
    # ratings is a dict keyed by (season, round, driverId) -> feature dict
"""

import os
import sys
import math
import datetime
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ELO_INITIAL, ELO_K_BASE, FORM_WINDOW


# ---------------------------------------------------------------------------
# Section 1: Elo rating engine (steep recency decay + season reset)
# ---------------------------------------------------------------------------
# Elo works by awarding points after each head-to-head comparison.
# In F1, every driver races against every other driver simultaneously,
# so we decompose into pairwise comparisons: for every pair (A, B) in
# the same race, A "beat" B if A finished ahead.
#
# TWO recency mechanisms work together:
#
# 1. STEEP K DECAY (0.90 per month):
#    Older races contribute much less to ratings than recent ones.
#    - 3 months ago  → 73% weight   (last season's final races)
#    - 6 months ago  → 53% weight
#    - 1 year ago    → 28% weight
#    - 2 years ago   → 8%  weight
#    This means 8 races of P15-21 in 2026 will genuinely tank a rating,
#    not get buffered by 3 years of historical dominance.
#
# 2. SEASON RESET (30% pull toward 1500 between seasons):
#    At the start of each new season, every driver's rating is pulled
#    30% back toward ELO_INITIAL. New car, new regulations, fresh slate.
#    A driver at 1800 becomes: 1800 - 0.3*(1800-1500) = 1710
#    A driver at 1200 becomes: 1200 - 0.3*(1200-1500) = 1290
#    This prevents historical dominance from being an immovable anchor
#    and reflects the genuine uncertainty of a new season.
#    The 2026 regulation reset makes this especially justified.

ELO_DECAY_PER_MONTH = 0.97
ELO_SEASON_RESET    = 0.0    # disabled

_TODAY = datetime.date.today()


def _months_ago(season: int, round_num: int) -> float:
    """
    Estimate how many months ago a race was, given its season and round.
    Approximates race dates as evenly spaced March–November per season.
    """
    approx_month = 3 + (round_num - 1) * 8 / 23
    approx_month = min(11, max(3, approx_month))
    race_date = datetime.date(season, int(approx_month), 15)
    days_ago = (_TODAY - race_date).days
    return max(0.0, days_ago / 30.44)


def _apply_season_reset(ratings: dict) -> None:
    """
    Pull every driver's rating 30% back toward ELO_INITIAL.
    Called once at the start of each new season. Mutates ratings in place.
    """
    for driver in ratings:
        ratings[driver] = ratings[driver] + ELO_SEASON_RESET * (ELO_INITIAL - ratings[driver])


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score for player A vs player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _update_elo_after_race(
    ratings: dict,
    finishing_order: list,
    dnf_set: set,
    k: float = ELO_K_BASE,
) -> None:
    """
    Update Elo ratings for all drivers after one race via pairwise comparisons.
    k is already decay-adjusted before this is called.
    """
    n = len(finishing_order)
    deltas = defaultdict(float)

    for i in range(n):
        for j in range(i + 1, n):
            a = finishing_order[i]
            b = finishing_order[j]

            a_dnf = a in dnf_set
            b_dnf = b in dnf_set

            if a_dnf and b_dnf:
                continue

            if a_dnf:
                score_a, score_b = 0.0, 1.0
            else:
                score_a, score_b = 1.0, 0.0

            ra = ratings.get(a, ELO_INITIAL)
            rb = ratings.get(b, ELO_INITIAL)

            ea = _expected_score(ra, rb)
            eb = _expected_score(rb, ra)

            k_scaled = k / (n - 1)

            deltas[a] += k_scaled * (score_a - ea)
            deltas[b] += k_scaled * (score_b - eb)

    for driver, delta in deltas.items():
        ratings[driver] = ratings.get(driver, ELO_INITIAL) + delta


def compute_elo_history(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replay every race chronologically, computing pre-race Elo ratings
    with steep decay and season resets.

    Returns DataFrame with: season, round, raceId, driverId, elo_pre_race
    All values are pre-race — no leakage.
    """
    results_df = results_df.sort_values(["season", "round"]).reset_index(drop=True)

    ratings = {}
    records = []
    current_season = None

    for (season, rnd, race_id), group in results_df.groupby(
        ["season", "round", "raceId"], sort=False
    ):
        if season != current_season:
            current_season = season

        # Snapshot pre-race ratings
        for driver_id in group["driverId"]:
            records.append({
                "season":      season,
                "round":       rnd,
                "raceId":      race_id,
                "driverId":    driver_id,
                "elo_pre_race": ratings.get(driver_id, ELO_INITIAL),
            })

        # Decay-adjusted K — recent races hit harder
        months = _months_ago(season, rnd)
        k_decayed = ELO_K_BASE * (ELO_DECAY_PER_MONTH ** months)

        group_sorted = group.sort_values("positionOrder")
        finishing_order = group_sorted["driverId"].tolist()
        dnf_set = set(
            group_sorted.loc[group_sorted["status"] != "Finished", "driverId"]
        )

        _update_elo_after_race(ratings, finishing_order, dnf_set, k=k_decayed)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Section 2: Recent form
# ---------------------------------------------------------------------------
# Simple rolling average of finishing positions over the last FORM_WINDOW races.
# Lower = better (P1 is better than P10).
# DNFs are scored as (max_position + 1) to penalise without being infinite.
# Drivers with fewer than FORM_WINDOW races get an average of whatever they have.

def compute_recent_form(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each driver's average finishing position over the last
    FORM_WINDOW races, measured BEFORE each race (no leakage).

    Args:
        results_df: same schema as above. 'position' is the numeric
                    finishing position (NaN for DNF).

    Returns:
        DataFrame with columns:
            season, round, raceId, driverId, recent_form
        recent_form = avg finishing pos over last FORM_WINDOW races.
        NaN when a driver has no prior races in the dataset.
    """
    results_df = results_df.sort_values(["season", "round"]).reset_index(drop=True)

    # DNF penalty: use one position worse than the largest field
    max_pos = results_df["positionOrder"].max()
    dnf_penalty = max_pos + 1

    # Fill NaN positions (DNFs) with penalty score
    results_df = results_df.copy()
    results_df["pos_filled"] = results_df["position"].fillna(dnf_penalty)

    records = []
    # Track each driver's last N results as we move forward in time
    driver_history: dict = defaultdict(list)  # {driverId: [pos, pos, ...]}

    for (season, rnd, race_id), group in results_df.groupby(
        ["season", "round", "raceId"], sort=False
    ):
        for _, row in group.iterrows():
            driver_id = row["driverId"]
            history = driver_history[driver_id]

            if history:
                recent_form = sum(history[-FORM_WINDOW:]) / len(history[-FORM_WINDOW:])
            else:
                recent_form = float("nan")  # no prior data

            records.append({
                "season": season,
                "round": rnd,
                "raceId": race_id,
                "driverId": driver_id,
                "recent_form": recent_form,
            })

        # After recording pre-race form, update histories with this race's result
        for _, row in group.iterrows():
            driver_history[row["driverId"]].append(row["pos_filled"])

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Section 3: Teammate delta
# ---------------------------------------------------------------------------
# The core insight: two drivers in the same car at the same circuit.
# Any gap between them is attributable to driver skill, not the car.
#
# We compute: teammate_delta = driver_position - teammate_position
# (negative = driver finished AHEAD of teammate = good)
#
# We then compute a rolling average of this delta over each driver's last
# FORM_WINDOW head-to-head comparisons (same window as recent_form, added
# after the Antonelli investigation showed the previous unweighted
# full-career average permanently anchored old performance regardless of
# recent form — see Final_Project_History.md, Session 3, section 7.7).
#
# Known limitation (from spec): doesn't work cleanly when teammates receive
# unequal machinery (e.g. Red Bull preferential treatment for Verstappen).
# This is a Phase 2 concern — mixed effects model addresses it properly.

def compute_teammate_delta(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each driver's rolling average finishing position delta vs
    their same-team teammate at each race (pre-race value, no leakage).

    Args:
        results_df: schema as above, plus 'constructorId' column.

    Returns:
        DataFrame with columns:
            season, round, raceId, driverId, teammate_delta
        teammate_delta = rolling mean of (own_pos - teammate_pos) over the
        driver's last FORM_WINDOW comparisons (same window as recent_form).
        NaN when no prior teammate comparison exists.
    """
    results_df = results_df.sort_values(["season", "round"]).reset_index(drop=True)

    max_pos = results_df["positionOrder"].max()
    dnf_penalty = max_pos + 1
    df = results_df.copy()
    df["pos_filled"] = df["position"].fillna(dnf_penalty)

    records = []
    # Track raw delta observations per driver
    driver_deltas: dict = defaultdict(list)  # {driverId: [delta, delta, ...]}

    for (season, rnd, race_id), group in df.groupby(
        ["season", "round", "raceId"], sort=False
    ):
        # Build constructor → [driver, ...] lookup for this race
        constructor_drivers: dict = defaultdict(list)
        for _, row in group.iterrows():
            constructor_drivers[row["constructorId"]].append(row["driverId"])

        # Record pre-race teammate delta for each driver
        for _, row in group.iterrows():
            driver_id = row["driverId"]
            history = driver_deltas[driver_id]

            if history:
                window = history[-FORM_WINDOW:]
                rolling_delta = sum(window) / len(window)
            else:
                rolling_delta = float("nan")

            records.append({
                "season": season,
                "round": rnd,
                "raceId": race_id,
                "driverId": driver_id,
                "teammate_delta": rolling_delta,
            })

        # After recording, update delta history for this race
        pos_map = {row["driverId"]: row["pos_filled"] for _, row in group.iterrows()}

        for constructor_id, drivers in constructor_drivers.items():
            if len(drivers) != 2:
                # Only compute delta for standard 2-driver teams
                # (3-driver races or solo entries don't give clean comparisons)
                continue

            d1, d2 = drivers
            p1 = pos_map[d1]
            p2 = pos_map[d2]

            # delta < 0 means this driver finished ahead — better performance
            driver_deltas[d1].append(p1 - p2)
            driver_deltas[d2].append(p2 - p1)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Section 4: Master builder — combine all three features
# ---------------------------------------------------------------------------

def build_driver_ratings(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Computes all driver-level features and returns
    a single merged DataFrame ready for joining to the full feature table.

    Args:
        results_df: DataFrame with columns:
            season, round, raceId, driverId, constructorId,
            position (float, NaN for DNF), positionOrder (int), status (str)

    Returns:
        DataFrame with columns:
            season, round, raceId, driverId,
            elo_pre_race, recent_form, teammate_delta

    All values are computed from data strictly prior to each race — safe
    for use as ML features without temporal leakage.
    """
    print("  → Computing Elo ratings...")
    elo_df = compute_elo_history(results_df)

    print("  → Computing recent form...")
    form_df = compute_recent_form(results_df)

    print("  → Computing teammate deltas...")
    delta_df = compute_teammate_delta(results_df)

    # Merge on the four key columns
    merge_keys = ["season", "round", "raceId", "driverId"]

    driver_ratings = (
        elo_df
        .merge(form_df[merge_keys + ["recent_form"]], on=merge_keys, how="left")
        .merge(delta_df[merge_keys + ["teammate_delta"]], on=merge_keys, how="left")
    )

    print(f"  ✓ Driver ratings ready: {len(driver_ratings)} rows, "
          f"{driver_ratings['driverId'].nunique()} unique drivers")

    return driver_ratings


# ---------------------------------------------------------------------------
# Quick sanity check (run this file directly to test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("driver_ratings.py — sanity check with synthetic data\n")

    # Minimal synthetic race results: 3 races, 4 drivers, 2 constructors
    # Driver A and B are in Constructor 1; C and D are in Constructor 2
    synthetic = pd.DataFrame([
        # Race 1
        {"season": 2023, "round": 1, "raceId": 1, "driverId": "A", "constructorId": "C1", "position": 1.0, "positionOrder": 1, "status": "Finished"},
        {"season": 2023, "round": 1, "raceId": 1, "driverId": "B", "constructorId": "C1", "position": 2.0, "positionOrder": 2, "status": "Finished"},
        {"season": 2023, "round": 1, "raceId": 1, "driverId": "C", "constructorId": "C2", "position": 3.0, "positionOrder": 3, "status": "Finished"},
        {"season": 2023, "round": 1, "raceId": 1, "driverId": "D", "constructorId": "C2", "position":  None, "positionOrder": 4, "status": "Engine"},
        # Race 2
        {"season": 2023, "round": 2, "raceId": 2, "driverId": "A", "constructorId": "C1", "position": 2.0, "positionOrder": 2, "status": "Finished"},
        {"season": 2023, "round": 2, "raceId": 2, "driverId": "B", "constructorId": "C1", "position": 1.0, "positionOrder": 1, "status": "Finished"},
        {"season": 2023, "round": 2, "raceId": 2, "driverId": "C", "constructorId": "C2", "position": 4.0, "positionOrder": 4, "status": "Finished"},
        {"season": 2023, "round": 2, "raceId": 2, "driverId": "D", "constructorId": "C2", "position": 3.0, "positionOrder": 3, "status": "Finished"},
        # Race 3
        {"season": 2023, "round": 3, "raceId": 3, "driverId": "A", "constructorId": "C1", "position": 1.0, "positionOrder": 1, "status": "Finished"},
        {"season": 2023, "round": 3, "raceId": 3, "driverId": "B", "constructorId": "C1", "position": 3.0, "positionOrder": 3, "status": "Finished"},
        {"season": 2023, "round": 3, "raceId": 3, "driverId": "C", "constructorId": "C2", "position": 2.0, "positionOrder": 2, "status": "Finished"},
        {"season": 2023, "round": 3, "raceId": 3, "driverId": "D", "constructorId": "C2", "position": 4.0, "positionOrder": 4, "status": "Finished"},
    ])

    ratings = build_driver_ratings(synthetic)
    print("\nOutput (first 8 rows):")
    print(ratings.to_string(index=False))

    print("\nChecks:")
    # All drivers should start Race 1 with ELO_INITIAL (1500)
    r1 = ratings[ratings["round"] == 1]
    assert (r1["elo_pre_race"] == ELO_INITIAL).all(), "Race 1 Elo should all be initial"
    print("  ✓ Race 1: all Elo values are ELO_INITIAL")

    # After Race 1, A beat everyone → should have highest Elo in Race 2
    r2 = ratings[ratings["round"] == 2]
    elo_r2 = r2.set_index("driverId")["elo_pre_race"]
    assert elo_r2["A"] > elo_r2["B"] > elo_r2["C"], "Elo order wrong after Race 1"
    print("  ✓ Race 2: Elo order A > B > C (D had DNF)")

    # Race 1 form: all NaN (no prior races)
    assert r1["recent_form"].isna().all(), "Race 1 form should be NaN"
    print("  ✓ Race 1: recent_form is NaN for all drivers (no prior races)")

    # Race 2 form: each driver's race 1 position
    r2_form = r2.set_index("driverId")["recent_form"]
    assert r2_form["A"] == 1.0
    assert r2_form["B"] == 2.0
    print("  ✓ Race 2: recent_form matches Race 1 positions")

    # Race 1 teammate delta: all NaN
    assert r1["teammate_delta"].isna().all(), "Race 1 teammate delta should be NaN"
    print("  ✓ Race 1: teammate_delta is NaN for all drivers (no prior races)")

    # After Race 1: A beat B by 1 position (1 - 2 = -1), so A's delta should be -1
    r2_delta = r2.set_index("driverId")["teammate_delta"]
    assert r2_delta["A"] == -1.0, f"Expected -1.0, got {r2_delta['A']}"
    assert r2_delta["B"] == 1.0,  f"Expected 1.0, got {r2_delta['B']}"
    print("  ✓ Race 2: teammate_delta: A=-1 (beat teammate), B=+1 (lost to teammate)")

    print("\nAll checks passed ✓")