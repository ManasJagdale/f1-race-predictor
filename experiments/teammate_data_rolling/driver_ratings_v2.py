"""
experiments/teammate_delta_rolling/driver_ratings_v2.py

EXPERIMENTAL VARIANT of feature_engineering/driver_ratings.py.

ONLY CHANGE FROM THE ORIGINAL: compute_teammate_delta() now averages
over a ROLLING WINDOW of a driver's last TEAMMATE_DELTA_WINDOW races
against their teammate, instead of an unweighted average over their
ENTIRE career. Everything else (Elo, recent form) is byte-for-byte
identical to the original.

WHY THIS EXISTS:
Diagnostic work (diagnose_field_scores.py) on the Silverstone 2026
prediction found Antonelli genuinely mid-pack among the front-runners
by total strength score (9.6% below the leader, not just Monte Carlo
noise), despite having the 2nd-best recent_form in the field. The root
cause traced to compute_teammate_delta() in the ORIGINAL file:

    if history:
        rolling_delta = sum(history) / len(history)   # <- career average

Unlike elo_pre_race (0.97/month decay) and recent_form (hard 5-race
window, FORM_WINDOW), teammate_delta has NO recency mechanism at all.
For a driver early in their career, one number has to represent their
entire history, so a strong recent stretch can't move it much relative
to how it's anchored by early-career races. This is the one driver-
skill feature in the project that doesn't already have some form of
"weight recent performance more" built in.

THIS SCRIPT DOES NOT MODIFY driver_ratings.py. It's a separate copy
used only by the other _v2 scripts in this folder, so the change can
be A/B tested against the real backtest before deciding whether to
port it into the actual project.

Usage: not run directly -- imported by build_master_features_v2.py
"""

import datetime
import pandas as pd
from collections import defaultdict

import sys
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from config import ELO_INITIAL, ELO_K_BASE, FORM_WINDOW


# ---------------------------------------------------------------------------
# EXPERIMENTAL CONSTANT -- the only new knob this variant introduces.
# ---------------------------------------------------------------------------
# How many of a driver's most recent teammate head-to-heads to average,
# instead of their entire career. One teammate comparison happens per
# race (same cadence as recent_form's per-race finishing position), so
# this is directly comparable to FORM_WINDOW (currently 5) -- start
# here, but try 5 (exact parity with recent_form) or a larger value
# (e.g. 12) as further A/B variants if this first pass looks promising.
TEAMMATE_DELTA_WINDOW = 8


# ---------------------------------------------------------------------------
# Section 1: Elo rating engine -- UNCHANGED from the original file
# ---------------------------------------------------------------------------

ELO_DECAY_PER_MONTH = 0.97
ELO_SEASON_RESET = 0.0

_TODAY = datetime.date.today()


def _months_ago(season: int, round_num: int) -> float:
    approx_month = 3 + (round_num - 1) * 8 / 23
    approx_month = min(11, max(3, approx_month))
    race_date = datetime.date(season, int(approx_month), 15)
    days_ago = (_TODAY - race_date).days
    return max(0.0, days_ago / 30.44)


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _update_elo_after_race(ratings: dict, finishing_order: list, dnf_set: set,
                            k: float = ELO_K_BASE) -> None:
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
    results_df = results_df.sort_values(["season", "round"]).reset_index(drop=True)
    ratings = {}
    records = []

    for (season, rnd, race_id), group in results_df.groupby(
        ["season", "round", "raceId"], sort=False
    ):
        for driver_id in group["driverId"]:
            records.append({
                "season": season, "round": rnd, "raceId": race_id,
                "driverId": driver_id,
                "elo_pre_race": ratings.get(driver_id, ELO_INITIAL),
            })

        months = _months_ago(season, rnd)
        k_decayed = ELO_K_BASE * (ELO_DECAY_PER_MONTH ** months)

        group_sorted = group.sort_values("positionOrder")
        finishing_order = group_sorted["driverId"].tolist()
        dnf_set = set(group_sorted.loc[group_sorted["status"] != "Finished", "driverId"])

        _update_elo_after_race(ratings, finishing_order, dnf_set, k=k_decayed)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Section 2: Recent form -- UNCHANGED from the original file
# ---------------------------------------------------------------------------

def compute_recent_form(results_df: pd.DataFrame) -> pd.DataFrame:
    results_df = results_df.sort_values(["season", "round"]).reset_index(drop=True)
    max_pos = results_df["positionOrder"].max()
    dnf_penalty = max_pos + 1
    results_df = results_df.copy()
    results_df["pos_filled"] = results_df["position"].fillna(dnf_penalty)

    records = []
    driver_history: dict = defaultdict(list)

    for (season, rnd, race_id), group in results_df.groupby(
        ["season", "round", "raceId"], sort=False
    ):
        for _, row in group.iterrows():
            driver_id = row["driverId"]
            history = driver_history[driver_id]
            if history:
                recent_form = sum(history[-FORM_WINDOW:]) / len(history[-FORM_WINDOW:])
            else:
                recent_form = float("nan")
            records.append({
                "season": season, "round": rnd, "raceId": race_id,
                "driverId": driver_id, "recent_form": recent_form,
            })

        for _, row in group.iterrows():
            driver_history[row["driverId"]].append(row["pos_filled"])

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Section 3: Teammate delta -- *** THE MODIFIED FUNCTION ***
# ---------------------------------------------------------------------------

def compute_teammate_delta(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    EXPERIMENTAL VERSION. Only change from the original: the rolling
    average is now taken over the last TEAMMATE_DELTA_WINDOW races
    against the teammate, not the driver's entire career. This mirrors
    exactly how compute_recent_form() above already works (note the
    `history[-TEAMMATE_DELTA_WINDOW:]` slice, identical pattern to
    `history[-FORM_WINDOW:]`).

    Everything else -- the delta definition itself (own_pos - teammate_pos,
    negative = beat teammate = good), the pre-race-only computation, the
    NaN-for-no-history behavior -- is unchanged from the original.
    """
    results_df = results_df.sort_values(["season", "round"]).reset_index(drop=True)

    max_pos = results_df["positionOrder"].max()
    dnf_penalty = max_pos + 1
    df = results_df.copy()
    df["pos_filled"] = df["position"].fillna(dnf_penalty)

    records = []
    driver_deltas: dict = defaultdict(list)

    for (season, rnd, race_id), group in df.groupby(
        ["season", "round", "raceId"], sort=False
    ):
        constructor_drivers: dict = defaultdict(list)
        for _, row in group.iterrows():
            constructor_drivers[row["constructorId"]].append(row["driverId"])

        for _, row in group.iterrows():
            driver_id = row["driverId"]
            history = driver_deltas[driver_id]

            # --- THE ONLY LINE THAT DIFFERS FROM THE ORIGINAL ---
            # Original:  rolling_delta = sum(history) / len(history)
            # Here:      only the last TEAMMATE_DELTA_WINDOW observations
            #            count, same recency principle as recent_form.
            if history:
                window = history[-TEAMMATE_DELTA_WINDOW:]
                rolling_delta = sum(window) / len(window)
            else:
                rolling_delta = float("nan")

            records.append({
                "season": season, "round": rnd, "raceId": race_id,
                "driverId": driver_id, "teammate_delta": rolling_delta,
            })

        pos_map = {row["driverId"]: row["pos_filled"] for _, row in group.iterrows()}

        for constructor_id, drivers in constructor_drivers.items():
            if len(drivers) != 2:
                continue
            d1, d2 = drivers
            p1 = pos_map[d1]
            p2 = pos_map[d2]
            driver_deltas[d1].append(p1 - p2)
            driver_deltas[d2].append(p2 - p1)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Section 4: Master builder -- UNCHANGED from the original file
# ---------------------------------------------------------------------------

def build_driver_ratings(results_df: pd.DataFrame) -> pd.DataFrame:
    print("  → [v2] Computing Elo ratings (unchanged logic)...")
    elo_df = compute_elo_history(results_df)

    print("  → [v2] Computing recent form (unchanged logic)...")
    form_df = compute_recent_form(results_df)

    print(f"  → [v2] Computing teammate deltas (ROLLING window={TEAMMATE_DELTA_WINDOW}, "
          f"was: unweighted career average)...")
    delta_df = compute_teammate_delta(results_df)

    merge_keys = ["season", "round", "raceId", "driverId"]
    driver_ratings = (
        elo_df
        .merge(form_df[merge_keys + ["recent_form"]], on=merge_keys, how="left")
        .merge(delta_df[merge_keys + ["teammate_delta"]], on=merge_keys, how="left")
    )

    print(f"  ✓ [v2] Driver ratings ready: {len(driver_ratings)} rows, "
          f"{driver_ratings['driverId'].nunique()} unique drivers")

    return driver_ratings


# ---------------------------------------------------------------------------
# Quick sanity check -- identical assertions to the original file's
# synthetic 3-race test. Since TEAMMATE_DELTA_WINDOW (8) is larger than
# the 3 races in this synthetic dataset, every driver's window slice
# equals their full history here -- so this test should pass IDENTICALLY
# to the original, confirming the change is a no-op for short histories
# and only diverges once a driver has more races than the window size.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("driver_ratings_v2.py -- sanity check (should match original exactly, "
          "since 3 races < window size)\n")

    synthetic = pd.DataFrame([
        {"season": 2023, "round": 1, "raceId": 1, "driverId": "A", "constructorId": "C1", "position": 1.0, "positionOrder": 1, "status": "Finished"},
        {"season": 2023, "round": 1, "raceId": 1, "driverId": "B", "constructorId": "C1", "position": 2.0, "positionOrder": 2, "status": "Finished"},
        {"season": 2023, "round": 1, "raceId": 1, "driverId": "C", "constructorId": "C2", "position": 3.0, "positionOrder": 3, "status": "Finished"},
        {"season": 2023, "round": 1, "raceId": 1, "driverId": "D", "constructorId": "C2", "position": None, "positionOrder": 4, "status": "Engine"},
        {"season": 2023, "round": 2, "raceId": 2, "driverId": "A", "constructorId": "C1", "position": 2.0, "positionOrder": 2, "status": "Finished"},
        {"season": 2023, "round": 2, "raceId": 2, "driverId": "B", "constructorId": "C1", "position": 1.0, "positionOrder": 1, "status": "Finished"},
        {"season": 2023, "round": 2, "raceId": 2, "driverId": "C", "constructorId": "C2", "position": 4.0, "positionOrder": 4, "status": "Finished"},
        {"season": 2023, "round": 2, "raceId": 2, "driverId": "D", "constructorId": "C2", "position": 3.0, "positionOrder": 3, "status": "Finished"},
    ])

    ratings = build_driver_ratings(synthetic)
    r2_delta = ratings[ratings["round"] == 2].set_index("driverId")["teammate_delta"]
    assert r2_delta["A"] == -1.0, f"Expected -1.0, got {r2_delta['A']}"
    assert r2_delta["B"] == 1.0, f"Expected 1.0, got {r2_delta['B']}"
    print("  ✓ Matches original behavior for short histories (window not yet binding)")
    print("\nAll checks passed ✓")
