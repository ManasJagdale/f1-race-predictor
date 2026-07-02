"""
feature_engineering/car_performance.py

Computes two constructor-level features for the F1 predictor:

  1. Pace delta       — how far off pole each constructor is on average
                        over the last 3 qualifying sessions (rolling).
                        Captures current car performance, implicitly
                        absorbs upgrade effects without needing upgrade data.

  2. DNF rate         — rolling mechanical reliability rate per constructor
                        over the last 2 seasons. Sampled in Monte Carlo
                        to determine whether a driver retires each run.

Both features are computed strictly from pre-race data — no leakage.

Usage:
    from feature_engineering.car_performance import build_car_performance
    car_df = build_car_performance(quali_df, race_df)
"""

import re
import pandas as pd
from collections import defaultdict


# ---------------------------------------------------------------------------
# Section 1: Qualifying pace delta
# ---------------------------------------------------------------------------
# For each race, we want to know: how far off pole is each constructor,
# on average over the last N qualifying sessions?
#
# Why qualifying and not race pace?
# - Qualifying is a clean 1-lap effort with no traffic, tyre strategy,
#   or safety car interference. It isolates raw car speed better than
#   race pace does.
# - Pole time is always available before the race starts (no leakage).
#
# How we compute it:
# 1. Parse Q times (Q3 → Q2 → Q1, whichever is available) to seconds
# 2. Find pole time for that race (minimum time across all drivers)
# 3. Each constructor's gap = best_constructor_time - pole_time
#    (0.0 for the pole-sitting constructor, positive for everyone else)
# 4. Roll this gap over the last PACE_WINDOW races per constructor
#
# Known limitation: sprint weekends sometimes have shorter qualifying.
# We just use whatever time is available — gap is still meaningful.

PACE_WINDOW = 3   # number of recent qualifying sessions to average


def _parse_time_to_seconds(time_str) -> float | None:
    """
    Convert a lap time string like '1:23.456' or '83.456' to seconds.
    Returns None if the string is missing or unparseable.
    """
    if not time_str or pd.isna(time_str):
        return None

    time_str = str(time_str).strip()

    # Format: M:SS.mmm
    match = re.match(r'^(\d+):(\d+\.\d+)$', time_str)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds

    # Format: SS.mmm (no minutes)
    try:
        return float(time_str)
    except ValueError:
        return None


def _best_constructor_time(group_for_constructor: pd.DataFrame) -> float | None:
    """
    Get the fastest qualifying time for a constructor in a given race.
    Tries Q3 first, falls back to Q2, then Q1.
    Returns the minimum (fastest) time across both drivers.
    """
    times = []
    for _, row in group_for_constructor.iterrows():
        for col in ["q3", "q2", "q1"]:
            t = _parse_time_to_seconds(row.get(col))
            if t is not None:
                times.append(t)
                break  # use best available session for this driver

    return min(times) if times else None


def compute_pace_delta(quali_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling qualifying pace delta per constructor per race.

    Args:
        quali_df: DataFrame from fetch_jolpica.load_qualifying(), columns:
            season, round, raceId, driverId, constructorId, q1, q2, q3

    Returns:
        DataFrame with columns:
            season, round, raceId, constructorId, pace_delta_vs_pole
        pace_delta_vs_pole = rolling avg gap to pole (seconds).
        0.0 = pole-sitting team. Larger = slower.
        NaN when no prior qualifying data exists for this constructor.
    """
    quali_df = quali_df.sort_values(["season", "round"]).reset_index(drop=True)

    records = []
    # {constructorId: [gap, gap, ...]} — raw gaps, most recent at end
    constructor_gaps: dict = defaultdict(list)

    for (season, rnd, race_id), race_group in quali_df.groupby(
        ["season", "round", "raceId"], sort=False
    ):
        # Find pole time: minimum best time across ALL constructors this race
        all_times = []
        constructor_best: dict = {}

        for constructor_id, con_group in race_group.groupby("constructorId"):
            best = _best_constructor_time(con_group)
            if best is not None:
                constructor_best[constructor_id] = best
                all_times.append(best)

        pole_time = min(all_times) if all_times else None

        # Record pre-race rolling delta for each constructor
        seen_constructors = race_group["constructorId"].unique()
        for constructor_id in seen_constructors:
            history = constructor_gaps[constructor_id]

            if history:
                window = history[-PACE_WINDOW:]
                rolling_delta = sum(window) / len(window)
            else:
                rolling_delta = float("nan")

            records.append({
                "season":              season,
                "round":               rnd,
                "raceId":              race_id,
                "constructorId":       constructor_id,
                "pace_delta_vs_pole":  rolling_delta,
            })

        # After recording, update gap history with this race's result
        if pole_time is not None:
            for constructor_id, best_time in constructor_best.items():
                gap = best_time - pole_time
                constructor_gaps[constructor_id].append(gap)

    pace_df = pd.DataFrame(records)
    print(f"  ✓ Pace delta: {len(pace_df)} constructor-race rows")
    return pace_df


# ---------------------------------------------------------------------------
# Section 2: Constructor DNF rate
# ---------------------------------------------------------------------------
# Mechanical reliability is a real signal — Red Bull's 2022 reliability
# issues vs their near-perfect 2023 directly affected race outcomes.
#
# We compute a rolling DNF rate per constructor over the last 2 seasons
# worth of races (not calendar years — just a sliding window of entries).
#
# What counts as a DNF?
# - status != 'Finished' AND not a lapped car ('+1 Lap', '+2 Laps', etc.)
# - 'Retired', 'Engine', 'Gearbox', 'Collision', 'Accident', etc. = DNF
# - Lapped but classified finishers are NOT DNFs
#
# We use a rolling window of the last DNF_WINDOW driver-race entries
# per constructor rather than a calendar-year window. This means a
# constructor with many drivers (e.g. customer engine teams) updates
# their reliability estimate faster.
#
# SHRINKAGE (added after the June 30 simulation run flagged this):
# A constructor with only a handful of historical entries (a brand-new
# team, or an early-season window before DNF_WINDOW fills up) can show
# wildly noisy DNF rates — e.g. 2 DNFs out of 3 races = 66.7%, even if
# the team is not actually that unreliable. Small-sample swings like
# this fed directly into simulation.py and produced DNF probabilities
# in the 40-70%+ range for several midfield/backmarker teams, which is
# implausibly high in aggregate.
#
# Fix: empirical Bayes shrinkage. Blend each constructor's raw rolling
# rate toward the dataset-wide rate, weighted by how much history that
# constructor actually has. A constructor with a full DNF_WINDOW of
# history is barely shrunk; a brand-new team with 2-3 entries is pulled
# heavily toward the global average until more data accumulates.
#
#   shrunk_rate = (n * raw_rate + K * global_rate) / (n + K)
#
# where n = constructor's available history (capped at DNF_WINDOW),
# K = SHRINKAGE_STRENGTH (effectively "K pseudo-observations of the
# global average"), and global_rate is the cumulative all-time DNF rate
# across ALL constructors observed so far (pre-race, no leakage — same
# principle as the per-constructor rolling rate).

DNF_WINDOW = 40   # last 40 driver-race entries ≈ ~1 season for a 2-driver team
SHRINKAGE_STRENGTH = 15  # pseudo-observations pulling toward the global rate


def _is_dnf(status: str) -> bool:
    """Return True if the status represents a DNF (not a classified finish)."""
    if status == "Finished":
        return False
    # Lapped cars: '+1 Lap', '+2 Laps', etc.
    if re.match(r'^\+\d+ Lap', status):
        return False
    return True


def compute_dnf_rate(race_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling, shrinkage-adjusted DNF rate per constructor per race.

    Args:
        race_df: DataFrame from fetch_jolpica.load_race_results(), columns:
            season, round, raceId, driverId, constructorId, status

    Returns:
        DataFrame with columns:
            season, round, raceId, constructorId, dnf_rate, dnf_rate_raw, n_history
        dnf_rate     = shrinkage-adjusted rate (use this — the model feature)
        dnf_rate_raw = unshrunk rolling rate, kept for diagnostics/debugging
        n_history    = how many historical entries informed this row,
                       capped at DNF_WINDOW (low n = more shrinkage applied)
        NaN only possible in dnf_rate_raw when a constructor has zero
        prior history; dnf_rate itself is never NaN (falls back fully to
        the global rate via the shrinkage formula).
    """
    race_df = race_df.sort_values(["season", "round"]).reset_index(drop=True)

    records = []
    # {constructorId: [0/1, 0/1, ...]} — 1 = DNF, 0 = finished
    constructor_dnf_history: dict = defaultdict(list)
    # All-time, all-constructor DNF history — pre-race cumulative, no leakage
    global_dnf_history: list = []

    for (season, rnd, race_id), race_group in race_df.groupby(
        ["season", "round", "raceId"], sort=False
    ):
        seen_constructors = race_group["constructorId"].unique()

        # Global rate BEFORE this race's results are folded in
        if global_dnf_history:
            global_rate = sum(global_dnf_history) / len(global_dnf_history)
        else:
            global_rate = 0.15  # reasonable generic prior before any data exists

        # Record pre-race rolling DNF rate per constructor, shrunk toward global
        for constructor_id in seen_constructors:
            history = constructor_dnf_history[constructor_id]
            window = history[-DNF_WINDOW:]
            n = len(window)

            if n > 0:
                raw_rate = sum(window) / n
            else:
                raw_rate = float("nan")

            # Empirical Bayes shrinkage toward the global rate.
            # n=0 → shrunk_rate == global_rate (full shrinkage).
            # n>=DNF_WINDOW → shrunk_rate ≈ raw_rate (minimal shrinkage).
            effective_raw = raw_rate if n > 0 else global_rate
            shrunk_rate = (n * effective_raw + SHRINKAGE_STRENGTH * global_rate) / (n + SHRINKAGE_STRENGTH)

            records.append({
                "season":        season,
                "round":         rnd,
                "raceId":        race_id,
                "constructorId": constructor_id,
                "dnf_rate":      shrunk_rate,
                "dnf_rate_raw":  raw_rate,
                "n_history":     n,
            })

        # After recording, update DNF history with this race's results
        for _, row in race_group.iterrows():
            dnf_flag = 1 if _is_dnf(row["status"]) else 0
            constructor_dnf_history[row["constructorId"]].append(dnf_flag)
            global_dnf_history.append(dnf_flag)

    dnf_df = pd.DataFrame(records)
    avg_shrinkage = (dnf_df["n_history"] < SHRINKAGE_STRENGTH).mean()
    print(f"  ✓ DNF rate: {len(dnf_df)} constructor-race rows "
          f"({avg_shrinkage:.1%} of rows had n_history < {SHRINKAGE_STRENGTH}, "
          f"meaningfully shrunk toward the global rate)")
    return dnf_df


# ---------------------------------------------------------------------------
# Section 3: Master builder
# ---------------------------------------------------------------------------

def build_car_performance(
    quali_df: pd.DataFrame,
    race_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all constructor-level features and return a merged DataFrame.

    Args:
        quali_df: qualifying results from load_qualifying()
        race_df:  race results from load_race_results()

    Returns:
        DataFrame with columns:
            season, round, raceId, constructorId,
            pace_delta_vs_pole, dnf_rate, dnf_rate_raw, n_history
    """
    print("  → Computing qualifying pace delta...")
    pace_df = compute_pace_delta(quali_df)

    print("  → Computing constructor DNF rate (shrinkage-adjusted)...")
    dnf_df = compute_dnf_rate(race_df)

    merge_keys = ["season", "round", "raceId", "constructorId"]
    car_df = pace_df.merge(dnf_df, on=merge_keys, how="outer")

    print(f"  ✓ Car performance ready: {len(car_df)} rows, "
          f"{car_df['constructorId'].nunique()} unique constructors")
    return car_df


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.fetch_jolpica import load_qualifying, load_race_results

    print("car_performance.py — sanity check with real cached data\n")

    quali_df = load_qualifying()
    race_df  = load_race_results()

    print("Building car performance features...")
    car_df = build_car_performance(quali_df, race_df)

    # Show constructor pace deltas going into 2026 Round 8
    last_season = car_df["season"].max()
    last_round  = car_df[car_df["season"] == last_season]["round"].max()

    final = (
        car_df[
            (car_df["season"] == last_season) &
            (car_df["round"]  == last_round)
        ]
        .sort_values("pace_delta_vs_pole")
        .reset_index(drop=True)
    )

    print(f"\n--- Constructor performance going into {last_season} Round {last_round} ---")
    print(f"{'Pos':<5} {'Constructor':<25} {'Pace gap (s)':>12} {'DNF rate':>10} {'Raw rate':>10} {'n_hist':>8}")
    print("-" * 80)
    for i, row in final.iterrows():
        pace = f"+{row['pace_delta_vs_pole']:.3f}s" if pd.notna(row['pace_delta_vs_pole']) else "   —"
        dnf  = f"{row['dnf_rate']:.1%}" if pd.notna(row['dnf_rate']) else "  —"
        raw  = f"{row['dnf_rate_raw']:.1%}" if pd.notna(row['dnf_rate_raw']) else "  —"
        nhist = f"{int(row['n_history'])}" if pd.notna(row['n_history']) else "—"
        print(f"{i+1:<5} {row['constructorId']:<25} {pace:>12} {dnf:>10} {raw:>10} {nhist:>8}")

    # Basic checks
    print("\nChecks:")
    # Pole-sitting constructor should have pace_delta near 0
    min_delta = final["pace_delta_vs_pole"].min()
    assert min_delta >= 0.0, f"Minimum pace delta should be >= 0, got {min_delta}"
    print(f"  ✓ Minimum pace delta is {min_delta:.4f}s (pole-sitter, expected ~0)")

    # DNF rates should all be between 0 and 1
    valid_dnf = final["dnf_rate"].dropna()
    assert (valid_dnf >= 0).all() and (valid_dnf <= 1).all(), "DNF rates out of range"
    print(f"  ✓ All DNF rates in valid range [0, 1]")

    # dnf_rate should never be NaN now (shrinkage always produces a value)
    assert final["dnf_rate"].isna().sum() == 0, "dnf_rate should never be NaN post-shrinkage"
    print(f"  ✓ No NaN in shrinkage-adjusted dnf_rate (was possible pre-fix)")

    print("\nAll checks passed ✓")
