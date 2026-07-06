"""
feature_engineering/driver_track_affinity.py

Computes a per-driver, per-circuit affinity feature: does this driver
systematically outperform their expected finish specifically at THIS
circuit, beyond what their current form (Elo) and car pace
(pace_delta_vs_pole) would predict?

This is a RESIDUAL feature, not a raw average finish position per circuit.
Raw averages mostly re-measure car performance and current-era form — e.g.
a driver's average finish at a given track over 2021-2023 is dominated by
"my car was fast those years," not circuit mastery specifically. The
residual approach isolates the part of a driver's performance at a circuit
that ISN'T explained by how fast their car is or how good they currently
are in general — closer to true race-day circuit-specific skill (tire
management, overtaking character, comfort with a particular layout).

How the residual is computed:
  1. Within each race, rank drivers by a composite pre-race strength score
     (z-scored Elo + z-scored car pace, both direction-normalized so
     "higher composite = expected to finish better").
  2. That composite rank is the "expected finish" — what we'd predict
     ignoring where they actually qualified and what happened in the race.
  3. residual = actual finishing position − expected rank.
     Negative residual = did better than the composite predicted (good).
     Positive residual = did worse (bad).
  4. Average a driver's residuals at a given circuit, across their prior
     starts there only (no leakage), and shrink toward 0 by sample size.

Deliberately independent of grid_position: the baseline here is Elo + car
pace, NOT where the driver actually qualified that weekend. This credits a
driver for genuinely strong one-lap pace at a specific track too (e.g. a
street-circuit specialist), not just race-day execution net of an
already-known starting spot.

DNFs are excluded from history-building (a mechanical failure isn't
circuit mastery, and dnf_rate already models that separately), but a DNF
race still gets a pre-race affinity value computed from prior starts.

Sample sizes are small: most drivers have only 3-8 career starts at any
given circuit. Same empirical Bayes shrinkage principle as
car_performance.py's DNF rate, but shrunk toward ZERO (no circuit-specific
effect assumed) rather than toward a fitted global rate, since a residual
against a composite baseline has no other principled non-zero prior.

STATUS: this is a testable hypothesis, not an assumed win. Run it through
models/compare_feature_sets.py (with vs. without) before trusting it —
shrinkage may collapse it to near-zero for most drivers, in which case
it's not worth the added model complexity.

Usage:
    from feature_engineering.driver_track_affinity import build_track_affinity
    affinity_df = build_track_affinity(race_df, driver_ratings_df, car_performance_df)
"""

import pandas as pd
from collections import defaultdict

# Empirical Bayes shrinkage strength — "K pseudo-observations of zero
# circuit-specific effect". Much smaller than car_performance.py's DNF
# rate shrinkage (K=15), because circuit-specific history is inherently
# tiny — a driver typically races each circuit once per season, vs. DNF's
# 40-entry rolling window. K=6 means a driver needs roughly 6 career
# starts at a specific circuit before their circuit-specific signal is
# trusted about as much as the "assume no effect" prior.
#
# Untuned — a reasonable starting guess, same status as SCORE_NOISE_STD_FRAC
# (simulation.py) and PIT_SWAP_PROB_PER_STOP (simulation.py). Worth
# grid-searching against backtest performance later (backlog item D5).
SHRINKAGE_STRENGTH = 6


def _zscore(s: pd.Series) -> pd.Series:
    """Z-score a series. Returns all-zero if std is 0 or NaN (degenerate field)."""
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _expected_finish_rank(race_group: pd.DataFrame) -> pd.Series:
    """
    Rank drivers in a single race's field by a composite pre-race strength
    score, and return the rank each driver would be EXPECTED to finish in
    — ignoring where they actually started and what happened in the race.

    Args:
        race_group: rows for one race, with elo_pre_race and
            pace_delta_vs_pole already joined in.

    Returns:
        Series aligned to race_group's index: expected finish rank
        (1 = strongest pre-race composite score in the field).
    """
    elo = race_group["elo_pre_race"]
    # pace_delta_vs_pole: 0 = fastest car, larger = slower. Negate so
    # higher = better, matching Elo's "higher = better" direction.
    pace = -race_group["pace_delta_vs_pole"]

    # Z-score both within this race's field so Elo (scale ~1400-1800) and
    # pace delta (scale ~0-3 seconds) contribute comparably to the composite.
    composite = _zscore(elo) + _zscore(pace)
    return composite.rank(ascending=False, method="first")


def build_track_affinity(
    race_df: pd.DataFrame,
    driver_ratings_df: pd.DataFrame,
    car_performance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute rolling, shrinkage-adjusted driver-track affinity.

    Args:
        race_df: from load_race_results(), must include:
            season, round, raceId, circuitId, driverId, constructorId,
            positionOrder, status
        driver_ratings_df: from build_driver_ratings(), must include:
            season, round, raceId, driverId, elo_pre_race
        car_performance_df: from build_car_performance(), must include:
            season, round, raceId, constructorId, pace_delta_vs_pole

    Returns:
        DataFrame with columns:
            season, round, raceId, circuitId, driverId,
            track_affinity, track_affinity_raw, n_circuit_history

        track_affinity     = shrunk toward 0 (use this — the model feature).
                              Never NaN: a driver's first-ever start at a
                              circuit correctly evaluates to exactly 0.0
                              (no evidence yet of over/underperformance).
        track_affinity_raw = unshrunk rolling average residual, kept for
                              diagnostics only. NaN when n_circuit_history == 0.
        n_circuit_history   = prior starts by this driver at this circuit
                              specifically (low n = heavier shrinkage).

        Negative track_affinity = driver tends to outperform their
        car-pace + current-form expectation at this circuit (good sign).
        Positive = tends to underperform expectation there.
    """
    df = race_df.merge(
        driver_ratings_df[["season", "round", "raceId", "driverId", "elo_pre_race"]],
        on=["season", "round", "raceId", "driverId"],
        how="left",
    ).merge(
        car_performance_df[["season", "round", "raceId", "constructorId", "pace_delta_vs_pole"]],
        on=["season", "round", "raceId", "constructorId"],
        how="left",
    )
    df = df.sort_values(["season", "round"]).reset_index(drop=True)

    finished_mask = df["status"] == "Finished"

    records = []
    # {(driverId, circuitId): [residual, residual, ...]}
    circuit_history: dict = defaultdict(list)

    for (season, rnd, race_id, circuit_id), race_group in df.groupby(
        ["season", "round", "raceId", "circuitId"], sort=False
    ):
        # Need both elo and pace to compute an expected rank; rows missing
        # either (mainly early-season rows before any history exists) are
        # dropped just for the ranking step.
        rankable = race_group.dropna(subset=["elo_pre_race", "pace_delta_vs_pole"])
        expected_rank = _expected_finish_rank(rankable) if len(rankable) > 1 else pd.Series(dtype=float)

        # Record PRE-RACE affinity for every driver in the field, using
        # only prior starts at this circuit — this race's own outcome is
        # not folded into history until after this block.
        for _, row in race_group.iterrows():
            key = (row["driverId"], circuit_id)
            history = circuit_history[key]
            n = len(history)

            raw_avg = (sum(history) / n) if n > 0 else float("nan")

            # Shrink toward 0. n=0 → shrunk == 0.0 exactly (full shrinkage,
            # no history to trust yet). n >= SHRINKAGE_STRENGTH → shrunk
            # approaches raw_avg.
            effective_raw = raw_avg if n > 0 else 0.0
            shrunk = (n * effective_raw) / (n + SHRINKAGE_STRENGTH)

            records.append({
                "season":             season,
                "round":              rnd,
                "raceId":             race_id,
                "circuitId":          circuit_id,
                "driverId":           row["driverId"],
                "track_affinity":     shrunk,
                "track_affinity_raw": raw_avg,
                "n_circuit_history":  n,
            })

        # After recording, fold this race's residual into history — only
        # for drivers who finished (classified) AND were rankable.
        for idx, row in race_group.iterrows():
            if idx not in expected_rank.index or not finished_mask.loc[idx]:
                continue
            residual = row["positionOrder"] - expected_rank.loc[idx]
            circuit_history[(row["driverId"], circuit_id)].append(residual)

    affinity_df = pd.DataFrame(records)
    avg_shrinkage = (affinity_df["n_circuit_history"] < SHRINKAGE_STRENGTH).mean()
    print(f"  ✓ Driver-track affinity: {len(affinity_df)} rows, "
          f"{affinity_df['driverId'].nunique()} unique drivers, "
          f"{affinity_df['circuitId'].nunique()} unique circuits "
          f"({avg_shrinkage:.1%} of rows had n_circuit_history < "
          f"{SHRINKAGE_STRENGTH}, meaningfully shrunk toward 0)")
    return affinity_df


# ---------------------------------------------------------------------------
# Sanity check — run this file directly to verify the logic independently
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("driver_track_affinity.py — sanity check with synthetic data\n")

    # Driver A: weakest by Elo/pace every time (expected rank 4, worst in
    # field), but ALWAYS actually finishes P1 — a genuine, repeated
    # beat-expectation signal at this circuit.
    # Driver B: strongest by Elo/pace every time (expected rank 1), but
    # ALWAYS actually finishes P4 — a genuine, repeated miss.
    # C and D finish exactly where their composite predicts — zero
    # residual, a useful contrast case.
    race_rows, elo_rows, pace_rows = [], [], []
    for rnd, season in enumerate([2021, 2022, 2023, 2024], start=1):
        race_rows += [
            {"season": season, "round": rnd, "raceId": rnd, "circuitId": "circuit_x",
             "driverId": "A", "constructorId": "C1", "positionOrder": 1, "status": "Finished"},
            {"season": season, "round": rnd, "raceId": rnd, "circuitId": "circuit_x",
             "driverId": "C", "constructorId": "C3", "positionOrder": 2, "status": "Finished"},
            {"season": season, "round": rnd, "raceId": rnd, "circuitId": "circuit_x",
             "driverId": "D", "constructorId": "C4", "positionOrder": 3, "status": "Finished"},
            {"season": season, "round": rnd, "raceId": rnd, "circuitId": "circuit_x",
             "driverId": "B", "constructorId": "C2", "positionOrder": 4, "status": "Finished"},
        ]
        elo_rows += [
            {"season": season, "round": rnd, "raceId": rnd, "driverId": "A", "elo_pre_race": 1500},
            {"season": season, "round": rnd, "raceId": rnd, "driverId": "B", "elo_pre_race": 1650},
            {"season": season, "round": rnd, "raceId": rnd, "driverId": "C", "elo_pre_race": 1580},
            {"season": season, "round": rnd, "raceId": rnd, "driverId": "D", "elo_pre_race": 1550},
        ]
        pace_rows += [
            {"season": season, "round": rnd, "raceId": rnd, "constructorId": "C1", "pace_delta_vs_pole": 0.7},
            {"season": season, "round": rnd, "raceId": rnd, "constructorId": "C2", "pace_delta_vs_pole": 0.2},
            {"season": season, "round": rnd, "raceId": rnd, "constructorId": "C3", "pace_delta_vs_pole": 0.4},
            {"season": season, "round": rnd, "raceId": rnd, "constructorId": "C4", "pace_delta_vs_pole": 0.5},
        ]

    race_df = pd.DataFrame(race_rows)
    driver_ratings_df = pd.DataFrame(elo_rows)
    car_performance_df = pd.DataFrame(pace_rows)

    result = build_track_affinity(race_df, driver_ratings_df, car_performance_df)
    print("\nOutput:")
    print(result.to_string(index=False))

    print("\nChecks:")
    a_2024 = result[(result.driverId == "A") & (result.season == 2024)]["track_affinity"].values[0]
    b_2024 = result[(result.driverId == "B") & (result.season == 2024)]["track_affinity"].values[0]
    assert a_2024 < 0, f"Expected A's affinity to be negative (outperforms), got {a_2024}"
    assert b_2024 > 0, f"Expected B's affinity to be positive (underperforms), got {b_2024}"
    assert a_2024 < b_2024, "A should have a more favorable (lower) affinity than B by 2024"
    print(f"  ✓ Driver A (consistently beats expectation): track_affinity = {a_2024:.3f} (negative, good)")
    print(f"  ✓ Driver B (consistently misses expectation): track_affinity = {b_2024:.3f} (positive, bad)")

    a_2021 = result[(result.driverId == "A") & (result.season == 2021)]["track_affinity"].values[0]
    assert a_2021 == 0.0, "First-ever start at a circuit should have zero history → fully shrunk to 0"
    print(f"  ✓ Driver A's very first start at circuit_x: track_affinity = {a_2021:.3f} (no history yet)")

    print("\nAll checks passed ✓")
