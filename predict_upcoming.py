"""
predict_upcoming.py

Predicts an UPCOMING race — one that may not have happened yet, and may
not even have qualifying results yet — using only:
  - rolling features computed from all races strictly BEFORE this round
    (Elo, recent form, teammate delta, pace delta, DNF rate)
  - grid position, either fetched live from Jolpica (if qualifying has
    happened) or supplied manually via --grid-csv (if it hasn't)

WHY THIS EXISTS:
build_master_features.py can only build a feature row for a race that
already exists in the cached race_df — i.e. one Jolpica has SOME record
of. A race that hasn't happened yet has no row there at all, so that
pipeline structurally can't produce a prediction for it, regardless of
what features it would compute. This script builds the feature row
directly instead of going through build_master_features.py, so it
works whether the target race is fully in the past, mid-weekend
(quali done, race not run), or entirely in the future.

HOW IT WORKS (placeholder-row technique):
build_driver_ratings() and build_car_performance() are already written
to process races in chronological order and record each race's ROLLING
value as of right BEFORE that race, then fold that race's own result
into history afterward. This script exploits that directly: it appends
ONE placeholder row for the target round (driverId/constructorId from
quali or a manual grid, no result data that's ever actually read) onto
the real cached historical race_df/quali_df, re-runs those same
functions completely unmodified, and reads off the pre-race value they
compute for the appended round. No changes needed to driver_ratings.py
or car_performance.py.

WHAT THIS DOES NOT DO:
- Does not modify any cached parquet files — all rolling recomputation
  happens in memory and is discarded after this run.
- Does not know the weather in advance — wet_flag defaults to 0 (dry)
  unless you pass --wet.
- Track telemetry features (full_throttle_pct, avg_corner_speed, etc.)
  can't come from the CURRENT season's session if the race hasn't run
  yet (telemetry requires the session to have happened) — this falls
  back to the most recent available season's telemetry for that
  circuit, which is a reasonable proxy since track layouts barely
  change year to year (same reasoning as circuit_profiler.py).

Usage (from project root):
    # Qualifying already happened — pulls grid positions live from Jolpica
    python predict_upcoming.py --season 2026 --round 9

    # Qualifying hasn't happened yet — supply grid positions manually
    python predict_upcoming.py --season 2026 --round 9 --grid-csv my_grid.csv

    # Assume a wet race (weather isn't knowable in advance otherwise)
    python predict_upcoming.py --season 2026 --round 9 --wet

    # See the season's schedule to find round numbers / circuit names
    python predict_upcoming.py --list-schedule --season 2026

grid-csv format (only needed if quali hasn't happened yet):
    driverId,grid_position
    russell,1
    max_verstappen,2
    ...
    (constructorId is looked up automatically from each driver's most
    recent known team in the cached historical data. Add a
    constructorId column yourself to override this — e.g. for a driver
    swap the historical data doesn't reflect yet, or a brand-new driver
    with no history at all, which would otherwise raise an error.)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetch_jolpica import (
    load_race_results,
    load_qualifying,
    fetch_race_schedule,
    fetch_qualifying_for_round,
)
from feature_engineering.driver_ratings import build_driver_ratings
from feature_engineering.car_performance import build_car_performance
from feature_engineering.track_features import load_track_features
from feature_engineering.track_car_interaction import add_track_car_interaction
from data.build_master_features import impute, CIRCUIT_PROFILES_PATH
from models.race_model import FEATURE_COLS
from simulation.simulation import _load_model, run_monte_carlo, summarize, N_RUNS_DEFAULT
from oracle import print_terminal_table, export_csv, build_heatmap, export_heatmap, CIRCUIT_TO_RACE_NAME

TRACK_FEATURE_COLS = ["full_throttle_pct", "avg_corner_speed", "lap_length_km",
                       "drs_zones", "is_street_circuit", "wet_flag"]


# ---------------------------------------------------------------------------
# Section 1: Resolve the target round's circuit
# ---------------------------------------------------------------------------

def resolve_circuit(season: int, round_num: int) -> tuple[str, str]:
    """
    Look up circuitId and raceName for (season, round) via the live
    schedule fetch — works even if the race hasn't happened yet, since
    F1 calendars are published months in advance.
    """
    schedule = fetch_race_schedule(season)
    match = schedule[schedule["round"] == round_num]
    if match.empty:
        available = sorted(schedule["round"].tolist())
        raise ValueError(
            f"No round {round_num} in the {season} schedule. "
            f"Available rounds: {available}"
        )
    return match["circuitId"].iloc[0], match["raceName"].iloc[0]


def print_schedule(season: int) -> None:
    schedule = fetch_race_schedule(season)
    print(f"\n{season} schedule:")
    for _, row in schedule.sort_values("round").iterrows():
        print(f"  Round {int(row['round']):>2}  {row['raceName']:<30} "
              f"(circuitId: {row['circuitId']}, date: {row['date']})")
    print()


# ---------------------------------------------------------------------------
# Section 2: Determine the driver/constructor/grid lineup
# ---------------------------------------------------------------------------

def get_lineup(season: int, round_num: int, grid_csv_path: str = None) -> tuple[pd.DataFrame, str]:
    """
    Determine (driverId, constructorId, grid_position) for the target
    round. Tries live qualifying first; falls back to a manually
    supplied CSV if quali hasn't happened yet.

    Returns:
        (lineup DataFrame, human-readable description of the source)
    """
    live_quali = fetch_qualifying_for_round(season, round_num)
    if not live_quali.empty:
        lineup = live_quali[["driverId", "constructorId", "grid_position"]].copy()
        return lineup, "live qualifying data from Jolpica"

    if grid_csv_path:
        manual = pd.read_csv(grid_csv_path)
        if "driverId" not in manual.columns or "grid_position" not in manual.columns:
            raise ValueError("--grid-csv must have at least 'driverId' and 'grid_position' columns")

        if "constructorId" not in manual.columns:
            hist_race = load_race_results()[["season", "round", "driverId", "constructorId"]]
            hist_quali = load_qualifying()[["season", "round", "driverId", "constructorId"]]
            combined_hist = pd.concat([hist_race, hist_quali], ignore_index=True)
            latest_team = (
                combined_hist.sort_values(["driverId", "season", "round"])
                .groupby("driverId").last()["constructorId"]
            )
            manual["constructorId"] = manual["driverId"].map(latest_team)
            missing = manual[manual["constructorId"].isna()]
            if not missing.empty:
                raise ValueError(
                    f"Couldn't auto-lookup constructorId for: {missing['driverId'].tolist()}. "
                    f"These drivers have no history (e.g. brand-new to F1) — add a "
                    f"constructorId column to your grid CSV for them."
                )

        lineup = manual[["driverId", "constructorId", "grid_position"]].copy()
        return lineup, f"manual grid CSV ({grid_csv_path})"

    raise ValueError(
        "No qualifying data available yet for this round, and no --grid-csv supplied.\n"
        "Either wait until qualifying happens, or provide grid positions manually via:\n"
        "  --grid-csv my_grid.csv\n"
        "CSV format: driverId,grid_position[,constructorId]"
    )


# ---------------------------------------------------------------------------
# Section 3: Placeholder-row technique for rolling features
# ---------------------------------------------------------------------------

def _align_dtypes(placeholder_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    """
    Force each placeholder column to match the real DataFrame's dtype
    before concatenation. The placeholder's None-valued columns (e.g.
    q1/q2/q3, position) would otherwise leave pandas to INFER a result
    dtype at concat time — which is exactly the ambiguity behind the
    'DataFrame concatenation with empty or all-NA entries' deprecation
    warning. Making dtypes match explicitly beforehand removes the
    ambiguity entirely, so there's nothing left to infer.
    """
    placeholder_df = placeholder_df.copy()
    for col in placeholder_df.columns:
        if col in real_df.columns:
            try:
                placeholder_df[col] = placeholder_df[col].astype(real_df[col].dtype)
            except (ValueError, TypeError):
                pass  # leave as-is if genuinely incompatible — shouldn't happen here
    return placeholder_df


def _build_placeholder_rows(season: int, round_num: int, circuit_id: str,
                             lineup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build one placeholder row per driver for the target round, in the
    same shape as real race_df/quali_df rows. Result/status/quali-time
    values are never actually read for meaningful computation — they
    only matter for the fold-into-history step that happens AFTER the
    round we're extracting, and since this is the last (most recent)
    round in the combined data, that step has nothing after it to
    affect. See module docstring for the full explanation.
    """
    race_rows, quali_rows = [], []
    for _, row in lineup.iterrows():
        race_rows.append({
            "season": season, "round": round_num, "circuitId": circuit_id,
            "raceId": circuit_id, "driverId": row["driverId"],
            "constructorId": row["constructorId"],
            "position": None, "positionOrder": 999, "status": "Unknown",
        })
        quali_rows.append({
            "season": season, "round": round_num, "circuitId": circuit_id,
            "raceId": circuit_id, "driverId": row["driverId"],
            "constructorId": row["constructorId"],
            "grid_position": int(row["grid_position"]),
            "q1": None, "q2": None, "q3": None,
        })
    return pd.DataFrame(race_rows), pd.DataFrame(quali_rows)


def compute_rolling_features(season: int, round_num: int, circuit_id: str,
                              lineup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Append the placeholder round onto real cached history, re-run the
    existing rolling feature functions unmodified, and extract just the
    pre-race values for our target round.

    Nothing here writes back to any cache file — all recomputation is
    in-memory and discarded after this function returns.

    Returns:
        (driver_target, car_target) — DataFrames with the rolling
        feature values for exactly this round's drivers/constructors.
    """
    real_race_df = load_race_results()
    real_quali_df = load_qualifying()

    # CRITICAL: if the target round already exists in the real cached
    # data (e.g. you're testing this tool against a round that's already
    # happened, for verification), drop those existing rows before
    # appending the placeholder. Without this, a round that's already
    # in history gets TWO entries per driver — one real, one placeholder
    # — which fans out into duplicate rows through every downstream
    # merge (car performance, driver ratings, track interaction) and
    # corrupts the final probabilities. This was caught by testing
    # against round 8 (already-raced) and seeing every driver appear
    # 8x with impossible probabilities (P(Top3) > 100%) — the fix makes
    # this tool safe to use against BOTH future rounds (nothing to
    # drop) and already-happened rounds (real entries cleanly replaced
    # by the placeholder's synthetic grid, for apples-to-apples testing
    # against oracle.py / predict_vs_actual.py).
    real_race_df = real_race_df[
        ~((real_race_df["season"] == season) & (real_race_df["round"] == round_num))
    ]
    real_quali_df = real_quali_df[
        ~((real_quali_df["season"] == season) & (real_quali_df["round"] == round_num))
    ]

    placeholder_race_df, placeholder_quali_df = _build_placeholder_rows(
        season, round_num, circuit_id, lineup
    )

    # Align dtypes before concatenating — removes the inference ambiguity
    # that triggers pandas' "empty or all-NA entries" FutureWarning on
    # older pandas versions (harmless either way, but worth silencing
    # cleanly rather than leaving a warning that could mask a real one
    # later).
    placeholder_race_df = _align_dtypes(placeholder_race_df, real_race_df)
    placeholder_quali_df = _align_dtypes(placeholder_quali_df, real_quali_df)

    combined_race_df = pd.concat([real_race_df, placeholder_race_df], ignore_index=True)
    combined_quali_df = pd.concat([real_quali_df, placeholder_quali_df], ignore_index=True)

    print("  Computing driver ratings (Elo, recent form, teammate delta)...")
    driver_df = build_driver_ratings(combined_race_df)
    driver_target = driver_df[
        (driver_df["season"] == season) & (driver_df["round"] == round_num)
    ][["driverId", "elo_pre_race", "recent_form", "teammate_delta"]]

    print("  Computing car performance (pace delta, DNF rate)...")
    car_df = build_car_performance(combined_quali_df, combined_race_df)
    car_target = car_df[
        (car_df["season"] == season) & (car_df["round"] == round_num)
    ][["constructorId", "pace_delta_vs_pole", "dnf_rate"]]

    return driver_target, car_target


# ---------------------------------------------------------------------------
# Section 4: Assemble the full feature row
# ---------------------------------------------------------------------------

def build_prediction_row(season: int, round_num: int, circuit_id: str,
                          lineup: pd.DataFrame, wet_override: bool) -> pd.DataFrame:
    driver_target, car_target = compute_rolling_features(season, round_num, circuit_id, lineup)

    race_df = lineup.copy()
    race_df["season"] = season
    race_df["round"] = round_num
    race_df["circuitId"] = circuit_id
    race_df["raceId"] = circuit_id

    race_df = race_df.merge(driver_target, on="driverId", how="left")
    race_df = race_df.merge(car_target, on="constructorId", how="left")

    print("  Adding track × car interaction features...")
    race_df = add_track_car_interaction(race_df, CIRCUIT_PROFILES_PATH)

    print("  Loading track features...")
    track_df = load_track_features()
    this_circuit_track = (
        track_df[track_df["circuitId"] == circuit_id]
        .sort_values("season", ascending=False)
    )
    if not this_circuit_track.empty:
        # Most recent available season's telemetry for this circuit —
        # layouts barely change year to year, and the CURRENT season's
        # session can't have telemetry yet if the race hasn't run.
        proxy_row = this_circuit_track.iloc[0]
        for col in TRACK_FEATURE_COLS:
            race_df[col] = proxy_row[col]
        print(f"    Using {int(proxy_row['season'])} telemetry for circuit "
              f"'{circuit_id}' as a proxy (this season's session hasn't run yet)")
    else:
        print(f"    ⚠ No track telemetry found for circuit '{circuit_id}' in any "
              f"season — will fall back to dataset medians via imputation")
        for col in TRACK_FEATURE_COLS:
            race_df[col] = np.nan

    if wet_override:
        race_df["wet_flag"] = 1
        print("    --wet flag set: overriding wet_flag=1 for this prediction")

    print("  Imputing any remaining missing values (same logic as build_master_features.py)...")
    race_df = impute(race_df)

    race_df["grid_position"] = race_df["grid_position"].astype(float)

    # Safety check — catch a broken row here with a clear message rather
    # than a cryptic error deep inside sklearn/the simulation loop.
    missing = [c for c in FEATURE_COLS
               if c not in race_df.columns or race_df[c].isna().any()]
    if missing:
        raise RuntimeError(
            f"Feature row incomplete after imputation — missing/NaN in: {missing}. "
            f"This shouldn't happen; something went wrong upstream. Inspect race_df."
        )

    return race_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict an upcoming F1 race")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--round", type=int, default=None, dest="round_num")
    parser.add_argument("--grid-csv", type=str, default=None,
                         help="CSV with driverId,grid_position[,constructorId] — "
                              "used only if qualifying hasn't happened yet")
    parser.add_argument("--wet", action="store_true",
                         help="Assume a wet race (weather isn't knowable in advance otherwise)")
    parser.add_argument("--runs", type=int, default=N_RUNS_DEFAULT)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--list-schedule", action="store_true",
                         help="Print the season's schedule and exit")
    args = parser.parse_args()

    if args.list_schedule:
        if args.season is None:
            print("--list-schedule requires --season, e.g. --list-schedule --season 2026")
            sys.exit(1)
        print_schedule(args.season)
        sys.exit(0)

    if args.season is None or args.round_num is None:
        print("Provide both --season and --round.")
        print("Example: python predict_upcoming.py --season 2026 --round 9")
        print("Use --list-schedule --season 2026 to see available rounds.")
        sys.exit(1)

    print(f"Resolving round {args.round_num} of season {args.season}...")
    circuit_id, race_name = resolve_circuit(args.season, args.round_num)
    print(f"  {race_name} ({circuit_id})")

    print("\nDetermining grid / lineup...")
    lineup, source = get_lineup(args.season, args.round_num, args.grid_csv)
    print(f"  ✓ {len(lineup)} drivers, source: {source}")

    print("\nBuilding prediction features (rolling history only — no leakage)...")
    race_df = build_prediction_row(args.season, args.round_num, circuit_id, lineup, args.wet)

    print("\nLoading trained model ensemble...")
    ensemble = _load_model()

    print(f"Running simulation ({args.runs:,} Monte Carlo runs)...")
    prob_matrix, dnf_prob = run_monte_carlo(race_df, ensemble, n_runs=args.runs)

    summary = summarize(prob_matrix, dnf_prob, race_df)

    print_terminal_table(summary, args.season, args.round_num, circuit_id)

    csv_path = export_csv(prob_matrix, args.season, args.round_num, circuit_id)
    print(f"✓ CSV saved → {csv_path}")

    fig = build_heatmap(prob_matrix, args.season, args.round_num, circuit_id)
    heatmap_path = export_heatmap(fig, args.season, args.round_num, circuit_id)
    print(f"✓ Heatmap saved → {heatmap_path}")

    if args.show:
        fig.show()
