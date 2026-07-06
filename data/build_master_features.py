"""
data/build_master_features.py

Joins all computed features into a single training-ready DataFrame.
One row per driver per race, all features present, ready for the ML model.

Sources joined:
  1. driver_ratings.parquet  — elo_pre_race, recent_form, teammate_delta
  2. car_performance         — pace_delta_vs_pole, dnf_rate (computed live)
  2b. driver_track_affinity  — track_affinity (computed live, NEW in v2 —
                               residual-based, shrinkage-adjusted; see
                               feature_engineering/driver_track_affinity.py)
  3. track_car_interaction   — straight_line_exposure, cornering_exposure
                               (pace_delta_vs_pole weighted by circuit's
                               sector throttle/braking character)
  4. track_features.parquet  — full_throttle_pct, avg_corner_speed,
                               lap_length_km, drs_zones, is_street_circuit
  5. quali_results.parquet + race_results.parquet — grid_position (actual,
                               penalty-adjusted starting grid — see
                               fetch_jolpica.py for the fix that made this
                               correct; falls back to qualifying
                               classification only when 'grid' is missing)
  6. race_results.parquet    — target variable: finishing position

Output: data/processed/master_features.parquet

Usage (from project root):
    python -m data.build_master_features
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_jolpica import load_race_results, load_qualifying
from feature_engineering.driver_ratings import build_driver_ratings
from feature_engineering.car_performance import build_car_performance
from feature_engineering.track_features import load_track_features
from feature_engineering.track_car_interaction import add_track_car_interaction
from feature_engineering.driver_track_affinity import build_track_affinity
from config import DRIVER_RATINGS_CACHE, TRACK_FEATURES_CACHE, PROCESSED_DIR

MASTER_FEATURES_PATH = os.path.join(PROCESSED_DIR, "master_features.parquet")

# One-time circuit telemetry analysis from circuit_profiler.py.
# Tracks barely change layout, so this is a long-lived reference table —
# not regenerated on every run, only re-profiled (via --circuit/--season
# --save) if a specific circuit's layout changes.
CIRCUIT_PROFILES_PATH = r"D:\Projects\F1 Predictor(Actual prediction)\files\Some data to save\circuit_profiles.xlsx"


# ---------------------------------------------------------------------------
# Section 1: Load all sources
# ---------------------------------------------------------------------------

def _load_driver_ratings(race_df: pd.DataFrame) -> pd.DataFrame:
    """Load from cache if available, else recompute."""
    if os.path.exists(DRIVER_RATINGS_CACHE):
        print("  ✓ Driver ratings: loaded from cache")
        return pd.read_parquet(DRIVER_RATINGS_CACHE)
    print("  Computing driver ratings...")
    df = build_driver_ratings(race_df)
    df.to_parquet(DRIVER_RATINGS_CACHE, index=False)
    return df


def _load_track_features_safe() -> pd.DataFrame:
    """Load track features if cache exists, else return empty DataFrame."""
    if os.path.exists(TRACK_FEATURES_CACHE):
        print("  ✓ Track features: loaded from cache")
        return load_track_features()
    print("  ⚠ Track features cache missing — run track_features.py first")
    print("    Continuing without track features (they'll be NaN)")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Section 2: Imputation
# ---------------------------------------------------------------------------
# The ML model can't handle NaN values. We need to fill them sensibly.
#
# Strategy per feature:
#   elo_pre_race            — NaN only for brand-new drivers in race 1. Fill
#                             with ELO_INITIAL (1500) — they start neutral.
#   recent_form              — NaN for race 1 of a driver's career. Fill with
#                             median finishing position (~ P10 for a midfield default).
#   teammate_delta           — NaN when no prior teammate comparison. Fill with 0
#                             (neutral — no evidence of being better or worse).
#   pace_delta               — NaN for new constructors. Fill with median pace delta
#                             (put them in the midfield, not assuming front or back).
#   dnf_rate                 — NaN for new constructors. Fill with median DNF rate.
#   track_affinity           — Should rarely be NaN (compute_track_affinity returns
#                             a numeric value, shrunk toward 0, for every row). Fill
#                             with 0.0 (no assumed circuit-specific effect) as a
#                             defensive fallback only.
#   straight_line_exposure   — NaN if pace_delta_vs_pole was NaN before this point.
#   cornering_exposure         Fill with 0.0 — no exposure assumed by default.
#                             (Circuit-level NaN from a missing circuitId match
#                             is already handled inside add_track_car_interaction().)
#   track features            — NaN when FastF1 session failed. Forward-filled within
#                             load_track_features(). Remaining NaN filled with median.
#   grid_position             — Should never be NaN (qualifying always happens).
#                             Fill with 20 if missing (start last as a safe default).

def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values with sensible defaults per feature."""
    from config import ELO_INITIAL

    fills = {
        "elo_pre_race":           ELO_INITIAL,
        "recent_form":            df["recent_form"].median()            if "recent_form"            in df else 10.0,
        "teammate_delta":         0.0,
        "pace_delta_vs_pole":     df["pace_delta_vs_pole"].median()     if "pace_delta_vs_pole"     in df else 1.0,
        "dnf_rate":               df["dnf_rate"].median()               if "dnf_rate"               in df else 0.1,
        "track_affinity":         0.0,
        "straight_line_exposure": df["straight_line_exposure"].median() if "straight_line_exposure" in df else 0.0,
        "cornering_exposure":     df["cornering_exposure"].median()     if "cornering_exposure"     in df else 0.0,
        "full_throttle_pct":      df["full_throttle_pct"].median()      if "full_throttle_pct"      in df else 55.0,
        "avg_corner_speed":       df["avg_corner_speed"].median()       if "avg_corner_speed"       in df else 150.0,
        "lap_length_km":          df["lap_length_km"].median()          if "lap_length_km"          in df else 5.0,
        "drs_zones":              2,
        "is_street_circuit":      0,
        "wet_flag":               0,
        "grid_position":          20,
    }

    for col, fill_val in fills.items():
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df[col].fillna(fill_val)
            after  = df[col].isna().sum()
            if before > 0:
                print(f"    Imputed {before} NaN → {fill_val:.2f} in '{col}'")

    return df


# ---------------------------------------------------------------------------
# Section 3: Master join
# ---------------------------------------------------------------------------

def build_master_features() -> pd.DataFrame:
    """
    Join all feature sources into one training-ready DataFrame.

    Returns:
        DataFrame with one row per driver per race, columns:
            season, round, raceId, circuitId, driverId, constructorId,
            elo_pre_race, recent_form, teammate_delta,
            pace_delta_vs_pole, dnf_rate,
            straight_line_exposure, cornering_exposure,
            full_throttle_pct, avg_corner_speed, lap_length_km,
            drs_zones, is_street_circuit, wet_flag,
            grid_position,
            finishing_position  ← target variable
    """
    print("\nLoading source data...")
    race_df  = load_race_results()
    quali_df = load_qualifying()
    print(f"  ✓ Race results: {len(race_df)} rows")
    print(f"  ✓ Qualifying:   {len(quali_df)} rows")

    # --- Driver ratings ---
    print("\nLoading driver ratings...")
    driver_df = _load_driver_ratings(race_df)

    # --- Car performance (computed live, fast) ---
    print("\nComputing car performance features...")
    car_df = build_car_performance(quali_df, race_df)

    # --- Driver-track affinity (residual-based, needs driver_df + car_df) ---
    print("\nComputing driver-track affinity...")
    affinity_df = build_track_affinity(race_df, driver_df, car_df)

    # --- Track features ---
    print("\nLoading track features...")
    track_df = _load_track_features_safe()

    # --- Base: race results with finishing position as target ---
    # Keep only the columns we need from race_df
    base = race_df[[
        "season", "round", "raceId", "circuitId",
        "driverId", "constructorId",
        "position", "positionOrder", "status",
    ]].copy()

    # Finishing position — use positionOrder (always numeric, even for DNFs)
    # DNFs get positionOrder = their retirement order (e.g. P18 if 3 drivers retired)
    # This is what the model predicts
    base["finishing_position"] = base["positionOrder"].astype(float)

    # --- Join 1: Driver ratings ---
    merge_keys_driver = ["season", "round", "raceId", "driverId"]
    master = base.merge(
        driver_df[merge_keys_driver + ["elo_pre_race", "recent_form", "teammate_delta"]],
        on=merge_keys_driver,
        how="left",
    )
    print(f"\n  After driver join: {len(master)} rows, "
          f"{master[['elo_pre_race']].isna().sum().sum()} NaN elo values")

    # --- Join 2: Car performance ---
    merge_keys_car = ["season", "round", "raceId", "constructorId"]
    master = master.merge(
        car_df[merge_keys_car + ["pace_delta_vs_pole", "dnf_rate"]],
        on=merge_keys_car,
        how="left",
    )

    # --- Join 2b: Driver-track affinity ---
    # Residual-based, shrinkage-adjusted (see feature_engineering/
    # driver_track_affinity.py for full methodology and caveats).
    # NEW in v2 — treat as a testable hypothesis, not an assumed win.
    # Run through models/compare_feature_sets.py before trusting it.
    master = master.merge(
        affinity_df[merge_keys_driver + ["track_affinity"]],
        on=merge_keys_driver,
        how="left",
    )

    # --- Track × car interaction features ---
    # Combines pace_delta_vs_pole with the upcoming circuit's sector
    # throttle/braking character (from the one-time circuit_profiler.py
    # output) to let the model learn whether a constructor's pace
    # advantage is circuit-dependent — without hardcoding any assumption
    # about WHY a car is fast on a given track.
    print("\nAdding track × car interaction features...")
    master = add_track_car_interaction(master, CIRCUIT_PROFILES_PATH)

    # --- Join 3: Track features ---
    if not track_df.empty:
        merge_keys_track = ["circuitId", "season"]
        track_cols = merge_keys_track + [
            "full_throttle_pct", "avg_corner_speed", "lap_length_km",
            "drs_zones", "is_street_circuit", "wet_flag",
        ]
        master = master.merge(
            track_df[track_cols],
            on=merge_keys_track,
            how="left",
        )
    else:
        # Add NaN columns so downstream code doesn't break
        for col in ["full_throttle_pct", "avg_corner_speed", "lap_length_km",
                    "drs_zones", "is_street_circuit", "wet_flag"]:
            master[col] = np.nan

    # --- Join 4: Grid position ---
    # Use the ACTUAL, penalty-adjusted starting grid (from race_df's
    # actual_grid_position, sourced from results.json's 'grid' field) rather
    # than the qualifying classification. These differ whenever a driver gets
    # a grid penalty after qualifying (engine/gearbox/impeding penalties,
    # DSQ from quali, etc.) — e.g. Gasly qualified 13th at the 2024 Azerbaijan
    # GP but actually started P18 after a fuel-flow DSQ. Previously this join
    # used quali_df's grid_position (pre-penalty), which silently mislabeled
    # every penalized driver's starting position in training data.
    #
    # Fall back to the qualifying classification only for the rare rows where
    # the results endpoint didn't provide a 'grid' value.
    actual_grid = race_df[["season", "round", "raceId", "driverId", "actual_grid_position"]].copy()
    master = master.merge(
        actual_grid,
        on=["season", "round", "raceId", "driverId"],
        how="left",
    )
    quali_grid = quali_df[["season", "round", "raceId", "driverId", "grid_position"]].copy()
    quali_grid = quali_grid.rename(columns={"grid_position": "quali_grid_position"})
    master = master.merge(
        quali_grid,
        on=["season", "round", "raceId", "driverId"],
        how="left",
    )
    master["grid_position"] = master["actual_grid_position"].fillna(master["quali_grid_position"])
    fallback_count = master["actual_grid_position"].isna().sum()
    if fallback_count > 0:
        print(f"  ⚠ {fallback_count} rows missing actual_grid_position — "
              f"fell back to qualifying classification for these.")
    master = master.drop(columns=["actual_grid_position", "quali_grid_position"])

    print(f"\n  Master table shape before imputation: {master.shape}")

    # --- Imputation ---
    print("\nImputing missing values...")
    master = impute(master)

    # --- Final column selection and ordering ---
    feature_cols = [
        "season", "round", "raceId", "circuitId", "driverId", "constructorId",
        # Driver features
        "elo_pre_race", "recent_form", "teammate_delta",
        # Driver-track affinity (NEW in v2 — residual-based, shrinkage-adjusted)
        "track_affinity",
        # Car features
        "pace_delta_vs_pole", "dnf_rate",
        # Track × car interaction features
        "straight_line_exposure", "cornering_exposure",
        # Track features
        "full_throttle_pct", "avg_corner_speed", "lap_length_km",
        "drs_zones", "is_street_circuit", "wet_flag",
        # Race context
        "grid_position",
        # Target
        "finishing_position",
        # Keep for reference / Monte Carlo
        "status",
    ]
    # Only keep columns that exist
    feature_cols = [c for c in feature_cols if c in master.columns]
    master = master[feature_cols]

    print(f"\n  ✓ Master features: {master.shape[0]} rows × {master.shape[1]} cols")
    print(f"  Features: {[c for c in feature_cols if c not in ['season','round','raceId','circuitId','driverId','constructorId','finishing_position','status']]}")

    return master


# ---------------------------------------------------------------------------
# Section 4: Summary stats
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the master feature table."""
    print(f"\n{'='*60}")
    print(f"MASTER FEATURE TABLE SUMMARY")
    print(f"{'='*60}")
    print(f"Rows:     {len(df):,}")
    print(f"Seasons:  {df['season'].min()} – {df['season'].max()}")
    print(f"Races:    {df.groupby(['season','round']).ngroups}")
    print(f"Drivers:  {df['driverId'].nunique()}")
    print(f"\nNaN counts per feature:")

    feat_cols = [c for c in df.columns if c not in
                 ["season","round","raceId","circuitId","driverId",
                  "constructorId","finishing_position","status"]]
    for col in feat_cols:
        n_nan = df[col].isna().sum()
        pct   = n_nan / len(df) * 100
        flag  = " ⚠" if pct > 5 else ""
        print(f"  {col:<25} {n_nan:>5} NaN  ({pct:.1f}%){flag}")

    print(f"\nSample (last race in dataset):")
    last = df[df["season"] == df["season"].max()]
    last = last[last["round"] == last["round"].max()]
    last = last.sort_values("finishing_position")
    print(last[["driverId", "grid_position", "elo_pre_race",
                "recent_form", "pace_delta_vs_pole",
                "straight_line_exposure", "cornering_exposure",
                "finishing_position"]
               ].to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    master = build_master_features()

    print(f"\nSaving to {MASTER_FEATURES_PATH}...")
    master.to_parquet(MASTER_FEATURES_PATH, index=False)
    print(f"  ✓ Saved")

    _print_summary(master)