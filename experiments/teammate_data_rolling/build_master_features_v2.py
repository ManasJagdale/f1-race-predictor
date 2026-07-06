"""
experiments/teammate_delta_rolling/build_master_features_v2.py

Rebuilds the full feature table using driver_ratings_v2.py's rolling-
window teammate_delta instead of the original career-average version.
Everything else (car performance, track features, track-car interaction,
driver-track affinity, imputation) is reused UNCHANGED from the real
project -- only the driver-ratings step differs.

WRITES ONLY WITHIN THIS FOLDER:
    experiments/teammate_delta_rolling/master_features_v2.parquet
    experiments/teammate_delta_rolling/driver_ratings_v2_cache.parquet

Never touches the real project's data/processed/master_features.parquet
or data/processed/driver_ratings.parquet. Deleting this whole folder
afterward removes every trace of this experiment.

IMPORTANT: unlike the real build_master_features.py, this ALWAYS
recomputes driver ratings from scratch (ignoring the real project's
cached driver_ratings.parquet) -- otherwise it would silently load the
OLD teammate_delta values from the real cache and this whole experiment
would be a no-op. It caches its own recomputed version locally instead,
so re-runs of this script are still fast.

Usage:
    python build_master_features_v2.py
"""

import os
import sys
import pandas as pd
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _PROJECT_ROOT)   # real project packages (feature_engineering, data, etc.)
sys.path.insert(0, _THIS_DIR)       # this folder's own driver_ratings_v2.py

from data.fetch_jolpica import load_race_results, load_qualifying
from feature_engineering.car_performance import build_car_performance
from feature_engineering.track_features import load_track_features
from feature_engineering.track_car_interaction import add_track_car_interaction
from feature_engineering.driver_track_affinity import build_track_affinity
from data.build_master_features import impute, CIRCUIT_PROFILES_PATH

from driver_ratings_v2 import build_driver_ratings   # <-- the ONE swapped import

MASTER_FEATURES_V2_PATH = os.path.join(_THIS_DIR, "master_features_v2.parquet")
DRIVER_RATINGS_V2_CACHE = os.path.join(_THIS_DIR, "driver_ratings_v2_cache.parquet")


def _load_driver_ratings_v2(race_df: pd.DataFrame) -> pd.DataFrame:
    """
    Same cache-or-recompute pattern as the real project's
    _load_driver_ratings(), but pointed at a LOCAL cache file so it
    never reads or writes the real project's driver_ratings.parquet.
    """
    if os.path.exists(DRIVER_RATINGS_V2_CACHE):
        print("  ✓ [v2] Driver ratings: loaded from local experiment cache")
        return pd.read_parquet(DRIVER_RATINGS_V2_CACHE)
    print("  [v2] Computing driver ratings (rolling teammate_delta)...")
    df = build_driver_ratings(race_df)
    df.to_parquet(DRIVER_RATINGS_V2_CACHE, index=False)
    return df


def _load_track_features_safe() -> pd.DataFrame:
    from config import TRACK_FEATURES_CACHE
    if os.path.exists(TRACK_FEATURES_CACHE):
        print("  ✓ Track features: loaded from real project cache (unchanged, read-only)")
        return load_track_features()
    print("  ⚠ Track features cache missing in the real project -- continuing without (NaN)")
    return pd.DataFrame()


def build_master_features_v2() -> pd.DataFrame:
    print("\nLoading source data (real project cache, read-only)...")
    race_df = load_race_results()
    quali_df = load_qualifying()
    print(f"  ✓ Race results: {len(race_df)} rows")
    print(f"  ✓ Qualifying:   {len(quali_df)} rows")

    print("\n[v2] Loading driver ratings (rolling teammate_delta)...")
    driver_df = _load_driver_ratings_v2(race_df)

    print("\nComputing car performance features (unchanged from real project)...")
    car_df = build_car_performance(quali_df, race_df)

    print("\nComputing driver-track affinity (unchanged logic -- depends on elo_pre_race, "
          "which is identical in v2, so this should barely move)...")
    affinity_df = build_track_affinity(race_df, driver_df, car_df)

    print("\nLoading track features (unchanged)...")
    track_df = _load_track_features_safe()

    base = race_df[[
        "season", "round", "raceId", "circuitId",
        "driverId", "constructorId",
        "position", "positionOrder", "status",
    ]].copy()
    base["finishing_position"] = base["positionOrder"].astype(float)

    merge_keys_driver = ["season", "round", "raceId", "driverId"]
    master = base.merge(
        driver_df[merge_keys_driver + ["elo_pre_race", "recent_form", "teammate_delta"]],
        on=merge_keys_driver, how="left",
    )

    merge_keys_car = ["season", "round", "raceId", "constructorId"]
    master = master.merge(
        car_df[merge_keys_car + ["pace_delta_vs_pole", "dnf_rate"]],
        on=merge_keys_car, how="left",
    )

    master = master.merge(
        affinity_df[merge_keys_driver + ["track_affinity"]],
        on=merge_keys_driver, how="left",
    )

    print("\nAdding track × car interaction features (unchanged)...")
    master = add_track_car_interaction(master, CIRCUIT_PROFILES_PATH)

    if not track_df.empty:
        merge_keys_track = ["circuitId", "season"]
        track_cols = merge_keys_track + [
            "full_throttle_pct", "avg_corner_speed", "lap_length_km",
            "drs_zones", "is_street_circuit", "wet_flag",
        ]
        master = master.merge(track_df[track_cols], on=merge_keys_track, how="left")
    else:
        for col in ["full_throttle_pct", "avg_corner_speed", "lap_length_km",
                    "drs_zones", "is_street_circuit", "wet_flag"]:
            master[col] = np.nan

    actual_grid = race_df[["season", "round", "raceId", "driverId", "actual_grid_position"]].copy()
    master = master.merge(actual_grid, on=["season", "round", "raceId", "driverId"], how="left")
    quali_grid = quali_df[["season", "round", "raceId", "driverId", "grid_position"]].copy()
    quali_grid = quali_grid.rename(columns={"grid_position": "quali_grid_position"})
    master = master.merge(quali_grid, on=["season", "round", "raceId", "driverId"], how="left")
    master["grid_position"] = master["actual_grid_position"].fillna(master["quali_grid_position"])
    master = master.drop(columns=["actual_grid_position", "quali_grid_position"])

    print(f"\n  Master table shape before imputation: {master.shape}")

    print("\nImputing missing values (unchanged logic)...")
    master = impute(master)

    feature_cols = [
        "season", "round", "raceId", "circuitId", "driverId", "constructorId",
        "elo_pre_race", "recent_form", "teammate_delta", "track_affinity",
        "pace_delta_vs_pole", "dnf_rate",
        "straight_line_exposure", "cornering_exposure",
        "full_throttle_pct", "avg_corner_speed", "lap_length_km",
        "drs_zones", "is_street_circuit", "wet_flag",
        "grid_position", "finishing_position", "status",
    ]
    feature_cols = [c for c in feature_cols if c in master.columns]
    master = master[feature_cols]

    print(f"\n  ✓ [v2] Master features: {master.shape[0]} rows × {master.shape[1]} cols")
    return master


if __name__ == "__main__":
    master = build_master_features_v2()

    print(f"\nSaving to {MASTER_FEATURES_V2_PATH} (local to this experiment folder)...")
    master.to_parquet(MASTER_FEATURES_V2_PATH, index=False)
    print("  ✓ Saved")

    # Quick invariant check: teammate_delta should be the only column
    # that meaningfully differs from the real master_features.parquet.
    real_path = os.path.join(_PROJECT_ROOT, "data", "processed", "master_features.parquet")
    if os.path.exists(real_path):
        real = pd.read_parquet(real_path)
        common_keys = ["season", "round", "driverId"]
        merged = master.merge(real, on=common_keys, suffixes=("_v2", "_orig"))
        print(f"\n{'='*60}")
        print("SANITY CHECK vs real master_features.parquet")
        print(f"{'='*60}")
        for col in ["elo_pre_race", "recent_form", "pace_delta_vs_pole"]:
            if f"{col}_v2" in merged.columns:
                max_diff = (merged[f"{col}_v2"] - merged[f"{col}_orig"]).abs().max()
                print(f"  {col}: max abs diff = {max_diff:.6f} "
                      f"({'✓ unchanged as expected' if max_diff < 1e-6 else '⚠ unexpectedly changed'})")
        if "teammate_delta_v2" in merged.columns:
            td_diff = (merged["teammate_delta_v2"] - merged["teammate_delta_orig"]).abs()
            print(f"  teammate_delta: mean abs diff = {td_diff.mean():.4f}, "
                  f"max abs diff = {td_diff.max():.4f}  "
                  f"(expected to differ -- this is the change under test)")
        print(f"{'='*60}\n")
