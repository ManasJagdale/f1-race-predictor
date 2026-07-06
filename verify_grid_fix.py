"""
verify_grid_fix.py

Sanity check for the actual_grid_position fix — confirms it correctly
distinguishes qualifying position from actual starting grid using a known
real-world case: Pierre Gasly, 2024 Azerbaijan GP (round 17).

Known facts (from F1 official results):
    Gasly qualified 13th in Q2, but was disqualified from qualifying for
    exceeding the fuel mass flow limit. He was permitted to race and
    actually started P18 (behind Ocon and Hamilton, who both started
    from the pit lane).

Expected result after the fix:
    quali_grid_position  == 13  (or NaN/DSQ marker, depending on how
                                  Jolpica records a qualifying DSQ)
    actual_grid_position == 18
    master_features.parquet's grid_position column should show 18,
    NOT 13 — that's the whole point of the fix.

Run from the project root (with venv active):
    python verify_grid_fix.py
"""

import pandas as pd

RACE_RESULTS_CACHE = "data/cache/race_results.parquet"
QUALI_RESULTS_CACHE = "data/cache/quali_results.parquet"
MASTER_FEATURES = "data/processed/master_features.parquet"

SEASON = 2024
ROUND = 17
DRIVER_ID = "gasly"  # Jolpica's driverId convention — adjust if this doesn't match

def main():
    print(f"Checking {DRIVER_ID} — season {SEASON}, round {ROUND} (2024 Azerbaijan GP)\n")

    # 1. Raw qualifying classification
    quali_df = pd.read_parquet(QUALI_RESULTS_CACHE)
    q_row = quali_df[
        (quali_df["season"] == SEASON) &
        (quali_df["round"] == ROUND) &
        (quali_df["driverId"] == DRIVER_ID)
    ]
    if q_row.empty:
        print("  ⚠ No qualifying row found — check driverId spelling in your cache.")
    else:
        print(f"  Qualifying classification (grid_position in quali cache): "
              f"{q_row['grid_position'].values[0]}")
        print("  Expected: 13 (this is his pre-DSQ qualifying result)")

    # 2. Raw race results — our new actual_grid_position field
    race_df = pd.read_parquet(RACE_RESULTS_CACHE)
    r_row = race_df[
        (race_df["season"] == SEASON) &
        (race_df["round"] == ROUND) &
        (race_df["driverId"] == DRIVER_ID)
    ]
    if r_row.empty:
        print("  ⚠ No race result row found — check driverId spelling in your cache.")
    elif "actual_grid_position" not in r_row.columns:
        print("  ⚠ actual_grid_position column not found — did you re-run "
              "fetch_jolpica.py --refresh after the fix?")
    else:
        print(f"\n  Actual starting grid (actual_grid_position in race cache): "
              f"{r_row['actual_grid_position'].values[0]}")
        print("  Expected: 18 (post-DSQ, his real starting position)")

    # 3. Final master_features.parquet — what the model actually trains on
    master_df = pd.read_parquet(MASTER_FEATURES)
    m_row = master_df[
        (master_df["season"] == SEASON) &
        (master_df["round"] == ROUND) &
        (master_df["driverId"] == DRIVER_ID)
    ]
    if m_row.empty:
        print("  ⚠ No row found in master_features.parquet — check driverId spelling.")
    else:
        print(f"\n  FINAL grid_position fed to the model: "
              f"{m_row['grid_position'].values[0]}")
        print("  Expected: 18.0 — if this shows 13.0, the fix isn't taking "
              "effect and the old qualifying-position bug is still live.")

    print("\nDone. Compare the three numbers above against the 'Expected' lines.")


if __name__ == "__main__":
    main()
