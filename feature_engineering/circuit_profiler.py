"""
feature_engineering/circuit_profiler.py

One-time circuit analysis tool. Pulls FastF1 telemetry for a reference
lap at each circuit and classifies each sector by character (straight-
dominant, cornering-dominant, mixed) based on real throttle/speed data
— not assumption.

Output: data/cache/circuit_profiles.xlsx
One row per circuit (or circuit-season if layout changed), columns:
    circuitId, season, sector1_character, sector1_avg_throttle,
    sector1_avg_speed, sector2_character, ..., sector3_character, ...,
    notes (layout changes, manual flags)

Run once. Re-run only for a specific circuit if its layout changes.

Usage:
    python -m feature_engineering.circuit_profiler --circuit monza --season 2026
    python -m feature_engineering.circuit_profiler --all   # all circuits in dataset
"""

import os
import sys
import argparse
import pandas as pd
import fastf1

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CACHE_DIR

fastf1.Cache.enable_cache(os.path.join(CACHE_DIR, "fastf1"))

# Saved here rather than data/cache — track layouts barely change, so this
# is a long-lived reference table, not a disposable cache artifact.
OUTPUT_DIR = r"D:\Projects\F1 Predictor(Actual prediction)\files\Some data to save"
CIRCUIT_PROFILES_PATH = os.path.join(OUTPUT_DIR, "circuit_profiles.xlsx")

# Per-SECOND thresholds — classify each 1-second window first, then
# derive the sector's overall character from the mix of seconds it contains.
FULL_THROTTLE_SECOND   = 95   # a second averaging >= this % throttle = "full throttle"
BRAKING_SECOND_THROTTLE = 30   # a second averaging <= this % throttle = "braking/cornering"
# Between the two = "transition" (lift-and-coast, partial throttle corners)

# Sector-level classification — based on % of seconds that are full-throttle
SECTOR_STRAIGHT_PCT = 60   # if >=60% of seconds in a sector are full-throttle → straight_dominant
SECTOR_CORNER_PCT   = 40   # if >=40% of seconds are braking/cornering → cornering_dominant
# Otherwise → mixed


def classify_second(avg_throttle: float) -> str:
    """Classify a single 1-second window based on its average throttle."""
    if avg_throttle >= FULL_THROTTLE_SECOND:
        return "full_throttle"
    elif avg_throttle <= BRAKING_SECOND_THROTTLE:
        return "braking_cornering"
    return "transition"


def bin_by_second(sector_data: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket telemetry rows into 1-second windows and compute the average
    throttle/speed within each window. Returns one row per second.
    """
    if sector_data.empty:
        return pd.DataFrame()

    # ElapsedTime is a Timedelta — convert to whole seconds for binning
    sector_data = sector_data.copy()
    sector_data["second_bin"] = sector_data["ElapsedTime"].dt.total_seconds().astype(int)

    per_second = (
        sector_data.groupby("second_bin")
        .agg(avg_throttle=("Throttle", "mean"), avg_speed=("Speed", "mean"))
        .reset_index()
    )
    per_second["state"] = per_second["avg_throttle"].apply(classify_second)
    return per_second


def classify_sector(per_second: pd.DataFrame) -> str:
    """
    Classify a sector based on the proportion of its 1-second windows
    that are full-throttle vs braking/cornering.
    """
    if per_second.empty:
        return None

    total = len(per_second)
    pct_full_throttle = (per_second["state"] == "full_throttle").sum() / total * 100
    pct_braking       = (per_second["state"] == "braking_cornering").sum() / total * 100

    if pct_full_throttle >= SECTOR_STRAIGHT_PCT:
        return "straight_dominant"
    elif pct_braking >= SECTOR_CORNER_PCT:
        return "cornering_dominant"
    return "mixed"


def find_round(season: int, circuit_id: str) -> int:
    """
    Look up the round number for a given season + circuitId using the
    cached race results. Needed because FastF1's get_session() wants
    a round number, but we key everything else off circuitId.
    """
    from data.fetch_jolpica import load_race_results
    race_df = load_race_results()
    match = race_df[
        (race_df["season"] == season) & (race_df["circuitId"] == circuit_id)
    ]
    if match.empty:
        raise ValueError(f"No race found for circuitId='{circuit_id}' in season={season}")
    return int(match["round"].iloc[0])


def profile_circuit(season: int, round_num: int, circuit_id: str) -> dict:
    """
    Pull a representative qualifying lap (fastest lap of the session)
    and compute real sector characteristics from telemetry.
    """
    session = fastf1.get_session(season, round_num, "Q")
    session.load()

    fastest_lap = session.laps.pick_fastest()
    telemetry   = fastest_lap.get_telemetry()

    # Split telemetry into 3 sectors using the lap's own sector time markers
    sector1_end = fastest_lap["Sector1Time"]
    sector2_end = fastest_lap["Sector1Time"] + fastest_lap["Sector2Time"]

    telemetry["ElapsedTime"] = telemetry["Time"] - telemetry["Time"].iloc[0]

    s1 = telemetry[telemetry["ElapsedTime"] <= sector1_end]
    s2 = telemetry[(telemetry["ElapsedTime"] > sector1_end) & (telemetry["ElapsedTime"] <= sector2_end)]
    s3 = telemetry[telemetry["ElapsedTime"] > sector2_end]

    profile = {"circuitId": circuit_id, "season": season}

    for i, sector_data in enumerate([s1, s2, s3], start=1):
        if sector_data.empty:
            print(f"    ⚠ Sector {i} has no telemetry rows — check sector time parsing")
            profile[f"sector{i}_character"]       = None
            profile[f"sector{i}_avg_throttle"]     = None
            profile[f"sector{i}_avg_speed"]        = None
            profile[f"sector{i}_top_speed"]        = None
            profile[f"sector{i}_brake_zones"]      = None
            profile[f"sector{i}_n_seconds"]        = None
            profile[f"sector{i}_pct_full_throttle"] = None
            profile[f"sector{i}_pct_braking"]       = None
            profile[f"sector{i}_per_second_states"] = None
            continue

        per_second = bin_by_second(sector_data)

        avg_throttle = sector_data["Throttle"].mean()
        avg_speed    = sector_data["Speed"].mean()
        top_speed    = sector_data["Speed"].max()
        brake_zones  = (sector_data["Brake"].astype(int).diff() == 1).sum()

        total = len(per_second)
        pct_full_throttle = (per_second["state"] == "full_throttle").sum() / total * 100
        pct_braking       = (per_second["state"] == "braking_cornering").sum() / total * 100

        profile[f"sector{i}_character"]        = classify_sector(per_second)
        profile[f"sector{i}_avg_throttle"]      = round(avg_throttle, 1)
        profile[f"sector{i}_avg_speed"]         = round(avg_speed, 1)
        profile[f"sector{i}_top_speed"]         = round(top_speed, 1)
        profile[f"sector{i}_brake_zones"]       = int(brake_zones)
        profile[f"sector{i}_n_seconds"]         = total
        profile[f"sector{i}_pct_full_throttle"] = round(pct_full_throttle, 1)
        profile[f"sector{i}_pct_braking"]       = round(pct_braking, 1)
        # Compact string like "FFFTTBBFFFF" for a quick visual read,
        # F=full throttle, T=transition, B=braking/cornering
        state_map = {"full_throttle": "F", "transition": "T", "braking_cornering": "B"}
        profile[f"sector{i}_per_second_states"] = "".join(
            state_map[s] for s in per_second["state"]
        )

    return profile


def build_all_profiles(race_calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Run profile_circuit for every unique circuit in the dataset.
    Uses the most recent season available for each circuit (current layout).
    """
    profiles = []
    latest_per_circuit = (
        race_calendar.sort_values("season")
        .groupby("circuitId")
        .last()
        .reset_index()
    )

    for _, row in latest_per_circuit.iterrows():
        print(f"  Profiling {row['circuitId']} ({row['season']})...")
        try:
            profile = profile_circuit(row["season"], row["round"], row["circuitId"])
            profiles.append(profile)
        except Exception as e:
            print(f"    ⚠ Failed: {e}")

    return pd.DataFrame(profiles)


def save_single_profile(profile: dict) -> None:
    """
    Save or update a single circuit's profile in the existing spreadsheet.
    If the circuit already has a row, it's replaced (e.g. after a layout
    change). Otherwise the row is appended. Used by --circuit/--season
    when you want the result to persist, not just print to terminal.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(CIRCUIT_PROFILES_PATH):
        existing = pd.read_excel(CIRCUIT_PROFILES_PATH)
        existing = existing[existing["circuitId"] != profile["circuitId"]]
        updated = pd.concat([existing, pd.DataFrame([profile])], ignore_index=True)
    else:
        updated = pd.DataFrame([profile])

    updated = updated.sort_values("circuitId").reset_index(drop=True)
    updated.to_excel(CIRCUIT_PROFILES_PATH, index=False)
    print(f"\n✓ Saved/updated '{profile['circuitId']}' in {CIRCUIT_PROFILES_PATH}")


def print_profile(profile: dict) -> None:
    """Pretty-print a single circuit profile to terminal for quick sanity check."""
    print(f"\n--- {profile['circuitId']} ({profile['season']}) ---")
    for i in range(1, 4):
        char    = profile.get(f"sector{i}_character")
        thr     = profile.get(f"sector{i}_avg_throttle")
        spd     = profile.get(f"sector{i}_avg_speed")
        top     = profile.get(f"sector{i}_top_speed")
        brakes  = profile.get(f"sector{i}_brake_zones")
        pct_ft  = profile.get(f"sector{i}_pct_full_throttle")
        pct_br  = profile.get(f"sector{i}_pct_braking")
        states  = profile.get(f"sector{i}_per_second_states")
        n_sec   = profile.get(f"sector{i}_n_seconds")
        print(f"  Sector {i}: {char:<20} avg_throttle={thr}%  avg_speed={spd}km/h  "
              f"top_speed={top}km/h  brake_zones={brakes}")
        print(f"             {n_sec}s long | {pct_ft}% full-throttle | {pct_br}% braking/cornering")
        print(f"             second-by-second: {states}  (F=full throttle, T=transition, B=braking/cornering)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Profile every circuit in the dataset")
    parser.add_argument("--circuit", type=str, help="Single circuitId to profile, e.g. 'monza'")
    parser.add_argument("--season", type=int, help="Season for single-circuit profiling, e.g. 2026")
    parser.add_argument("--save", action="store_true",
                         help="When used with --circuit/--season, also save the result "
                              "to circuit_profiles.xlsx (not just print to terminal)")
    args = parser.parse_args()

    if args.all:
        from data.fetch_jolpica import load_race_results
        race_df = load_race_results()
        calendar = race_df[["season", "round", "circuitId"]].drop_duplicates()
        profiles_df = build_all_profiles(calendar)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        profiles_df.to_excel(CIRCUIT_PROFILES_PATH, index=False)
        print(f"\n✓ Saved {len(profiles_df)} circuit profiles to {CIRCUIT_PROFILES_PATH}")

    elif args.circuit and args.season:
        print(f"Profiling {args.circuit} ({args.season})...")
        round_num = find_round(args.season, args.circuit)
        print(f"  Found round {round_num}")
        profile = profile_circuit(args.season, round_num, args.circuit)
        print_profile(profile)
        if args.save:
            save_single_profile(profile)

    else:
        print("Usage:")
        print("  python -m feature_engineering.circuit_profiler --all")
        print("  python -m feature_engineering.circuit_profiler --circuit monza --season 2026")