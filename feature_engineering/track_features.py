"""
feature_engineering/track_features.py

Computes per-circuit track characteristics using FastF1 telemetry data.

Features extracted per circuit:
  - full_throttle_pct   : % of lap distance at full throttle (>= 98%)
                          High = power circuit (Monza). Low = technical (Monaco).
  - avg_corner_speed    : mean speed (kph) through corners (throttle < 40%)
                          High = high-speed aero circuit. Low = slow twisty circuit.
  - lap_length_km       : lap distance in km (from telemetry distance)
  - drs_zones           : number of DRS activation zones
  - is_street_circuit   : 1 if street circuit, 0 if permanent track
  - wet_flag            : 1 if qualifying session was wet, 0 if dry
                          (per race, not per circuit — same circuit can vary)

Strategy:
  - We pick ONE representative qualifying session per circuit per season
    (the most recent dry session, to get clean telemetry)
  - FastF1 loads the fastest qualifying lap for that circuit
  - We extract telemetry at ~4Hz and compute the features from raw speed/throttle
  - Results are cached to TRACK_FEATURES_CACHE as Parquet

Why qualifying and not race?
  - Qualifying laps are clean: no following, no fuel load, no tyre saving
  - One driver pushes 100% the whole lap — perfect for measuring track character
  - Race laps have traffic, safety cars, lift-and-coast — add noise

Usage (from project root):
    python -m feature_engineering.track_features
    python -m feature_engineering.track_features --refresh
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import fastf1

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FIRST_SEASON,
    LAST_SEASON,
    FASTF1_CACHE,
    TRACK_FEATURES_CACHE,
    API_SLEEP,
)

# ── FastF1 cache setup ────────────────────────────────────────────────────────
# FastF1 caches downloaded session data locally so repeat runs are instant.
# We use the .fastf1_cache/ directory defined in config.py.
fastf1.Cache.enable_cache(FASTF1_CACHE)
fastf1.set_log_level("ERROR")   # suppress verbose INFO/WARNING output


# ── Jolpica circuitId → FastF1 EventName mapping ─────────────────────────────
# Jolpica uses snake_case circuit IDs; FastF1 uses full event names.
# This mapping bridges the two. Extend if new circuits are added.
CIRCUIT_TO_EVENT: dict[str, str] = {
    "albert_park":   "Australian Grand Prix",
    "americas":      "United States Grand Prix",
    "bahrain":       "Bahrain Grand Prix",
    "baku":          "Azerbaijan Grand Prix",
    "catalunya":     "Spanish Grand Prix",
    "hungaroring":   "Hungarian Grand Prix",
    "imola":         "Emilia Romagna Grand Prix",
    "interlagos":    "São Paulo Grand Prix",
    "jeddah":        "Saudi Arabian Grand Prix",
    "las_vegas":     "Las Vegas Grand Prix",
    "losail":        "Qatar Grand Prix",
    "marina_bay":    "Singapore Grand Prix",
    "miami":         "Miami Grand Prix",
    "monaco":        "Monaco Grand Prix",
    "monza":         "Italian Grand Prix",
    "red_bull_ring": "Austrian Grand Prix",
    "rodriguez":     "Mexico City Grand Prix",
    "sakhir":        "Bahrain Grand Prix",   # pre-2021 name
    "shanghai":      "Chinese Grand Prix",
    "silverstone":   "British Grand Prix",
    "spa":           "Belgian Grand Prix",
    "suzuka":        "Japanese Grand Prix",
    "villeneuve":    "Canadian Grand Prix",
    "yas_marina":    "Abu Dhabi Grand Prix",
    "zandvoort":     "Dutch Grand Prix",
    # 2025/2026 additions
    "madring":       "Spanish Grand Prix",   # new Madrid circuit (Jolpica circuitId is 'madring', not 'madrid')
}

# Street circuits — different tyre behaviour, safety car rates, overtaking
STREET_CIRCUITS: set[str] = {
    "baku", "monaco", "marina_bay", "jeddah", "las_vegas", "villeneuve", "madring",
}


# ── Telemetry feature extraction ──────────────────────────────────────────────

def _extract_telemetry_features(tel: pd.DataFrame) -> dict:
    """
    Compute track characteristics from a single fastest-lap telemetry DataFrame.

    FastF1 telemetry columns we use:
        Speed    : speed in kph (float)
        Throttle : throttle input 0-100 (float)
        Distance : cumulative distance from lap start (float, metres)

    Args:
        tel: telemetry DataFrame from lap.get_telemetry()

    Returns:
        dict with keys: full_throttle_pct, avg_corner_speed, lap_length_km
    """
    if tel is None or len(tel) == 0:
        return {"full_throttle_pct": np.nan, "avg_corner_speed": np.nan,
                "lap_length_km": np.nan}

    # Full throttle: throttle >= 98% (not 100 — drivers rarely pin it exactly)
    full_throttle_mask = tel["Throttle"] >= 98
    full_throttle_pct  = full_throttle_mask.sum() / len(tel) * 100

    # Corner speed: sections where driver is OFF throttle (< 40%)
    # This isolates braking zones + mid-corner where car aero matters most
    corner_mask  = tel["Throttle"] < 40
    corner_speed = tel.loc[corner_mask, "Speed"].mean() if corner_mask.any() else np.nan

    # Lap length: max cumulative distance (metres → km)
    lap_length_km = tel["Distance"].max() / 1000 if "Distance" in tel.columns else np.nan

    return {
        "full_throttle_pct": round(full_throttle_pct, 2),
        "avg_corner_speed":  round(corner_speed, 2),
        "lap_length_km":     round(lap_length_km, 3),
    }


def _count_drs_zones(session) -> int:
    """
    Count DRS zones. DRS was abolished from the 2026 season onwards.
    """
    # DRS removed in 2026
    if session.event["EventDate"].year >= 2026:
        return 0

    DRS_ZONES = {
        "monaco": 1, "silverstone": 2, "monza": 2, "spa": 2,
        "albert_park": 4, "americas": 2, "bahrain": 3, "baku": 2,
        "catalunya": 2, "hungaroring": 2, "imola": 2, "interlagos": 2,
        "jeddah": 3, "las_vegas": 2, "losail": 3, "marina_bay": 3,
        "miami": 3, "red_bull_ring": 3, "rodriguez": 2, "sakhir": 3,
        "shanghai": 2, "suzuka": 2, "villeneuve": 2, "yas_marina": 2,
        "zandvoort": 2,
    }
    # Match by location name
    location = session.event.get("Location", "").lower().replace(" ", "_")
    return DRS_ZONES.get(location, 2)


def _is_wet_session(session) -> int:
    """
    Determine if the qualifying session was wet from weather data.
    Returns 1 if wet/mixed, 0 if dry.
    Falls back to 0 if weather data unavailable.
    """
    try:
        weather = session.weather_data
        if weather is not None and len(weather) > 0 and "Rainfall" in weather.columns:
            return int(weather["Rainfall"].any())
    except Exception:
        pass
    return 0


# ── Session loader ────────────────────────────────────────────────────────────

def _load_qualifying_session(season: int, event_name: str):
    """
    Load a FastF1 qualifying session for a given season and event name.

    Args:
        season:     e.g. 2024
        event_name: e.g. "British Grand Prix"

    Returns:
        Loaded FastF1 Session object, or None if loading fails.
    """
    try:
        session = fastf1.get_session(season, event_name, "Q")
        # Load laps + telemetry + weather. This is the slow step (~10-30s per session).
        # FastF1 caches the result so subsequent runs load from disk instantly.
        session.load(laps=True, telemetry=True, weather=True, messages=False)
        return session
    except Exception as e:
        print(f"      ✗ Failed to load {event_name} {season}: {e}")
        return None


# ── Per-circuit feature builder ───────────────────────────────────────────────

def _compute_circuit_features(
    circuit_id: str,
    event_name: str,
    season: int,
) -> dict | None:
    """
    Compute all track features for one circuit in one season.

    Returns a dict of features, or None if the session fails to load.
    """
    print(f"      Loading {event_name} {season} qualifying...")

    session = _load_qualifying_session(season, event_name)
    if session is None:
        return None

    # Get the fastest qualifying lap across all drivers
    try:
        laps = session.laps
        if laps is None or len(laps) == 0:
            print(f"      ✗ No laps found for {event_name} {season}")
            return None

        # Pick single fastest lap of the whole session
        fastest_lap = laps.pick_fastest()
        tel = fastest_lap.get_telemetry()
    except Exception as e:
        print(f"      ✗ Could not get telemetry for {event_name} {season}: {e}")
        return None

    # Extract features
    tel_features  = _extract_telemetry_features(tel)
    drs_zones     = _count_drs_zones(session)
    wet_flag      = _is_wet_session(session)
    is_street     = 1 if circuit_id in STREET_CIRCUITS else 0

    return {
        "circuitId":         circuit_id,
        "season":            season,
        "event_name":        event_name,
        "full_throttle_pct": tel_features["full_throttle_pct"],
        "avg_corner_speed":  tel_features["avg_corner_speed"],
        "lap_length_km":     tel_features["lap_length_km"],
        "drs_zones":         drs_zones,
        "is_street_circuit": is_street,
        "wet_flag":          wet_flag,
    }


# ── Master builder ────────────────────────────────────────────────────────────

def build_track_features(
    circuit_ids: list[str] | None = None,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build track features for all circuits across all seasons.

    We process every (circuit, season) pair that appears in the race calendar.
    This gives us per-season track features, which matters because:
      - Wet flag varies year to year at the same circuit
      - Track resurfacing can affect corner speeds (e.g. Silverstone 2020)
      - DRS zone counts occasionally change between seasons

    Args:
        circuit_ids: list of Jolpica circuit IDs to process.
                     Defaults to all circuits in CIRCUIT_TO_EVENT.
        seasons:     list of season years. Defaults to FIRST_SEASON–LAST_SEASON.

    Returns:
        DataFrame with one row per (circuit, season), columns:
            circuitId, season, event_name, full_throttle_pct,
            avg_corner_speed, lap_length_km, drs_zones,
            is_street_circuit, wet_flag
    """
    if circuit_ids is None:
        circuit_ids = list(CIRCUIT_TO_EVENT.keys())
    if seasons is None:
        seasons = list(range(FIRST_SEASON, LAST_SEASON + 1))

    records = []
    total   = len(circuit_ids) * len(seasons)
    done    = 0

    print(f"\nFetching track features for {len(circuit_ids)} circuits × "
          f"{len(seasons)} seasons = up to {total} sessions\n"
          f"(FastF1 caches each session locally — slow first run, instant after)\n")

    for season in seasons:
        # Get the circuits that actually appeared in this season's calendar
        try:
            schedule = fastf1.get_event_schedule(season, include_testing=False)
            season_events = set(schedule["EventName"].str.strip())
        except Exception as e:
            print(f"  ✗ Could not get schedule for {season}: {e}")
            continue

        print(f"  Season {season} — {len(season_events)} events")

        for circuit_id in circuit_ids:
            event_name = CIRCUIT_TO_EVENT.get(circuit_id)
            if event_name is None:
                continue   # circuit not in mapping — skip

            if event_name not in season_events:
                continue   # this circuit not on the calendar this season — skip

            done += 1
            features = _compute_circuit_features(circuit_id, event_name, season)

            if features is not None:
                records.append(features)
                print(f"      ✓ {circuit_id}: throttle={features['full_throttle_pct']:.1f}% "
                      f"corner_speed={features['avg_corner_speed']:.1f}kph "
                      f"len={features['lap_length_km']:.2f}km "
                      f"wet={features['wet_flag']}")
            else:
                # Insert a row of NaNs so we know this (circuit, season) was attempted
                records.append({
                    "circuitId": circuit_id, "season": season,
                    "event_name": event_name,
                    "full_throttle_pct": np.nan, "avg_corner_speed": np.nan,
                    "lap_length_km": np.nan, "drs_zones": 0,
                    "is_street_circuit": 1 if circuit_id in STREET_CIRCUITS else 0,
                    "wet_flag": 0,
                })

            time.sleep(API_SLEEP)   # polite pause between sessions

    df = pd.DataFrame(records)
    print(f"\n✓ Track features: {len(df)} circuit-season rows, "
          f"{df['circuitId'].nunique()} unique circuits")
    return df


# ── Public load helper ────────────────────────────────────────────────────────

def load_track_features() -> pd.DataFrame:
    """
    Load cached track features. Call build_and_save() first if cache doesn't exist.

    For any circuit-season pair with NaN telemetry features (session failed to load),
    we forward-fill using the most recent available data for that circuit.
    This ensures the ML model always has a value.
    """
    df = pd.read_parquet(TRACK_FEATURES_CACHE)

    # Forward-fill NaN telemetry per circuit across seasons
    tel_cols = ["full_throttle_pct", "avg_corner_speed", "lap_length_km", "drs_zones"]
    df = df.sort_values(["circuitId", "season"])
    df[tel_cols] = df.groupby("circuitId")[tel_cols].ffill().bfill()

    return df


def build_and_save(refresh: bool = False) -> pd.DataFrame:
    """
    Build track features, resuming from existing cache if available.
    Only fetches circuit-season pairs not already in the cache.
    """
    existing = pd.DataFrame()
    if not refresh and os.path.exists(TRACK_FEATURES_CACHE):
        existing = pd.read_parquet(TRACK_FEATURES_CACHE)
        print(f"✓ Loaded {len(existing)} existing rows from cache")

    # Build the full set of (circuit, season) pairs needed
    all_circuits = list(CIRCUIT_TO_EVENT.keys())
    all_seasons  = list(range(FIRST_SEASON, LAST_SEASON + 1))

    # Find which pairs are missing
    if not existing.empty:
        done = set(zip(existing["circuitId"], existing["season"]))
        missing_circuits = []
        missing_seasons  = []
        for season in all_seasons:
            for circuit in all_circuits:
                if (circuit, season) not in done:
                    if circuit not in missing_circuits:
                        missing_circuits.append(circuit)
                    if season not in missing_seasons:
                        missing_seasons.append(season)
        print(f"  {len(done)} pairs already cached, fetching remaining...")
    else:
        missing_circuits = all_circuits
        missing_seasons  = all_seasons

    if not missing_circuits or not missing_seasons:
        print("✓ All circuit-season pairs already cached")
        return load_track_features()

    new_df = build_track_features(
        circuit_ids=missing_circuits,
        seasons=missing_seasons
    )

    # Merge new rows with existing
    combined = pd.concat([existing, new_df], ignore_index=True)
    # Drop duplicates keeping latest
    combined = combined.drop_duplicates(
        subset=["circuitId", "season"], keep="last"
    )
    combined.to_parquet(TRACK_FEATURES_CACHE, index=False)
    print(f"  Saved → {TRACK_FEATURES_CACHE}")
    return load_track_features()


# ── Sanity check ──────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame) -> None:
    """Print a summary table of track features for the most recent season."""
    last_season = df["season"].max()
    subset = df[df["season"] == last_season].sort_values(
        "full_throttle_pct", ascending=False
    ).reset_index(drop=True)

    print(f"\n--- Track characteristics ({last_season}) ---")
    print(f"{'Pos':<4} {'Circuit':<20} {'Throttle%':>10} {'Corner kph':>11} "
          f"{'Lap km':>7} {'DRS':>4} {'Street':>7} {'Wet':>4}")
    print("-" * 70)
    for i, row in subset.iterrows():
        throttle = f"{row['full_throttle_pct']:.1f}%" if pd.notna(row['full_throttle_pct']) else "  —"
        corner   = f"{row['avg_corner_speed']:.1f}" if pd.notna(row['avg_corner_speed']) else "  —"
        laplen   = f"{row['lap_length_km']:.2f}" if pd.notna(row['lap_length_km']) else " —"
        print(f"{i+1:<4} {row['circuitId']:<20} {throttle:>10} {corner:>11} "
              f"{laplen:>7} {int(row['drs_zones']):>4} {int(row['is_street_circuit']):>7} "
              f"{int(row['wet_flag']):>4}")

    # Sanity checks
    print("\nChecks:")
    throttle_range = df["full_throttle_pct"].dropna()
    assert (throttle_range >= 0).all() and (throttle_range <= 100).all(), \
        "full_throttle_pct out of range"
    print("  ✓ full_throttle_pct in [0, 100]")

    corner_range = df["avg_corner_speed"].dropna()
    assert (corner_range > 0).all(), "avg_corner_speed should be positive"
    print("  ✓ avg_corner_speed > 0 for all non-NaN rows")

    # Monza should be highest throttle, Monaco should be lowest
    monza  = df[(df["circuitId"] == "monza")  & (df["season"] == last_season)]
    monaco = df[(df["circuitId"] == "monaco") & (df["season"] == last_season)]
    if not monza.empty and not monaco.empty:
        monza_t  = monza["full_throttle_pct"].iloc[0]
        monaco_t = monaco["full_throttle_pct"].iloc[0]
        if pd.notna(monza_t) and pd.notna(monaco_t):
            assert monza_t > monaco_t, \
                f"Monza throttle ({monza_t:.1f}%) should exceed Monaco ({monaco_t:.1f}%)"
            print(f"  ✓ Monza ({monza_t:.1f}%) > Monaco ({monaco_t:.1f}%) — power vs technical")

    print("\nAll checks passed ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build F1 track features using FastF1")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-fetch from FastF1 even if cache exists"
    )
    parser.add_argument(
        "--circuit", type=str, default=None,
        help="Process a single circuit only (e.g. 'silverstone')"
    )
    parser.add_argument(
        "--season", type=int, default=None,
        help="Process a single season only (e.g. 2024)"
    )
    args = parser.parse_args()

    if args.circuit or args.season:
        # Targeted run for testing
        circuits = [args.circuit] if args.circuit else list(CIRCUIT_TO_EVENT.keys())
        seasons  = [args.season]  if args.season  else list(range(FIRST_SEASON, LAST_SEASON + 1))
        df = build_track_features(circuit_ids=circuits, seasons=seasons)
    else:
        df = build_and_save(refresh=args.refresh)

    if len(df) > 0:
        _print_summary(df)
