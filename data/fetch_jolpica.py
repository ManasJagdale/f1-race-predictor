"""
data/fetch_jolpica.py

Fetches and caches all historical F1 data we need from the Jolpica API
(the free Ergast successor at api.jolpi.ca).

What it pulls:
  1. Race results    — finishing positions, status (Finished / DNF reason),
                       constructor, for every race in FIRST_SEASON–LAST_SEASON
  2. Qualifying      — Q1/Q2/Q3 times + grid position per driver per race
  3. DNF history     — derived from race results (status != 'Finished')
  4. Race schedule    — round -> circuitId mapping for a season, including
                       rounds that haven't been raced yet (calendars are
                       published in advance)
  5. Single-round quali — qualifying results for one round without
                       fetching the whole season

Everything is saved as Parquet files in data/cache/ so subsequent runs
load from disk instantly without hitting the API again. (Schedule and
single-round quali fetches are the exception — they're used by
predict_upcoming.py for live/upcoming races and are NOT cached, since
their whole point is fetching current state, not historical data.)

Usage (from project root):
    python -m data.fetch_jolpica            # fetch everything
    python -m data.fetch_jolpica --refresh  # force re-fetch even if cached
"""

import time
import argparse
import requests
import pandas as pd

from config import (
    JOLPICA_BASE_URL,
    API_TIMEOUT,
    API_SLEEP,
    FIRST_SEASON,
    LAST_SEASON,
    RACE_RESULTS_CACHE,
    QUALI_RESULTS_CACHE,
    DNF_CACHE,
)


# ---------------------------------------------------------------------------
# Section 1: Core API fetcher with pagination
# ---------------------------------------------------------------------------
# Jolpica returns max 30 rows per page by default. For a full season of
# race results (20 races × 20 drivers = 400 rows) we need to paginate.
# We request 100 rows per page to minimise round trips.

ROWS_PER_PAGE = 100


def _get(endpoint: str) -> dict:
    """
    Make a single GET request to the Jolpica API.

    Args:
        endpoint: path after the base URL, e.g. '/2023/results.json?limit=100'

    Returns:
        Parsed JSON response as a dict.

    Raises:
        requests.HTTPError if the response status is not 2xx.
    """
    url = JOLPICA_BASE_URL + endpoint
    response = requests.get(url, timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _get_all_pages(endpoint_template: str) -> list:
    """
    Fetch all pages of a paginated Jolpica endpoint.

    Jolpica pagination works via ?limit=N&offset=M query params.
    We keep fetching until offset >= total rows reported by the API.

    Args:
        endpoint_template: endpoint path WITHOUT limit/offset params,
                           e.g. '/2023/results.json'

    Returns:
        List of all 'MRData' response dicts, one per page.
        The caller extracts the relevant nested data from each.
    """
    pages = []
    offset = 0

    while True:
        sep = "&" if "?" in endpoint_template else "?"
        endpoint = f"{endpoint_template}{sep}limit={ROWS_PER_PAGE}&offset={offset}"

        data = _get(endpoint)
        pages.append(data)

        mr = data["MRData"]
        total = int(mr["total"])
        returned = int(mr["limit"]) if "limit" in mr else ROWS_PER_PAGE
        offset += returned

        if offset >= total:
            break

        # Be polite — don't hammer the API
        time.sleep(API_SLEEP)

    return pages


# ---------------------------------------------------------------------------
# Section 2: Race results fetcher
# ---------------------------------------------------------------------------
# Jolpica endpoint: /{season}/results.json
# Returns every race in the season with all driver finishing positions.
#
# Key fields we extract:
#   season, round, raceId (circuit), driverId, constructorId,
#   position (numeric finish, None if DNF), positionOrder (always numeric),
#   status ('Finished' or DNF reason like 'Engine', 'Collision', etc.)

def _parse_race_results(pages: list) -> list[dict]:
    """Parse paginated race results pages into a flat list of row dicts."""
    rows = []
    for page in pages:
        races = page["MRData"]["RaceTable"]["Races"]
        for race in races:
            season = int(race["season"])
            rnd = int(race["round"])
            circuit_id = race["Circuit"]["circuitId"]

            for i, result in enumerate(race["Results"]):
                pos_str = result.get("position")
                position = float(pos_str) if pos_str and pos_str.isdigit() else None

                # positionText is 'R' for retired, 'D' for DSQ, 'W' for withdrawn
                # We use it to distinguish DNFs from classified finishers
                pos_text = result.get("positionText", "")

                rows.append({
                    "season":        season,
                    "round":         rnd,
                    "circuitId":     circuit_id,
                    "driverId":      result["Driver"]["driverId"],
                    "constructorId": result["Constructor"]["constructorId"],
                    "position":      position,
                    # Results come back sorted by finishing order already.
                    # Use enumeration index as positionOrder (0-based → 1-based)
                    "positionOrder": i + 1,
                    "status":        result["status"],
                    "raceId":        circuit_id,
                })
    return rows


def fetch_race_results(seasons: list[int]) -> pd.DataFrame:
    """
    Fetch race results for all given seasons from Jolpica.

    Args:
        seasons: list of season years, e.g. [2018, 2019, ..., 2024]

    Returns:
        DataFrame with columns:
            season, round, raceId, circuitId, driverId, constructorId,
            position, positionOrder, status
    """
    all_rows = []

    for season in seasons:
        print(f"    Fetching race results: {season}...")
        pages = _get_all_pages(f"/{season}/results.json")
        rows = _parse_race_results(pages)
        all_rows.extend(rows)
        print(f"      → {len(rows)} driver-race rows")
        time.sleep(API_SLEEP)

    df = pd.DataFrame(all_rows)
    print(f"  ✓ Race results: {len(df)} total rows across {df['season'].nunique()} seasons")
    return df


# ---------------------------------------------------------------------------
# Section 3: Qualifying results fetcher
# ---------------------------------------------------------------------------
# Jolpica endpoint: /{season}/qualifying.json
# Returns Q1, Q2, Q3 times and final grid position per driver.
#
# Not all sessions have Q2/Q3 (e.g. sprint weekends, wet sessions where
# drivers don't set times). We store NaN for missing times.

def _parse_qualifying(pages: list) -> list[dict]:
    """Parse paginated qualifying pages into a flat list of row dicts."""
    rows = []
    for page in pages:
        races = page["MRData"]["RaceTable"]["Races"]
        for race in races:
            season = int(race["season"])
            rnd = int(race["round"])
            circuit_id = race["Circuit"]["circuitId"]

            for result in race["QualifyingResults"]:
                rows.append({
                    "season":        season,
                    "round":         rnd,
                    "circuitId":     circuit_id,
                    "raceId":        circuit_id,
                    "driverId":      result["Driver"]["driverId"],
                    "constructorId": result["Constructor"]["constructorId"],
                    "grid_position": int(result["position"]),
                    "q1":            result.get("Q1"),   # time string or None
                    "q2":            result.get("Q2"),
                    "q3":            result.get("Q3"),
                })
    return rows


def fetch_qualifying(seasons: list[int]) -> pd.DataFrame:
    """
    Fetch qualifying results for all given seasons from Jolpica.

    Args:
        seasons: list of season years

    Returns:
        DataFrame with columns:
            season, round, raceId, circuitId, driverId, constructorId,
            grid_position, q1, q2, q3
    """
    all_rows = []

    for season in seasons:
        print(f"    Fetching qualifying: {season}...")
        pages = _get_all_pages(f"/{season}/qualifying.json")
        rows = _parse_qualifying(pages)
        all_rows.extend(rows)
        print(f"      → {len(rows)} driver-race rows")
        time.sleep(API_SLEEP)

    df = pd.DataFrame(all_rows)
    print(f"  ✓ Qualifying: {len(df)} total rows across {df['season'].nunique()} seasons")
    return df


# ---------------------------------------------------------------------------
# Section 3b: Race schedule and single-round qualifying
# ---------------------------------------------------------------------------
# Added to support predict_upcoming.py — predicting a race that may not
# have happened yet (and may not even have qualifying results yet).
#
# These are deliberately NOT cached to parquet like the functions above.
# fetch_race_results()/fetch_qualifying() cache because historical data
# never changes once a season is over — refetching would be wasteful.
# These two do the opposite job: they fetch CURRENT state (this week's
# calendar entry, this weekend's grid) for a race that's still ongoing
# or upcoming, so the whole point is hitting the live API each time
# rather than serving stale cached data.

def fetch_race_schedule(season: int) -> pd.DataFrame:
    """
    Fetch the full race schedule for a season — every round, including
    ones that haven't been raced yet. Works because F1 calendars are
    published well before the season starts; this hits /{season}.json
    rather than /{season}/results.json, so it doesn't depend on any
    results existing yet.

    Returns:
        DataFrame with columns: season, round, raceName, circuitId, date
    """
    print(f"    Fetching schedule: {season}...")
    pages = _get_all_pages(f"/{season}.json")

    rows = []
    for page in pages:
        races = page["MRData"]["RaceTable"]["Races"]
        for race in races:
            rows.append({
                "season":    int(race["season"]),
                "round":     int(race["round"]),
                "raceName":  race.get("raceName"),
                "circuitId": race["Circuit"]["circuitId"],
                "date":      race.get("date"),
            })

    df = pd.DataFrame(rows)
    print(f"  ✓ Schedule: {len(df)} races for season {season}")
    return df


def fetch_qualifying_for_round(season: int, round_num: int) -> pd.DataFrame:
    """
    Fetch qualifying results for a SINGLE round directly — doesn't
    require or trigger a full-season fetch. Returns an empty (but
    correctly-shaped) DataFrame if qualifying hasn't happened yet for
    this round rather than raising — Jolpica returns an empty Races
    list in that case, not an error, so the caller can check len()==0
    to detect "not available yet" and fall back to a manual grid.

    Returns:
        DataFrame with columns: season, round, raceId, circuitId,
        driverId, constructorId, grid_position, q1, q2, q3
    """
    print(f"    Fetching qualifying: {season} round {round_num}...")
    data = _get(f"/{season}/{round_num}/qualifying.json")
    races = data["MRData"]["RaceTable"]["Races"]

    empty_cols = ["season", "round", "raceId", "circuitId",
                  "driverId", "constructorId", "grid_position", "q1", "q2", "q3"]
    if not races:
        print(f"    ⚠ No qualifying data yet for {season} round {round_num} "
              f"(session hasn't happened, or hasn't been recorded yet)")
        return pd.DataFrame(columns=empty_cols)

    rows = _parse_qualifying([data])
    df = pd.DataFrame(rows)
    print(f"  ✓ Qualifying: {len(df)} driver rows")
    return df


# ---------------------------------------------------------------------------
# Section 4: DNF summary (derived, not a separate API call)
# ---------------------------------------------------------------------------
# We don't need a separate API call for DNFs — they're in the race results.
# status != 'Finished' and status != '+1 Lap' etc. = mechanical/crash DNF.
#
# We compute a per-constructor DNF rate rolling over the last 2 seasons.
# This is used in simulation.py to sample DNF probability per driver.
#
# 'Finished' and lapped cars ('+1 Lap', '+2 Laps', ...) are NOT DNFs.
# Anything else ('Engine', 'Gearbox', 'Collision', 'Accident', etc.) is.

_CLASSIFIED_STATUSES = {"Finished"}

def _is_lapped(status: str) -> bool:
    """Return True if the status indicates a lapped but classified finish."""
    return status.startswith("+") and "Lap" in status


def build_dnf_table(race_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-constructor DNF counts and rates from race results.

    Args:
        race_results_df: output of fetch_race_results()

    Returns:
        DataFrame with columns:
            season, constructorId, races, dnfs, dnf_rate
        One row per constructor per season.
        dnf_rate = dnfs / (total driver-race entries for that constructor)
    """
    df = race_results_df.copy()

    # Mark each entry as DNF or not
    df["is_dnf"] = ~(
        (df["status"] == "Finished") |
        df["status"].apply(_is_lapped)
    )

    summary = (
        df.groupby(["season", "constructorId"])
        .agg(
            entries=("driverId", "count"),
            dnfs=("is_dnf", "sum"),
        )
        .reset_index()
    )
    summary["dnf_rate"] = summary["dnfs"] / summary["entries"]

    print(f"  ✓ DNF table: {len(summary)} constructor-season rows")
    return summary


# ---------------------------------------------------------------------------
# Section 5: Main fetch pipeline
# ---------------------------------------------------------------------------

def fetch_all(refresh: bool = False) -> None:
    """
    Fetch all data from Jolpica and save to cache.

    Args:
        refresh: if True, re-fetch even if cache files already exist.
                 if False (default), skip fetch if cache exists.
    """
    seasons = list(range(FIRST_SEASON, LAST_SEASON + 1))
    print(f"\nFetching data for seasons: {seasons[0]}–{seasons[-1]}\n")

    # --- Race results ---
    if not refresh and _cache_exists(RACE_RESULTS_CACHE):
        print(f"  ✓ Race results: loaded from cache ({RACE_RESULTS_CACHE})")
        race_df = pd.read_parquet(RACE_RESULTS_CACHE)
    else:
        print("  Fetching race results from Jolpica API...")
        race_df = fetch_race_results(seasons)
        race_df.to_parquet(RACE_RESULTS_CACHE, index=False)
        print(f"    Saved → {RACE_RESULTS_CACHE}")

    # --- Qualifying ---
    if not refresh and _cache_exists(QUALI_RESULTS_CACHE):
        print(f"  ✓ Qualifying: loaded from cache ({QUALI_RESULTS_CACHE})")
    else:
        print("  Fetching qualifying from Jolpica API...")
        quali_df = fetch_qualifying(seasons)
        quali_df.to_parquet(QUALI_RESULTS_CACHE, index=False)
        print(f"    Saved → {QUALI_RESULTS_CACHE}")

    # --- DNF table (derived from race results, no extra API call) ---
    if not refresh and _cache_exists(DNF_CACHE):
        print(f"  ✓ DNF table: loaded from cache ({DNF_CACHE})")
    else:
        print("  Building DNF table from race results...")
        dnf_df = build_dnf_table(race_df)
        dnf_df.to_parquet(DNF_CACHE, index=False)
        print(f"    Saved → {DNF_CACHE}")

    print("\nAll data ready.\n")


def _cache_exists(path: str) -> bool:
    import os
    return os.path.exists(path)


def load_race_results() -> pd.DataFrame:
    """Load cached race results. Call fetch_all() first if cache doesn't exist."""
    return pd.read_parquet(RACE_RESULTS_CACHE)


def load_qualifying() -> pd.DataFrame:
    """Load cached qualifying results."""
    return pd.read_parquet(QUALI_RESULTS_CACHE)


def load_dnf_table() -> pd.DataFrame:
    """Load cached DNF table."""
    return pd.read_parquet(DNF_CACHE)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch F1 data from Jolpica API")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch from API even if cache files already exist",
    )
    args = parser.parse_args()
    fetch_all(refresh=args.refresh)