"""Central configuration for the F1 Race Outcome Predictor.

All hardcoded constants, paths, and toggles live here.
Import this in every module instead of scattering magic numbers.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR        = os.path.join(ROOT_DIR, "data")
CACHE_DIR       = os.path.join(DATA_DIR, "cache")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR      = os.path.join(ROOT_DIR, "output")
FASTF1_CACHE    = os.path.join(ROOT_DIR, ".fastf1_cache")

# Create directories if they don't exist yet
for _dir in (CACHE_DIR, PROCESSED_DIR, OUTPUT_DIR, FASTF1_CACHE):
    os.makedirs(_dir, exist_ok=True)

# ── Data range ───────────────────────────────────────────────────────────────

FIRST_SEASON = 2018    # Earliest season to pull from Jolpica
LAST_SEASON  = 2026    # Most recent completed season

TRAIN_SEASONS = list(range(2020, 2023))   # 2020-2022 inclusive
TEST_SEASONS  = list(range(2023, 2027))   # 2023-2026 inclusive

# ── Jolpica API ───────────────────────────────────────────────────────────────

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
API_TIMEOUT      = 30    # seconds per request
API_SLEEP        = 0.3   # seconds between requests (be polite to the API)

# ── Elo settings ──────────────────────────────────────────────────────────────

ELO_INITIAL       = 1500.0   # Starting rating for every driver
ELO_K_BASE        = 32.0     # Base k-factor (how fast ratings move)
ELO_HOME_ADVANTAGE = 0.0     # F1 has no home advantage (neutral circuits)

# ── Recent form window ────────────────────────────────────────────────────────

FORM_WINDOW = 5    # Number of recent races to average for form feature

# ── Monte Carlo ───────────────────────────────────────────────────────────────

NUM_SIMULATIONS = 10_000

# ── Cache filenames ───────────────────────────────────────────────────────────

RACE_RESULTS_CACHE   = os.path.join(CACHE_DIR, "race_results.parquet")
QUALI_RESULTS_CACHE  = os.path.join(CACHE_DIR, "quali_results.parquet")
DNF_CACHE            = os.path.join(CACHE_DIR, "dnf_history.parquet")
TRACK_FEATURES_CACHE = os.path.join(CACHE_DIR, "track_features.parquet")
DRIVER_RATINGS_CACHE = os.path.join(PROCESSED_DIR, "driver_ratings.parquet")
