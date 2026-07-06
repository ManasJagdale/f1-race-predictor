"""
car_rankings.py

Ranks every constructor's current car pace on a 0-10 scale, using the
SAME pace_delta_vs_pole rolling feature race_model.py and simulation.py
already use as their car-performance input. 0 = the slowest car in the
field right now, 10 = the fastest (right at pole pace).

WHY 0-10 AND NOT A RAW TIME GAP:
pace_delta_vs_pole is expressed in seconds off pole -- accurate, but
not intuitive at a glance ("+1.494s" vs "+0.000s" doesn't immediately
read as "this team is mid-pack, that one's on pole"). This rescales
the SAME underlying number onto a fixed 0-10 scale anchored at
0.0s = 10 (the model's own definition of "as fast as it currently
gets"), so scores are directly comparable across seasons/rounds
without re-deriving the scale each time.

SCORING:
    score = 10 * max(0, 1 - pace_delta_vs_pole / CAP_SECONDS)

CAP_SECONDS is the pace gap that maps to a score of 0.0. Defaults to
4.0s -- comfortably past the largest gap actually observed in this
project's own cached history (Aston Martin, 2026: +3.735s, per
Final_Project_History.md section 7.1) -- so the scale reflects a real,
historically-slow car rather than an arbitrary round number, and won't
need re-anchoring every time a new backmarker briefly gets even slower.
Override with --cap if you want a different fixed scale.

TWO MODES:

  --historical   Rank cars as of a specific ALREADY-RACED round, using
                 the pace_delta_vs_pole values already sitting in
                 master_features.parquet. Fast -- no live API calls.
                 Omit --round to use the latest round available for
                 that season.

  --upcoming     Rank cars going into a round that hasn't been raced
                 yet (or whose results aren't in master_features.parquet
                 yet), by live-fetching that round's qualifying times
                 and recomputing the rolling pace delta the same way
                 predict_upcoming.py does. Reuses compute_pace_delta()
                 from feature_engineering/car_performance.py directly
                 -- not a reimplementation, so it can't drift from what
                 the live prediction pipeline actually uses.

Usage:
    python car_rankings.py --historical --season 2026 --round 8
    python car_rankings.py --historical --season 2026
    python car_rankings.py --upcoming --season 2026 --round 9
    python car_rankings.py --historical --season 2026 --cap 3.0
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROCESSED_DIR
from data.fetch_jolpica import load_qualifying, fetch_qualifying_for_round
from feature_engineering.car_performance import compute_pace_delta

MASTER_FEATURES_PATH = os.path.join(PROCESSED_DIR, "master_features.parquet")

DEFAULT_CAP_SECONDS = 4.0


# ---------------------------------------------------------------------------
# Section 1: Scoring
# ---------------------------------------------------------------------------

def score_from_delta(delta: float, cap_seconds: float) -> float:
    """0.0s delta -> 10.0. cap_seconds delta -> 0.0. Clamped at both ends."""
    raw = 10.0 * (1 - delta / cap_seconds)
    return max(0.0, min(10.0, raw))


def print_ranking(ranking_df: pd.DataFrame, cap_seconds: float, label: str) -> None:
    df = ranking_df.copy()
    df["car_score_0_10"] = df["pace_delta_vs_pole"].apply(
        lambda d: score_from_delta(d, cap_seconds)
    )
    df = df.sort_values("car_score_0_10", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    print(f"\n{'='*68}")
    print(f"CAR PERFORMANCE RANKING -- {label}")
    print(f"(0 = slowest car in field, 10 = fastest/pole-pace -- cap: {cap_seconds:.2f}s off pole)")
    print(f"{'='*68}")
    print(f"{'Rank':<6}{'Constructor':<18}{'Pace gap':>12}{'Score (0-10)':>16}")
    print("-" * 68)
    for _, row in df.iterrows():
        bar = "#" * int(round(row["car_score_0_10"]))
        print(f"{row['rank']:<6}{row['constructorId']:<18}"
              f"+{row['pace_delta_vs_pole']:>9.3f}s{row['car_score_0_10']:>12.2f}  {bar}")
    print(f"{'='*68}\n")


# ---------------------------------------------------------------------------
# Section 2: Historical mode (already-raced data)
# ---------------------------------------------------------------------------

def run_historical(season: int, round_num: int | None, cap_seconds: float) -> None:
    if not os.path.exists(MASTER_FEATURES_PATH):
        print(f"Error: {MASTER_FEATURES_PATH} not found. "
              f"Run `python -m data.build_master_features` first.")
        sys.exit(1)

    df = pd.read_parquet(MASTER_FEATURES_PATH)
    season_df = df[df["season"] == season]
    if season_df.empty:
        available = sorted(df["season"].unique().tolist())
        print(f"No data for season {season}. Available seasons: {available}")
        sys.exit(1)

    if round_num is None:
        round_num = int(season_df["round"].max())
        print(f"No --round given -- using the latest available round: {round_num}")

    round_df = season_df[season_df["round"] == round_num]
    if round_df.empty:
        available = sorted(season_df["round"].unique().tolist())
        print(f"No data for season={season}, round={round_num}. Available rounds: {available}")
        sys.exit(1)

    # pace_delta_vs_pole is a constructor-level feature -- both drivers
    # on a team carry the same value, so drop duplicates down to one
    # row per constructor before ranking.
    ranking = (
        round_df[["constructorId", "pace_delta_vs_pole"]]
        .drop_duplicates(subset="constructorId")
        .reset_index(drop=True)
    )
    print_ranking(ranking, cap_seconds, f"season {season}, round {round_num} (post-hoc)")


# ---------------------------------------------------------------------------
# Section 3: Upcoming mode (live qualifying, not yet in master_features)
# ---------------------------------------------------------------------------

def run_upcoming(season: int, round_num: int, cap_seconds: float) -> None:
    print(f"Fetching live qualifying for season={season}, round={round_num}...")
    live_quali = fetch_qualifying_for_round(season, round_num)
    if live_quali.empty:
        print(f"No qualifying data yet for season={season}, round={round_num} -- "
              f"this round's grid hasn't been set. Use --historical for a past "
              f"round, or wait until qualifying happens.")
        sys.exit(1)

    print("Loading cached historical qualifying...")
    hist_quali = load_qualifying()

    combined = pd.concat([hist_quali, live_quali], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["season", "round", "driverId"], keep="last"
    )

    print("Computing rolling pace delta (reuses car_performance.py directly)...")
    pace_df = compute_pace_delta(combined)

    round_pace = pace_df[
        (pace_df["season"] == season) & (pace_df["round"] == round_num)
    ][["constructorId", "pace_delta_vs_pole"]].dropna().reset_index(drop=True)

    if round_pace.empty:
        print("No pace delta could be computed for this round -- check that "
              "quali data fetched correctly.")
        sys.exit(1)

    print_ranking(round_pace, cap_seconds,
                  f"season {season}, round {round_num} (live -- going into this weekend)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank constructor car pace on a 0-10 scale")
    parser.add_argument("--historical", action="store_true",
                         help="Use already-raced data from master_features.parquet")
    parser.add_argument("--upcoming", action="store_true",
                         help="Live-fetch qualifying for a round not yet in master_features.parquet")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, default=None, dest="round_num",
                         help="Omit with --historical to use the latest available round")
    parser.add_argument("--cap", type=float, default=DEFAULT_CAP_SECONDS,
                         help=f"Pace gap in seconds that scores as 0.0 (default: {DEFAULT_CAP_SECONDS})")
    args = parser.parse_args()

    if args.historical == args.upcoming:  # both False or both True
        print("Specify exactly one of --historical or --upcoming.")
        sys.exit(1)

    if args.historical:
        run_historical(args.season, args.round_num, args.cap)
    else:
        if args.round_num is None:
            print("--upcoming requires --round.")
            sys.exit(1)
        run_upcoming(args.season, args.round_num, args.cap)
