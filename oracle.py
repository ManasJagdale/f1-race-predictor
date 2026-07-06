"""
oracle.py

CLI entry point for the F1 predictor. Ties together the trained ensemble,
the Monte Carlo simulation, and three output formats: terminal table,
CSV (for the Dutching engine), and a Plotly heatmap (portfolio-facing).

This is intentionally a thin wrapper — all the real logic already lives
in simulation.py (get_race_features, run_monte_carlo, summarize) and
models/race_model.py (the trained ensemble). Reusing those functions
directly means oracle.py can't silently diverge from what backtest.py
already validated.

RACE NAME RESOLUTION:
The underlying data only has circuitId (e.g. "silverstone"), not human
race names. This maintains a static circuitId -> race name lookup for
fuzzy matching "British Grand Prix 2024" style queries, with a fallback
to exact --season/--round flags when a name doesn't resolve cleanly
(new circuits, ambiguous names, or years not yet mapped).

Usage:
    python oracle.py --race "British Grand Prix 2024"
    python oracle.py --season 2026 --round 8
    python oracle.py --season 2026 --round 8 --runs 20000 --show
    python oracle.py --list --season 2026
"""

import os
import sys
import re
import argparse
import difflib
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.race_model import MASTER_FEATURES_PATH
from simulation.simulation import (
    get_race_features,
    run_monte_carlo,
    summarize,
    _load_model,
    N_RUNS_DEFAULT,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


# ---------------------------------------------------------------------------
# Section 1: Race name resolution
# ---------------------------------------------------------------------------
# circuitId -> human-readable race name. Covers the current F1 calendar
# plus a handful of recent legacy circuits that may appear in 2018+ data.
# Not exhaustive — if a circuit isn't here, --race falls back to fuzzy
# matching against raw circuitIds, and --season/--round always works
# regardless of this table's completeness.

CIRCUIT_TO_RACE_NAME = {
    "albert_park":     "Australian Grand Prix",
    "bahrain":         "Bahrain Grand Prix",
    "jeddah":          "Saudi Arabian Grand Prix",
    "miami":           "Miami Grand Prix",
    "imola":           "Emilia Romagna Grand Prix",
    "monaco":          "Monaco Grand Prix",
    "catalunya":       "Spanish Grand Prix",
    "madring":         "Spanish Grand Prix",   # new Madrid circuit (Jolpica circuitId is 'madring')
    "villeneuve":      "Canadian Grand Prix",
    "red_bull_ring":   "Austrian Grand Prix",
    "silverstone":     "British Grand Prix",
    "hungaroring":     "Hungarian Grand Prix",
    "spa":             "Belgian Grand Prix",
    "zandvoort":       "Dutch Grand Prix",
    "monza":           "Italian Grand Prix",
    "baku":            "Azerbaijan Grand Prix",
    "marina_bay":      "Singapore Grand Prix",
    "suzuka":          "Japanese Grand Prix",
    "losail":          "Qatar Grand Prix",
    "americas":        "United States Grand Prix",
    "rodriguez":       "Mexico City Grand Prix",
    "interlagos":      "Sao Paulo Grand Prix",
    "vegas":           "Las Vegas Grand Prix",
    "yas_marina":      "Abu Dhabi Grand Prix",
    "shanghai":        "Chinese Grand Prix",
    # Legacy circuits that may appear in 2018-2021 data
    "hockenheimring":  "German Grand Prix",
    "nurburgring":     "Eifel Grand Prix",
    "istanbul":        "Turkish Grand Prix",
    "sochi":           "Russian Grand Prix",
    "portimao":        "Portuguese Grand Prix",
    "mugello":         "Tuscan Grand Prix",
    "ricard":          "French Grand Prix",
}

RACE_NAME_TO_CIRCUIT = {v.lower(): k for k, v in CIRCUIT_TO_RACE_NAME.items()}


def _parse_race_query(race_query: str) -> tuple[str, int | None]:
    """
    Split '<name> <year>' into (name, year). Year is optional — if
    absent, resolution will require --season to disambiguate.
    """
    match = re.match(r"^(.+?)\s+(\d{4})\s*$", race_query.strip())
    if match:
        return match.group(1).strip(), int(match.group(2))
    return race_query.strip(), None


def _strip_boilerplate(s: str) -> str:
    """Remove 'Grand Prix' / 'GP' so fuzzy matching compares the part of
    the name that's actually distinctive, not the part every race shares."""
    s = re.sub(r"\bgrand prix\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bgp\b", "", s, flags=re.IGNORECASE)
    return s.strip()


def resolve_race(race_query: str, master_df: pd.DataFrame) -> tuple[int, int, str]:
    """
    Resolve a free-text race query like 'British Grand Prix 2024' into
    (season, round, circuitId). Raises ValueError with a helpful message
    (including close-match suggestions) if it can't be resolved.

    Matching priority (safest/most specific first — see note below on
    why order matters here):
      1. Exact circuitId match (case-insensitive) — e.g. 'monza 2026'
      2. Exact race-name match — e.g. 'Monaco Grand Prix 2026'
      3. Fuzzy race-name match, with 'Grand Prix'/'GP' stripped first —
         e.g. 'british gp 2026' or a typo like 'britsh grand prix 2026'
      4. Fuzzy circuitId match — for typos in the raw circuitId itself

    NOTE ON ORDER: fuzzy-matching 'Grand Prix'-stripped race names looks
    obviously right for abbreviations, but it has a sharp edge — 'Monza'
    (a circuitId) fuzzy-matches 'Monaco' (a stripped race name) closely
    enough to pass a reasonable cutoff, since both are short single
    words with several overlapping letters. Checking exact circuitId
    first means a real circuit name never falls through to a fuzzy
    match against something else. This was caught by testing before
    considering the resolver reliable — worth knowing if you extend
    the matching logic further.
    """
    name_part, year = _parse_race_query(race_query)

    if year is None:
        raise ValueError(
            f"Couldn't find a year in '{race_query}'. "
            f"Try '{race_query} 2026' or use --season/--round directly."
        )

    season_df = master_df[master_df["season"] == year]
    if season_df.empty:
        available = sorted(master_df["season"].unique())
        raise ValueError(f"No data for season {year}. Available seasons: {available}")

    circuits_this_season = season_df["circuitId"].unique().tolist()
    display_names = {c: CIRCUIT_TO_RACE_NAME.get(c, c) for c in circuits_this_season}

    name_lower = name_part.lower().strip()

    # --- Priority 1: exact circuitId match ---
    for circuit_id in circuits_this_season:
        if circuit_id.lower() == name_lower or circuit_id.lower().replace("_", " ") == name_lower:
            round_num = int(season_df[season_df["circuitId"] == circuit_id]["round"].iloc[0])
            return year, round_num, circuit_id

    # --- Priority 2: exact race-name match ---
    for circuit_id, display_name in display_names.items():
        if display_name.lower() == name_lower:
            round_num = int(season_df[season_df["circuitId"] == circuit_id]["round"].iloc[0])
            return year, round_num, circuit_id

    # --- Priority 3: fuzzy race-name match, boilerplate stripped ---
    stripped_query = _strip_boilerplate(name_part).lower()
    stripped_to_circuit = {
        _strip_boilerplate(name).lower(): circuit_id
        for circuit_id, name in display_names.items()
    }
    if stripped_query:  # guard against a query that's ONLY "Grand Prix"
        close = difflib.get_close_matches(
            stripped_query, list(stripped_to_circuit.keys()), n=1, cutoff=0.6
        )
        if close:
            circuit_id = stripped_to_circuit[close[0]]
            round_num = int(season_df[season_df["circuitId"] == circuit_id]["round"].iloc[0])
            return year, round_num, circuit_id

    # --- Priority 4: fuzzy circuitId match (typo in raw circuitId) ---
    close = difflib.get_close_matches(name_lower, [c.lower() for c in circuits_this_season],
                                       n=1, cutoff=0.7)
    if close:
        circuit_id = next(c for c in circuits_this_season if c.lower() == close[0])
        round_num = int(season_df[season_df["circuitId"] == circuit_id]["round"].iloc[0])
        return year, round_num, circuit_id

    # --- Nothing matched — fail loudly with suggestions rather than guessing ---
    suggestions = difflib.get_close_matches(
        stripped_query, list(stripped_to_circuit.keys()), n=3, cutoff=0.3
    )
    suggestion_names = [display_names[stripped_to_circuit[s]] for s in suggestions]
    suggestion_str = ", ".join(suggestion_names) if suggestion_names else "no close matches found"
    raise ValueError(
        f"Couldn't resolve '{name_part}' to a circuit in season {year}. "
        f"Closest matches: {suggestion_str}. "
        f"Use --list --season {year} to see all races that season."
    )


def list_races(master_df: pd.DataFrame, season: int) -> None:
    """Print every race available for a season — helps when --race fails to resolve."""
    season_df = master_df[master_df["season"] == season]
    if season_df.empty:
        print(f"No data for season {season}.")
        return

    races = (
        season_df[["round", "circuitId"]]
        .drop_duplicates()
        .sort_values("round")
    )
    print(f"\nRaces available for season {season}:")
    for _, row in races.iterrows():
        name = CIRCUIT_TO_RACE_NAME.get(row["circuitId"], row["circuitId"])
        print(f"  Round {int(row['round']):>2}  {name:<30} (circuitId: {row['circuitId']})")
    print()


# ---------------------------------------------------------------------------
# Section 2: Output — CSV export
# ---------------------------------------------------------------------------

def export_csv(prob_matrix: pd.DataFrame, season: int, round_num: int, circuit_id: str) -> str:
    """
    Save the full P1-Pn matrix as CSV, one row per driver, one column
    per finishing position — formatted for direct input to a Dutching
    engine or any downstream consumer expecting a flat probability table.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{season}_r{round_num:02d}_{circuit_id}_probabilities.csv"
    path = os.path.join(OUTPUT_DIR, filename)

    out = prob_matrix.reset_index()  # driverId becomes a normal column
    out.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Section 3: Output — Plotly heatmap
# ---------------------------------------------------------------------------

def build_heatmap(prob_matrix: pd.DataFrame, season: int, round_num: int, circuit_id: str) -> go.Figure:
    """
    Drivers (rows) x finishing positions (cols), cell intensity =
    probability. Drivers sorted by P(Win) descending so the strongest
    contenders sit at the top of the chart.
    """
    sorted_matrix = prob_matrix.loc[
        prob_matrix["P1"].sort_values(ascending=False).index
    ]

    race_name = CIRCUIT_TO_RACE_NAME.get(circuit_id, circuit_id)

    fig = go.Figure(data=go.Heatmap(
        z=sorted_matrix.values,
        x=list(sorted_matrix.columns),
        y=list(sorted_matrix.index),
        colorscale="YlOrRd",
        colorbar=dict(title="Probability"),
        hovertemplate="Driver: %{y}<br>Position: %{x}<br>Probability: %{z:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{race_name} {season} — Predicted Finishing Position Probabilities",
        xaxis_title="Finishing Position",
        yaxis_title="Driver",
        yaxis=dict(autorange="reversed"),  # strongest driver at top
        height=max(400, 30 * len(sorted_matrix)),
    )
    return fig


def export_heatmap(fig: go.Figure, season: int, round_num: int, circuit_id: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{season}_r{round_num:02d}_{circuit_id}_heatmap.html"
    path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(path)
    return path


# ---------------------------------------------------------------------------
# Section 4: Terminal output
# ---------------------------------------------------------------------------

def print_terminal_table(summary: pd.DataFrame, season: int, round_num: int, circuit_id: str) -> None:
    race_name = CIRCUIT_TO_RACE_NAME.get(circuit_id, circuit_id)
    print(f"\n{race_name} {season} (Round {round_num}) — Predicted Finishing Probabilities\n")
    print(f"{'Driver':<20} {'Team':<15} {'P(Win)':>8} {'P(P1-3)':>9} {'P(P1-6)':>9} {'P(DNF)':>8}")
    print("-" * 73)
    for driver_id, row in summary.iterrows():
        print(f"{driver_id:<20} {row['constructorId']:<15} "
              f"{row['P(Win)']:>7.1%} {row['P(P1-3)']:>9.1%} "
              f"{row['P(P1-6)']:>9.1%} {row['P(DNF)']:>8.1%}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Race Outcome Predictor — Oracle CLI")
    parser.add_argument("--race", type=str, default=None,
                         help='Race query, e.g. "British Grand Prix 2024"')
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--round", type=int, default=None, dest="round_num")
    parser.add_argument("--runs", type=int, default=N_RUNS_DEFAULT,
                         help=f"Monte Carlo runs (default: {N_RUNS_DEFAULT:,})")
    parser.add_argument("--show", action="store_true",
                         help="Open the heatmap in a browser after saving")
    parser.add_argument("--list", action="store_true",
                         help="List available races for --season and exit")
    args = parser.parse_args()

    print("Loading master features...")
    master_df = pd.read_parquet(MASTER_FEATURES_PATH)

    if args.list:
        if args.season is None:
            print("--list requires --season, e.g. --list --season 2026")
            sys.exit(1)
        list_races(master_df, args.season)
        sys.exit(0)

    # --- Resolve season/round ---
    if args.season is not None and args.round_num is not None:
        season, round_num = args.season, args.round_num
        match = master_df[(master_df["season"] == season) & (master_df["round"] == round_num)]
        circuit_id = match["circuitId"].iloc[0] if not match.empty else "unknown"
    elif args.race is not None:
        try:
            season, round_num, circuit_id = resolve_race(args.race, master_df)
        except ValueError as e:
            print(f"\nError: {e}\n")
            sys.exit(1)
    else:
        print("Provide either --race \"<name> <year>\" or both --season and --round.")
        print('Example: python oracle.py --race "British Grand Prix 2024"')
        print("Example: python oracle.py --season 2026 --round 8")
        sys.exit(1)

    print("Loading trained model ensemble...")
    ensemble = _load_model()

    print(f"Loading race features (season={season}, round={round_num}, circuit={circuit_id})...")
    race_df = get_race_features(season, round_num)
    print(f"  {len(race_df)} drivers found")

    prob_matrix, dnf_prob = run_monte_carlo(race_df, ensemble, n_runs=args.runs)

    summary = summarize(prob_matrix, dnf_prob, race_df)

    # --- Output 1: Terminal table ---
    print_terminal_table(summary, season, round_num, circuit_id)

    # --- Output 2: CSV export ---
    csv_path = export_csv(prob_matrix, season, round_num, circuit_id)
    print(f"✓ CSV saved → {csv_path}")

    # --- Output 3: Plotly heatmap ---
    fig = build_heatmap(prob_matrix, season, round_num, circuit_id)
    heatmap_path = export_heatmap(fig, season, round_num, circuit_id)
    print(f"✓ Heatmap saved → {heatmap_path}")

    if args.show:
        fig.show()
