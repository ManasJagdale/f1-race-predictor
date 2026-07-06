"""
simulation/simulation.py

Monte Carlo race simulation. Takes the trained pairwise ranking model's
strength scores for a race and runs 10,000 simulated finishing orders,
sampling DNFs, safety cars, and pit-stop-driven volatility per run.
Aggregates into a full P1-P20 probability distribution per driver.

KNOWN MVP LIMITATIONS (flagged deliberately, not hidden):
  - Safety car rates below are an ILLUSTRATIVE starter table, not fetched
    historical data. Jolpica doesn't track safety car periods directly,
    so this needs either a manually curated table (researched per circuit,
    same pattern as circuit_profiles.xlsx) or a proxy derived from race
    data before it should be relied on for real probabilities.
  - Pit stop sampling is a lightweight volatility perturbation, not a
    real historical distribution. Jolpica's /pitstops.json endpoint has
    the real data; a dedicated fetcher (similar to fetch_jolpica.py)
    would replace this placeholder.
  - DNF sampling is real (uses the existing dnf_rate feature) — no
    placeholder there.

Usage (from project root):
    python -m simulation.simulation --season 2026 --round 8 --runs 10000
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DIR
from models.race_model import MODEL_PATH, MASTER_FEATURES_PATH, FEATURE_COLS

N_RUNS_DEFAULT = 10_000

# Per-run Gaussian noise added to each driver's strength score, as a
# fraction of the score spread within that race. This is what gives
# Monte Carlo runs variance instead of producing the same order every
# time. Not yet tuned against backtest results — a reasonable starting
# point, revisit once backtest.py exists and can score calibration.
SCORE_NOISE_STD_FRAC = 0.20

# --- Safety car rates: ILLUSTRATIVE STARTER TABLE, see module docstring ---
# Approximate probability of at least one safety car/VSC period per race,
# based on general circuit reputation (street circuits and high-attrition
# tracks trend higher). Replace with real historical rates before relying
# on these for anything beyond architecture testing.
SAFETY_CAR_RATES = {
    "monaco":        0.75,
    "baku":          0.80,
    "singapore":     0.75,
    "jeddah":        0.70,
    "las_vegas":     0.55,
    "miami":         0.55,
    "montreal":      0.60,
    "melbourne":     0.50,
    "spa":           0.45,
    "silverstone":   0.35,
    "monza":         0.30,
    "suzuka":        0.30,
    "interlagos":    0.45,
    "red_bull_ring": 0.30,
    "hungaroring":   0.35,
}
DEFAULT_SAFETY_CAR_RATE = 0.40  # fallback for any circuit not in the table above


def _load_model() -> list:
    """Load the trained ensemble (list of Pipeline objects) saved by race_model.py."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH} — run `python -m models.race_model` first"
        )
    ensemble = joblib.load(MODEL_PATH)
    if not isinstance(ensemble, list):
        # Backward compatibility: an older single-model race_model.joblib
        # (pre-ensemble) would load as a single Pipeline, not a list.
        print("  ⚠ Loaded model is a single Pipeline, not an ensemble list — "
              "wrapping it as a 1-member ensemble. Re-run `python -m models.race_model` "
              "to get the full 8-seed ensemble.")
        ensemble = [ensemble]
    return ensemble


def get_race_features(season: int, round_num: int) -> pd.DataFrame:
    """Pull the pre-race feature rows for a single race from master_features."""
    df = pd.read_parquet(MASTER_FEATURES_PATH)
    race = df[(df["season"] == season) & (df["round"] == round_num)].reset_index(drop=True)
    if race.empty:
        raise ValueError(f"No data found for season={season}, round={round_num}")
    return race


# ---------------------------------------------------------------------------
# Section 1: Strength scores from the trained model
# ---------------------------------------------------------------------------

def compute_strength_scores(ensemble: list, race_df: pd.DataFrame) -> np.ndarray:
    """
    Round-robin pairwise scoring — same approach as race_model.py's
    predict_race_rankings, but returns raw scores (not ranks) so we can
    add per-run noise and re-rank inside the Monte Carlo loop.

    Each driver's score = sum of P(beats driver_j) across all other
    drivers in the race, where each pairwise probability is itself
    averaged across every model in the ensemble (see race_model.py —
    a single seed's estimate proved unstable; averaging N seeds is the
    fix). Higher score = stronger.
    """
    n = len(race_df)
    feats = race_df[FEATURE_COLS].values
    scores = np.zeros(n)

    for i, j in combinations(range(n), 2):
        diff = feats[i] - feats[j]
        probs = [pipeline.predict_proba([diff])[0][1] for pipeline in ensemble]
        prob = float(np.mean(probs))
        scores[i] += prob
        scores[j] += (1 - prob)

    return scores


# ---------------------------------------------------------------------------
# Section 2: Single simulation run
# ---------------------------------------------------------------------------

def _sample_dnf_mask(race_df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """One Bernoulli draw per driver using their constructor's dnf_rate."""
    dnf_probs = race_df["dnf_rate"].fillna(0.1).values
    draws = rng.random(len(race_df))
    return draws < dnf_probs


def _safety_car_triggered(circuit_id: str, rng: np.random.Generator) -> bool:
    rate = SAFETY_CAR_RATES.get(circuit_id, DEFAULT_SAFETY_CAR_RATE)
    return rng.random() < rate


def _apply_safety_car_compression(order: list, rng: np.random.Generator) -> list:
    """
    Compress a random window of the finishing order and shuffle within it.
    Crude approximation of how a safety car bunches the field and creates
    positional randomness — not a physical strategy model.
    """
    n = len(order)
    if n < 4:
        return order

    window_size = min(rng.integers(4, 9), n)
    start = rng.integers(0, n - window_size + 1)
    window = order[start:start + window_size]
    rng.shuffle(window)
    return order[:start] + window + order[start + window_size:]


def _apply_pitstop_volatility(order: list, race_df: pd.DataFrame, rng: np.random.Generator) -> list:
    """
    Placeholder pit-stop effect: sample a stop count per driver from a
    generic 1-3 stop distribution. Each individual stop is treated as an
    independent opportunity for a small track-position swing (undercut/
    overcut gain or loss, pit-lane traffic, safety car timing luck) —
    so drivers with more stops get more chances at a swing, rather than
    only 3-stop strategies getting any variance at all.

    NOTE: PIT_SWAP_PROB_PER_STOP below is a placeholder constant, same
    status as SCORE_NOISE_STD_FRAC — not yet tuned against real data.
    Real per-circuit historical strategy data (Jolpica /pitstops.json)
    would replace both the stop-count distribution and this probability.

    Bug history: previously this only checked `stop_counts[idx] >= 3`,
    meaning 1- and 2-stop strategies (80% of the sampled distribution,
    including the most common real-world strategy) got zero pit-related
    position variance. Fixed to scale with actual stop count instead.
    """
    PIT_SWAP_PROB_PER_STOP = 0.12  # chance any single stop causes a swing

    n = len(order)
    stop_counts = rng.choice([1, 2, 3], size=n, p=[0.25, 0.55, 0.20])

    for idx in range(n):
        for _ in range(stop_counts[idx]):
            if rng.random() < PIT_SWAP_PROB_PER_STOP:
                swap_idx = idx + rng.integers(-1, 2)
                swap_idx = max(0, min(n - 1, swap_idx))
                order[idx], order[swap_idx] = order[swap_idx], order[idx]

    return order


def run_single_simulation(
    race_df: pd.DataFrame,
    base_scores: np.ndarray,
    circuit_id: str,
    rng: np.random.Generator,
) -> list:
    """
    Run one Monte Carlo iteration. Returns a tuple of:
      - list of driverIds in finishing order (index 0 = P1), with DNF'd
        drivers placed at the back in random order
      - set of driverIds that DNF'd this run (for accurate P(DNF) tracking)
    """
    n = len(race_df)
    driver_ids = race_df["driverId"].values

    # 1. Noisy strength scores → base finishing order
    noise = rng.normal(0, base_scores.std() * SCORE_NOISE_STD_FRAC + 1e-6, size=n)
    noisy_scores = base_scores + noise
    ranked_idx = np.argsort(-noisy_scores)  # descending: best first
    order = list(driver_ids[ranked_idx])

    # 2. DNFs — remove finishers, push DNF'd drivers to the back (random order)
    dnf_mask = _sample_dnf_mask(race_df, rng)
    dnf_driver_ids = set(race_df.loc[dnf_mask, "driverId"].values)
    if dnf_driver_ids:
        finishers = [d for d in order if d not in dnf_driver_ids]
        dnfs = [d for d in order if d in dnf_driver_ids]
        rng.shuffle(dnfs)
        order = finishers + dnfs

    # 3. Safety car — sampled once per run, applied to the finisher portion only
    n_dnf = len(dnf_driver_ids)
    finisher_order = order[: n - n_dnf] if n_dnf else order
    dnf_tail = order[n - n_dnf:] if n_dnf else []

    if _safety_car_triggered(circuit_id, rng):
        finisher_order = _apply_safety_car_compression(finisher_order, rng)

    # 4. Pit stop volatility — placeholder perturbation
    finisher_order = _apply_pitstop_volatility(finisher_order, race_df, rng)

    return finisher_order + dnf_tail, dnf_driver_ids


# ---------------------------------------------------------------------------
# Section 3: Full Monte Carlo + aggregation
# ---------------------------------------------------------------------------

def run_monte_carlo(
    race_df: pd.DataFrame,
    ensemble,
    n_runs: int = N_RUNS_DEFAULT,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run n_runs simulations and aggregate into a P1-Pn probability matrix.

    Args:
        verbose: if False, suppresses progress prints — used by backtest.py,
                 which calls this once per race across many races and would
                 otherwise flood the terminal. Default True preserves the
                 normal single-race CLI experience.

    Returns:
        DataFrame indexed by driverId, columns P1..Pn, values = probability
        of finishing in that exact position across all runs.
    """
    rng = np.random.default_rng(seed)
    n_drivers = len(race_df)
    driver_ids = race_df["driverId"].values
    circuit_id = race_df["circuitId"].iloc[0]

    if verbose:
        print(f"  Computing strength scores for {n_drivers} drivers...")
    base_scores = compute_strength_scores(ensemble, race_df)

    position_counts = {d: np.zeros(n_drivers, dtype=int) for d in driver_ids}
    dnf_counts = {d: 0 for d in driver_ids}

    if verbose:
        print(f"  Running {n_runs:,} Monte Carlo simulations...")
    for run_i in range(n_runs):
        order, dnf_driver_ids = run_single_simulation(race_df, base_scores, circuit_id, rng)
        for pos, driver_id in enumerate(order):
            position_counts[driver_id][pos] += 1
        for driver_id in dnf_driver_ids:
            dnf_counts[driver_id] += 1

        if verbose and (run_i + 1) % 2000 == 0:
            print(f"    {run_i + 1:,} / {n_runs:,} runs complete")

    prob_matrix = pd.DataFrame(
        {d: position_counts[d] / n_runs for d in driver_ids}
    ).T
    prob_matrix.columns = [f"P{i+1}" for i in range(n_drivers)]
    prob_matrix.index.name = "driverId"

    dnf_prob = pd.Series(
        {d: dnf_counts[d] / n_runs for d in driver_ids}, name="P(DNF)"
    )
    dnf_prob.index.name = "driverId"

    return prob_matrix, dnf_prob


# ---------------------------------------------------------------------------
# Section 4: Display helpers
# ---------------------------------------------------------------------------

def summarize(prob_matrix: pd.DataFrame, dnf_prob: pd.Series, race_df: pd.DataFrame) -> pd.DataFrame:
    """Build the terminal-table summary: P(Win), P(P1-3), P(P1-6), P(DNF)."""
    summary = pd.DataFrame(index=prob_matrix.index)
    summary["P(Win)"] = prob_matrix["P1"]
    summary["P(P1-3)"] = prob_matrix[[f"P{i}" for i in range(1, 4)]].sum(axis=1)
    summary["P(P1-6)"] = prob_matrix[[f"P{i}" for i in range(1, 7)]].sum(axis=1)
    summary["P(DNF)"] = dnf_prob
    summary = summary.merge(
        race_df[["driverId", "constructorId"]].set_index("driverId"),
        left_index=True, right_index=True,
    )
    return summary.sort_values("P(Win)", ascending=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo race simulation")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_num")
    parser.add_argument("--runs", type=int, default=N_RUNS_DEFAULT)
    args = parser.parse_args()

    print("\nLoading trained model ensemble...")
    ensemble = _load_model()

    print(f"Loading race features (season={args.season}, round={args.round_num})...")
    race_df = get_race_features(args.season, args.round_num)
    print(f"  {len(race_df)} drivers found")

    prob_matrix, dnf_prob = run_monte_carlo(race_df, ensemble, n_runs=args.runs)

    summary = summarize(prob_matrix, dnf_prob, race_df)
    print(f"\n{'='*70}")
    print(f"Predicted Finishing Probabilities — season {args.season}, round {args.round_num}")
    print(f"{'='*70}")
    print(summary[["constructorId", "P(Win)", "P(P1-3)", "P(P1-6)", "P(DNF)"]]
          .to_string(float_format=lambda x: f"{x:.1%}"))

    out_path = os.path.join(PROCESSED_DIR, f"sim_{args.season}_r{args.round_num}.parquet")
    prob_matrix.to_parquet(out_path)
    print(f"\n✓ Full P1-Pn matrix saved to {out_path}")
