"""
experiments/teammate_delta_rolling/backtest_v2.py

Re-runs the exact same Brier-score backtest as the real project's
backtest.py, but against the v2 model + v2 features (rolling
teammate_delta). This is THE number that actually matters for deciding
whether the change is worth porting into the real project -- not the
raw ranking MAE from train_v2.py, which is known to be seed-noisy.

Reuses evaluate_race(), run_backtest(), and print_report() DIRECTLY
from the real project's backtest.py -- unmodified. Those functions take
the ensemble and test_df as parameters; they don't hardcode any path,
so pointing them at v2 data/model requires no changes to that file at
all.

WRITES ONLY WITHIN THIS FOLDER:
    experiments/teammate_delta_rolling/backtest_results_v2.parquet

Usage:
    python backtest_v2.py
    python backtest_v2.py --runs 2000
"""

import os
import sys
import argparse
import joblib
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from backtest import evaluate_race, run_backtest, print_report, N_BACKTEST_RUNS_DEFAULT
from models.race_model import TEST_SEASONS

MASTER_FEATURES_V2_PATH = os.path.join(_THIS_DIR, "master_features_v2.parquet")
MODEL_V2_PATH = os.path.join(_THIS_DIR, "race_model_v2.joblib")
BACKTEST_RESULTS_V2_PATH = os.path.join(_THIS_DIR, "backtest_results_v2.parquet")


def _load_model_v2() -> list:
    if not os.path.exists(MODEL_V2_PATH):
        raise FileNotFoundError(
            f"No trained v2 model at {MODEL_V2_PATH} -- run `python train_v2.py` first"
        )
    ensemble = joblib.load(MODEL_V2_PATH)
    if not isinstance(ensemble, list):
        ensemble = [ensemble]
    return ensemble


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brier score backtest for the v2 (rolling teammate_delta) model")
    parser.add_argument("--runs", type=int, default=N_BACKTEST_RUNS_DEFAULT)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if not os.path.exists(MASTER_FEATURES_V2_PATH):
        print(f"Error: {MASTER_FEATURES_V2_PATH} not found. "
              f"Run `python build_master_features_v2.py` first.")
        sys.exit(1)

    print("Loading v2 trained model ensemble...")
    ensemble = _load_model_v2()

    print("Loading v2 master features...")
    df = pd.read_parquet(MASTER_FEATURES_V2_PATH)

    seasons = args.seasons if args.seasons else list(TEST_SEASONS)
    test_df = df[df["season"].isin(seasons)].copy()
    print(f"  Test set: {len(test_df)} rows, seasons {sorted(test_df['season'].unique())}")

    print("\n[v2] Running backtest (rolling teammate_delta model)...")
    results = run_backtest(test_df, ensemble, n_runs=args.runs)

    print(f"\nSaving detailed results to {BACKTEST_RESULTS_V2_PATH}...")
    results.to_parquet(BACKTEST_RESULTS_V2_PATH, index=False)
    print("  ✓ Saved")

    print_report(results)

    print(f"{'='*70}")
    print("REMINDER: compare the 'Full P1-Pn distribution' skill score above")
    print("against the real project's baseline (+0.465, per Final_Project_History.md")
    print("section 7.5). Higher = the rolling teammate_delta change helped.")
    print("Lower or negative = it hurt, and should NOT be ported into the real")
    print("project despite being more internally consistent with elo/recent_form.")
    print(f"{'='*70}\n")
