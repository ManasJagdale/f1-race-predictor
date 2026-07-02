"""
feature_engineering/track_car_interaction.py

Builds the straight-line vs cornering interaction features that let the
model learn whether a constructor's pace advantage is circuit-dependent
(e.g. "is this car better suited to power tracks or technical tracks?").

Why this exists:
  car_performance.py gives us ONE number per constructor per race —
  pace_delta_vs_pole — which is the average qualifying gap to pole.
  This treats every circuit the same, even though we know cars have
  different strengths (some excel on straights, some in corners).

  We don't have real per-sector qualifying times from Jolpica (it only
  gives full lap times), so we can't directly measure "how much of this
  car's pace gap comes from sector 1 vs sector 2". Instead we build an
  INTERACTION feature: the existing pace_delta_vs_pole multiplied by
  how straight-line-dominant or cornering-dominant the UPCOMING circuit
  is (from circuit_profiles.xlsx, the one-time FastF1 telemetry analysis).

  This lets the model learn the relationship itself — e.g. "a car with
  a small pace_delta tends to do even better at high-throttle tracks" —
  without us hardcoding any assumption about WHY a car is fast.

Output columns added to the feature table:
  straight_line_exposure  = pace_delta_vs_pole * circuit's avg %full-throttle
  cornering_exposure      = pace_delta_vs_pole * circuit's avg %braking

Usage:
    from feature_engineering.track_car_interaction import add_track_car_interaction
    master = add_track_car_interaction(master_df, circuit_profiles_path)
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Section 1: Load and summarise circuit profiles
# ---------------------------------------------------------------------------
# circuit_profiles.xlsx has per-sector pct_full_throttle / pct_braking.
# We collapse the 3 sectors into one circuit-level summary: the average
# %full-throttle and %braking across all 3 sectors. This is a simplification
# (it loses which SPECIFIC sector is straight vs corner) but matches the
# granularity of pace_delta_vs_pole, which is also a single whole-lap number.
#
# A more granular version (per-sector pace deltas) would need real FastF1
# sector times pulled per constructor per race — a bigger lift, logged as
# a possible Phase 2 extension if this coarser version proves useful first.

def load_circuit_summary(circuit_profiles_path: str) -> pd.DataFrame:
    """
    Load circuit_profiles.xlsx and collapse to one row per circuit with
    a single avg_pct_full_throttle and avg_pct_braking value.

    Returns:
        DataFrame with columns: circuitId, avg_pct_full_throttle, avg_pct_braking
    """
    profiles = pd.read_excel(circuit_profiles_path)

    ft_cols = [c for c in profiles.columns if c.endswith("_pct_full_throttle")]
    br_cols = [c for c in profiles.columns if c.endswith("_pct_braking")]

    profiles["avg_pct_full_throttle"] = profiles[ft_cols].mean(axis=1)
    profiles["avg_pct_braking"]       = profiles[br_cols].mean(axis=1)

    # If a circuit has multiple rows (shouldn't, but just in case a layout
    # changed and both old/new rows exist), keep the most recent season.
    summary = (
        profiles.sort_values("season")
        .groupby("circuitId")
        .last()
        .reset_index()[["circuitId", "avg_pct_full_throttle", "avg_pct_braking"]]
    )

    print(f"  ✓ Circuit summary: {len(summary)} circuits loaded from {circuit_profiles_path}")
    return summary


# ---------------------------------------------------------------------------
# Section 2: Build the interaction features
# ---------------------------------------------------------------------------

def add_track_car_interaction(
    master_df: pd.DataFrame,
    circuit_profiles_path: str,
) -> pd.DataFrame:
    """
    Add straight_line_exposure and cornering_exposure columns to the
    master features table.

    Args:
        master_df: the existing master features table — must already
                   contain 'circuitId' and 'pace_delta_vs_pole' columns
        circuit_profiles_path: path to circuit_profiles.xlsx

    Returns:
        master_df with two new columns added.
    """
    if "circuitId" not in master_df.columns:
        raise ValueError("master_df must contain a 'circuitId' column")
    if "pace_delta_vs_pole" not in master_df.columns:
        raise ValueError("master_df must contain a 'pace_delta_vs_pole' column "
                          "— run car_performance.py first")

    circuit_summary = load_circuit_summary(circuit_profiles_path)

    before_rows = len(master_df)
    master_df = master_df.merge(circuit_summary, on="circuitId", how="left")

    missing = master_df["avg_pct_full_throttle"].isna().sum()
    if missing > 0:
        print(f"  ⚠ {missing} rows have no circuit profile match — "
              f"these circuitIds are missing from circuit_profiles.xlsx")
        # Impute with the dataset-wide median rather than leaving NaN,
        # consistent with how other features are imputed in build_master_features.py
        median_ft = master_df["avg_pct_full_throttle"].median()
        median_br = master_df["avg_pct_braking"].median()
        master_df["avg_pct_full_throttle"] = master_df["avg_pct_full_throttle"].fillna(median_ft)
        master_df["avg_pct_braking"]       = master_df["avg_pct_braking"].fillna(median_br)

    assert len(master_df) == before_rows, "Merge changed row count — check for duplicate circuitIds"

    # The actual interaction features
    master_df["straight_line_exposure"] = (
        master_df["pace_delta_vs_pole"] * master_df["avg_pct_full_throttle"]
    )
    master_df["cornering_exposure"] = (
        master_df["pace_delta_vs_pole"] * master_df["avg_pct_braking"]
    )

    # Drop the intermediate circuit-level columns — they're now folded
    # into the two interaction features and would otherwise just be
    # near-duplicate information for the model (track_features.py already
    # supplies similar raw track descriptors).
    master_df = master_df.drop(columns=["avg_pct_full_throttle", "avg_pct_braking"])

    print(f"  ✓ Added straight_line_exposure and cornering_exposure "
          f"({len(master_df)} rows, 0 NaN)")
    return master_df


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("track_car_interaction.py — sanity check\n")

    # Build a tiny synthetic master_df to test the join + math in isolation,
    # without needing the full pipeline to be re-run.
    test_df = pd.DataFrame({
        "circuitId":          ["monza", "monaco", "baku", "unknown_track"],
        "constructorId":      ["mercedes", "ferrari", "red_bull", "haas"],
        "pace_delta_vs_pole": [0.0, 1.2, 0.5, 1.0],
    })

    circuit_profiles_path = sys.argv[1] if len(sys.argv) > 1 else "circuit_profiles.xlsx"

    result = add_track_car_interaction(test_df, circuit_profiles_path)
    print("\nResult:")
    print(result.to_string(index=False))

    # Checks
    print("\nChecks:")
    assert "straight_line_exposure" in result.columns
    assert "cornering_exposure" in result.columns
    assert result["straight_line_exposure"].isna().sum() == 0
    assert result["cornering_exposure"].isna().sum() == 0
    print("  ✓ Both interaction columns present, 0 NaN")

    # A pace_delta of 0.0 (pole-sitting car) should produce 0 exposure
    # regardless of circuit — there's no gap to weight.
    monza_row = result[result["circuitId"] == "monza"].iloc[0]
    assert monza_row["straight_line_exposure"] == 0.0
    print("  ✓ Zero pace_delta produces zero exposure (no gap to weight)")

    print("\nAll checks passed ✓")
