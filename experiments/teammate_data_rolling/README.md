# Experiment: rolling-window `teammate_delta`

**What this tests:** whether giving `teammate_delta` the same kind of
recency treatment `elo_pre_race` (0.97/month decay) and `recent_form`
(hard 5-race window) already have — instead of its current unweighted
career average — improves the model's Brier skill score, and fixes the
Antonelli-at-Silverstone anomaly (pole position, rank 6/22 by total
strength score, 9.6% gap from the leader).

**Isolation guarantee:** every script in this folder only reads from
your real project's cached data (`data/cache/*.parquet`,
`circuit_profiles.xlsx`) — all read-only, nothing is ever modified.
Every write goes to a file inside *this* folder
(`master_features_v2.parquet`, `race_model_v2.joblib`,
`backtest_results_v2.parquet`, `output_v2/`). Your real
`master_features.parquet`, `driver_ratings.parquet`, and
`race_model.joblib` are never read from or written to.

**To delete this experiment entirely:** delete this whole folder. That's it.
Nothing outside it is affected.

## What changed

Exactly one function, in `driver_ratings_v2.py`:

```python
# Original (career average, no recency):
rolling_delta = sum(history) / len(history)

# v2 (rolling window, same pattern as recent_form's FORM_WINDOW):
window = history[-TEAMMATE_DELTA_WINDOW:]
rolling_delta = sum(window) / len(window)
```

`TEAMMATE_DELTA_WINDOW` defaults to `8` at the top of
`driver_ratings_v2.py`. Worth trying `5` (exact parity with
`FORM_WINDOW`) or `12` as further variants if the first pass looks
promising — just edit the constant and re-run steps 1–3 below.

## Run order

From inside this folder:

```bash
# 1. Rebuild features with the rolling teammate_delta (a few minutes)
python build_master_features_v2.py

# 2. Retrain the 8-seed ensemble on the v2 features
python train_v2.py

# 3. Run the REAL comparison metric — Brier skill score
python backtest_v2.py
```

Step 3's output is what actually decides this. Compare the "Full
P1-Pn distribution" skill score against the real project's documented
baseline of **+0.465** (78 races, `Final_Project_History.md` section
7.5):

- **Higher than +0.465** → the change is a real improvement, worth
  porting into the real `driver_ratings.py`.
- **Lower / negative** → the career-average version was doing
  something useful that the rolling window loses (e.g. smoothing out
  single-race teammate variance) — don't port it, despite it being
  more internally consistent with the other two driver features.
- **About the same** → inconclusive on this metric alone; still worth
  checking Antonelli's case specifically (step 4) since a wash on
  aggregate Brier doesn't rule out a real, localized fix for rookies.

```bash
# 4. Re-check the specific Antonelli case (needs your grid CSV --
#    adjust the relative path if you copied it somewhere else)
python diagnose_field_scores_v2.py --season 2026 --round 9 \
    --grid-csv ../../silverstone_2026_grid.csv --focus antonelli
```

Compare the printed rank/gap against the original: **rank 6/22, 9.6%
gap from the leader, 52.5% average vs the front pack.** If v2 narrows
that gap and/or improves his rank, the change is doing what it was
designed to do specifically for a driver early in their career on a
hot streak.

```bash
# 5. Optional: full live prediction table, same as the real
#    predict_upcoming.py, for a side-by-side view of every driver
python predict_upcoming_v2.py --season 2026 --round 9 \
    --grid-csv ../../silverstone_2026_grid.csv --runs 10000
```

## Reading a mixed result

It's entirely possible for step 3 to come back roughly flat or even
slightly worse while step 4 shows Antonelli's specific gap narrowing.
That would mean the fix helps the exact case it was designed for, but
costs something elsewhere in the 78-race test set (e.g. it may make
established drivers' `teammate_delta` more sensitive to one bad race
than the career average was). Worth knowing before deciding to port
this into the real project — the honest tradeoff, not just "did the
one anomaly get fixed."
