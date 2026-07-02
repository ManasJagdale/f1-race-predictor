# F1 Race Outcome Predictor

A data-driven ML pipeline that predicts full finishing position distributions (P1–P20) for every driver in an F1 race, validated against historical results (2020–2026) and designed to feed into prediction-market signal engines (e.g. Polymarket).

## Problem

F1 outcomes are partially predictable — grid position, car pace, driver skill, and track characteristics all carry signal — but no clean open-source tool combines these into a calibrated probabilistic model. This project builds that pipeline end-to-end: raw historical data → feature engineering → ML ranking model → Monte Carlo race simulation → backtested probability output.

## What it does

Given a race (e.g. `--race "British Grand Prix 2024"`), the model:

1. Pulls pre-race features from the Jolpica API and FastF1
2. Scores each driver with a trained ML ranking model
3. Runs 10,000 Monte Carlo simulations of the race (sampling DNFs, safety cars, pit stop strategies from historical distributions)
4. Outputs a full P1–P20 probability distribution per driver

```
Driver              Team         P(Win)   P(P1-3)  P(P1-6)  P(DNF)
─────────────────────────────────────────────────────────────────
Norris, L           McLaren      31.2%    68.4%    89.1%    2.1%
Verstappen, M       Red Bull     28.7%    61.3%    82.4%    3.4%
Hamilton, L         Mercedes     11.3%    38.2%    67.3%    2.8%
...
```

## Architecture

```
Jolpica API          FastF1
(results, quali,     (telemetry, track
 DNF history)         features, weather)
      │                    │
      └────────┬───────────┘
               ↓
     feature_engineering/
     ├── driver_ratings.py        Elo rating + recent form + teammate delta
     ├── car_performance.py       Constructor pace delta + DNF rate (empirical Bayes shrinkage)
     ├── track_features.py        FastF1 track descriptors + weather
     ├── circuit_profiler.py      Per-sector circuit classification
     └── track_car_interaction.py Track × car interaction features
               ↓
        models/race_model.py      8-seed ensemble, pairwise learning-to-rank
               ↓
        simulation/simulation.py  Monte Carlo × 10,000 runs
               ↓
        backtest.py               Brier-score validated, 78 races
        oracle.py                 CLI — terminal / CSV / heatmap output
        predict_upcoming.py       Forecasts future (unraced) rounds
```

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Driver skill signal | Elo + recent form + teammate delta | Teammate delta isolates driver skill from car performance (same machinery, same conditions) |
| Car performance signal | Pace delta, not team identity | Prevents the model from memorizing team names instead of learning *why* they're fast |
| Model architecture | Pairwise learning-to-rank | Race prediction is fundamentally a ranking problem, not absolute regression |
| Model robustness | 8-seed ensemble | Single-seed results were seed-dependent; only 5/8 seeds beat the naive baseline alone |
| DNF rate | Empirical Bayes shrinkage (K=15) | Raw DNF rates are unreliable for small-sample / new teams |
| Training window | 2020–2022 | Widening to 2018 introduced pre-2022-reset noise and hurt performance |

## Validation

- **Baseline:** naive model (qualifying position = finishing position) — a deliberately strong benchmark, since pole position alone predicts a large share of race wins.
- **Result:** the full model beats the naive baseline with a **+0.467 Brier skill score across 78 backtested races**, stable across 1,000–25,000 Monte Carlo runs.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Configure paths / season ranges / Elo constants in `config.py`.

## Usage

```bash
# Fetch and build features
python data/fetch_jolpica.py
python data/build_master_features.py

# Predict a past or ongoing race
python oracle.py --race "British Grand Prix 2024"

# Forecast an upcoming (unraced) round
python predict_upcoming.py --race "British Grand Prix 2026"

# Backtest
python backtest.py
```

## Project status

v1 complete and independently validated end-to-end, from data ingestion through live prediction of both past and upcoming races. See `f1_predictor_spec.md` for the full public-facing spec and `f1_predictor_session_notes.md` for the build history and design log.

## Roadmap (Phase 2)

- Telemetry-based car decomposition (straight-line vs cornering speed per team)
- Mixed-effects model for driver/car separation
- Full tyre strategy simulation (undercut/overcut, compound degradation)
- Qualifying-position predictor to remove the last pre-race input dependency
- Polymarket odds comparison + Kelly-capped stake sizing

## License

Add a license of your choice (MIT is a common default for portfolio projects) — see [choosealicense.com](https://choosealicense.com).
