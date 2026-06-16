# ⚽ ApexSports Analytics

[![CI](https://github.com/denizo254/Sports_Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/denizo254/Sports_Analytics/actions/workflows/ci.yml)

**Live, tournament-driven predictive insights for elite coaches and teams.**

An end-to-end sports-analytics platform built around the 2026 FIFA World Cup:
it ingests team/player/shot data, trains predictive models (xG, Poisson
player-goals, XGBoost performance forecasting), simulates in-game scenarios,
and serves everything through a FastAPI backend and a Streamlit dashboard.

> **Status:** fully runnable end-to-end on synthetic data today. The synthetic
> generator mirrors the StatsBomb event grain, so swapping in real StatsBomb
> open data (or a live feed) requires no model or UI changes.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate data + train all models (writes ./data/apexsports.db + ./artifacts)
python scripts/build_all.py

# 2a. Launch the API           -> http://127.0.0.1:8000/docs
uvicorn apexsports.api.main:app --reload

# 2b. Launch the dashboard     -> http://localhost:8501
streamlit run apexsports/dashboard/app.py

# Run the tests
pytest -q
```

---

## Architecture

```
[ Synthetic generator ]                         (StatsBomb-grain events)
[ StatsBomb open data  ]  ──►  [ SQLAlchemy / SQLite ]  ──►  models + sim
        (optional)                  (swap → Postgres/TimescaleDB)
                                            │
[ Streamlit dashboard ]  ◄──  [ FastAPI ]  ◄┘  Scikit-Learn / XGBoost / SciPy
```

The blueprint's Kafka + TimescaleDB + live-feed + cloud stack maps cleanly:

| Blueprint component        | This build (runnable today)        | Upgrade path                              |
|----------------------------|------------------------------------|-------------------------------------------|
| Live API feed + Kafka      | Synthetic generator                | `apexsports/data/statsbomb.py`, then Kafka consumer writing to the same schema |
| PostgreSQL + TimescaleDB   | SQLite via SQLAlchemy              | Set `APEX_DATABASE_URL=postgresql+psycopg://…` |
| Scikit-Learn / XGBoost / PyTorch | LogReg xG, XGBoost forecaster | Add the LSTM variant on the same feature frame |
| FastAPI + Streamlit        | Both included                      | Containerise + deploy to Render/GCP/AWS   |

Swap to Postgres with **one env var** — no code change:
```bash
export APEX_DATABASE_URL="postgresql+psycopg://user:pass@host:5432/apex"
```

---

## Models

### 1. Expected Goals (xG) — logistic regression
`apexsports/models/xg.py`. Fits `log(p/(1−p)) = β₀ + β₁·distance + β₂·angle +
β₃·header + β₄·pressure + β₅·big_chance` on shot geometry.
Validation: on synthetic data the model **recovers the ground-truth
coefficients** (asserted in `tests/test_pipeline.py`), AUC ≈ 0.85.

### 2. Player goal distribution — Poisson
`apexsports/models/poisson.py`. `P(X=k) = λᵏe^(−λ)/k!` where λ is the player's
shrunk goals-per-90 scaled by expected minutes and the opponent's defensive
factor. Returns the full distribution plus P(≥1) and P(brace+).

### 3. Performance forecasting — XGBoost
`apexsports/models/forecast.py`. Predicts a player's next-match xG from recent
form + tournament context (fatigue index, rest days, travel km, elevation).
Top learned features: skill, position, rest days.

> **Honest note on forecasting accuracy:** single-match xG is intrinsically
> high-variance, so the tabular baseline's test R² is modest-but-positive
> (≈0.07). It captures the *directional* effects of fatigue/rest/travel rather
> than pinpoint values — the realistic ceiling for match-level prediction.

### 3b. Performance forecasting — LSTM sequence model (PyTorch)
`apexsports/models/lstm_forecast.py`. The time-series counterpart to the
XGBoost baseline. It encodes a player's ordered sequence of recent matches with
an LSTM, then **fuses the upcoming match's known pre-match context** (expected
minutes, rest days, travel, elevation, fatigue) before the prediction head:

```
past matches ──LSTM──► final hidden state ┐
                                          ├─ concat ─► MLP head ─► xG
upcoming match known context ─────────────┘
```

Trained with a fit/validation/test split, **early stopping** (restore best-val
weights), dropout and weight decay. On the synthetic data it reaches
**R² ≈ 0.22 / MAE ≈ 0.19**, outperforming the tabular baseline. `torch` is an
optional dependency: the build step, API route and dashboard panel all degrade
gracefully when it is absent (and the LSTM test is skipped on CI).

### 4. Monte Carlo match sim + substitution optimizer
`apexsports/sim/montecarlo.py`. Simulates remaining minutes as competing
Poisson goal processes modulated by strength, game state, mentality and
fatigue. `optimize_substitution(...)` ranks mentality switches for objectives
like `hold` (protect a 1-0 lead), `win`, or `comeback`.

---

## API endpoints

| Method | Path                          | Purpose                              |
|--------|-------------------------------|--------------------------------------|
| GET    | `/health`                     | Liveness                             |
| GET    | `/teams`                      | List teams + strengths               |
| GET    | `/teams/{id}/players`         | Squad                                |
| POST   | `/xg`                         | xG for a shot location               |
| POST   | `/forecast`                   | Project player xG (XGBoost)          |
| POST   | `/forecast/sequence`          | Project player xG (LSTM, sequence)   |
| POST   | `/poisson/player-goals`       | Player goal distribution             |
| POST   | `/simulate`                   | Monte Carlo match outcome            |
| POST   | `/optimize/substitution`      | Recommend mentality switch           |

Interactive docs at `/docs` once the server is running.

---

## Using real StatsBomb data

The loader populates the **same schema** as the synthetic generator, so all
models and the dashboard work unchanged on real World Cup data.

```bash
pip install statsbombpy

# Load + train on the 2022 FIFA World Cup (competition 43, season 106)
python scripts/build_all.py --source statsbomb

# Any open-data competition/season:
python scripts/build_all.py --source statsbomb --competition 43 --season 3   # 2018 WC
```

What it derives from the raw event feed:

| Field            | Source                                              |
|------------------|-----------------------------------------------------|
| shots + geometry | shot `location` → distance/angle                    |
| xG               | StatsBomb's own `shot_statsbomb_xg`                 |
| goals            | `shot_outcome == Goal`                              |
| minutes          | Starting XI + Substitution events                   |
| passes / assists | Pass counts, `pass_goal_assist`                     |
| rest_days        | per-team fixture schedule                           |
| player skill     | career xG-per-shot proxy                            |

Not in open data (set to 0, documented): `travel_km`, `elevation_m`,
`distance_km` — use the synthetic generator to exercise those context features.

StatsBomb open data is free for non-commercial use — see
<https://github.com/statsbomb/open-data> for competition IDs and the licence.

---

## Project layout

```
config.py                     paths, DB URL, host-city geo, pitch constants
apexsports/
  utils.py                    shot geometry + haversine travel distance
  data/
    schema.py                 SQLAlchemy ORM (StatsBomb grain)
    database.py               engine / session factory
    generate.py               synthetic tournament generator (known ground truth)
    statsbomb.py              optional real StatsBomb loader
  models/
    xg.py                     logistic xG
    poisson.py                Poisson player goals
    forecast.py               XGBoost performance forecasting
    lstm_forecast.py          LSTM sequence forecaster (PyTorch)
  sim/montecarlo.py           match sim + substitution optimizer
  api/main.py                 FastAPI backend
  dashboard/app.py            Streamlit UI (5 tabs)
scripts/build_all.py          one-command pipeline
tests/test_pipeline.py        smoke + correctness tests
```
