"""Central configuration for ApexSports Analytics.

All paths are anchored to this file's location so the project is portable.
The database URL is SQLAlchemy-based: swap SQLite for Postgres/TimescaleDB by
changing DATABASE_URL only (e.g. postgresql+psycopg://user:pass@host/db).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# --- Database -------------------------------------------------------------
DB_PATH = DATA_DIR / "apexsports.db"
DATABASE_URL = os.environ.get("APEX_DATABASE_URL", f"sqlite:///{DB_PATH}")

# --- Model artifact paths -------------------------------------------------
XG_MODEL_PATH = ARTIFACTS_DIR / "xg_model.joblib"
FORECAST_MODEL_PATH = ARTIFACTS_DIR / "forecast_xgb.joblib"
POISSON_PARAMS_PATH = ARTIFACTS_DIR / "poisson_ratings.json"
LSTM_MODEL_PATH = ARTIFACTS_DIR / "forecast_lstm.pt"
LSTM_SCALER_PATH = ARTIFACTS_DIR / "forecast_lstm_scaler.joblib"

# --- Reproducibility ------------------------------------------------------
RANDOM_SEED = 42

# --- Synthetic tournament context (2026 FIFA World Cup hosts) -------------
# Used by the feature engineering layer for travel / elevation / weather.
HOST_CITIES = {
    "Vancouver":    {"elevation_m": 4,    "lat": 49.28, "lon": -123.12},
    "Seattle":      {"elevation_m": 56,   "lat": 47.61, "lon": -122.33},
    "San Francisco":{"elevation_m": 16,   "lat": 37.77, "lon": -122.42},
    "Los Angeles":  {"elevation_m": 93,   "lat": 34.05, "lon": -118.24},
    "Kansas City":  {"elevation_m": 277,  "lat": 39.10, "lon": -94.58},
    "Dallas":       {"elevation_m": 131,  "lat": 32.78, "lon": -96.80},
    "Houston":      {"elevation_m": 24,   "lat": 29.76, "lon": -95.37},
    "Atlanta":      {"elevation_m": 320,  "lat": 33.75, "lon": -84.39},
    "Miami":        {"elevation_m": 2,    "lat": 25.76, "lon": -80.19},
    "New York":     {"elevation_m": 10,   "lat": 40.71, "lon": -74.01},
    "Boston":       {"elevation_m": 43,   "lat": 42.36, "lon": -71.06},
    "Philadelphia": {"elevation_m": 12,   "lat": 39.95, "lon": -75.17},
    "Toronto":      {"elevation_m": 76,   "lat": 43.65, "lon": -79.38},
    "Mexico City":  {"elevation_m": 2240, "lat": 19.43, "lon": -99.13},
    "Guadalajara":  {"elevation_m": 1566, "lat": 20.67, "lon": -103.35},
    "Monterrey":    {"elevation_m": 540,  "lat": 25.69, "lon": -100.32},
}

# Standard pitch dimensions (StatsBomb convention: 120 x 80), goal at x=120.
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0
GOAL_Y = 40.0
GOAL_WIDTH = 8.0  # ~7.32m mapped to StatsBomb units
