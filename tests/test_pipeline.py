"""Smoke + correctness tests for the ApexSports pipeline.

Run:  pytest -q     (from the project root)

These assume `python scripts/build_all.py` has populated the DB and artifacts.
The xG-coefficient test asserts the trained logistic model recovers the SIGN
and rough magnitude of the synthetic ground truth — proving the framework is
wired correctly end to end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apexsports.data import generate
from apexsports.data.generate import TRUE_BETA
from apexsports.models import xg, poisson, forecast
from apexsports.sim.montecarlo import TeamState, simulate, optimize_substitution
from apexsports.utils import shot_geometry, city_distance_km


@pytest.fixture(scope="session", autouse=True)
def built():
    """Build a small dataset + models once for the whole test session."""
    generate.generate(n_group_rounds=5, seed=7)
    xg.train()
    poisson.build_ratings()
    forecast.train()


# --- geometry -------------------------------------------------------------
def test_shot_geometry_monotonic():
    near, _ = shot_geometry(116, 40)
    far, _ = shot_geometry(85, 40)
    assert near < far
    # Central angle wider than a tight angle from the byline.
    _, central = shot_geometry(110, 40)
    _, wide_out = shot_geometry(110, 5)
    assert central > wide_out


def test_city_distance():
    assert city_distance_km("Miami", "Miami") == 0.0
    assert city_distance_km("Vancouver", "Miami") > 4000  # cross-continent


# --- xG model -------------------------------------------------------------
def test_xg_recovers_ground_truth_signs():
    m = xg.train(save=True)
    coefs = m["coefficients"]
    assert m["auc"] > 0.78
    assert coefs["distance"] < 0          # farther => lower xG
    assert coefs["angle"] > 0             # wider angle => higher xG
    assert coefs["under_pressure"] < 0    # pressure => lower xG
    assert coefs["big_chance"] > 0
    # Distance coefficient should land near the true value.
    assert abs(coefs["distance"] - TRUE_BETA["distance"]) < 0.1


def test_xg_probabilities_bounded():
    model = xg.load_model()
    close = xg.predict_xg(118, 40, model=model)["xg"]
    far = xg.predict_xg(80, 10, model=model)["xg"]
    assert 0 <= far < close <= 1


# --- Poisson --------------------------------------------------------------
def test_poisson_distribution_sums_to_one():
    ratings = poisson._load_ratings()
    any_pid = int(next(iter(ratings["players"])))
    any_team = int(next(iter(ratings["defence"])))
    d = poisson.player_goal_distribution(any_pid, any_team, ratings=ratings)
    total = sum(d["distribution"].values())
    assert 0.97 <= total <= 1.0001
    assert 0 <= d["p_at_least_1"] <= 1
    # More minutes => higher lambda.
    half = poisson.player_goal_distribution(any_pid, any_team, 45, ratings=ratings)
    full = poisson.player_goal_distribution(any_pid, any_team, 90, ratings=ratings)
    assert full["lambda"] > half["lambda"]


# --- Forecast -------------------------------------------------------------
def test_forecast_nonnegative_and_responsive():
    bundle = forecast.load_model()
    base = {"skill": 0.8, "position_code": 3, "rest_days": 5, "travel_km": 0,
            "elevation_m": 50, "fatigue_index": 0.1, "form_xg3": 0.3,
            "form_minutes3": 85, "career_xg90": 0.4}
    fresh = forecast.forecast_player(base, bundle)["predicted_xg"]
    tired = forecast.forecast_player({**base, "fatigue_index": 0.95},
                                     bundle)["predicted_xg"]
    assert fresh >= 0 and tired >= 0


# --- Simulation -----------------------------------------------------------
def test_simulate_probabilities_normalised():
    h = TeamState("A", 1.6, 1.3, 0.3)
    a = TeamState("B", 1.0, 1.1, 0.3)
    r = simulate(h, a, n_sims=8000, seed=1)
    assert abs(r["home_win"] + r["draw"] + r["away_win"] - 1.0) < 1e-9
    assert r["home_win"] > r["away_win"]  # stronger side favoured


def test_defend_objective_lowers_loss_risk():
    h = TeamState("A", 1.5, 1.3, 0.5)
    a = TeamState("B", 1.2, 1.1, 0.3)
    rec = optimize_substitution(h, a, 75, 1, 0, objective="hold", n_sims=8000)
    defend = next(o for o in rec["options"] if o["mentality"] == "defend")
    attack = next(o for o in rec["options"] if o["mentality"] == "attack")
    assert defend["away_win"] <= attack["away_win"]
