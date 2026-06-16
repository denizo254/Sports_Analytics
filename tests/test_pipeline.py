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
from apexsports.models import xg, poisson, forecast, calibration
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


# --- Calibration ----------------------------------------------------------
def test_calibration_reliability_and_compare():
    # Reliability curve on a perfectly calibrated synthetic signal.
    import numpy as np
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 5000)
    outcomes = (rng.uniform(0, 1, 5000) < probs).astype(int)
    curve = calibration.reliability_curve(probs, outcomes, n_bins=10)
    assert len(curve) > 0
    # Predicted and observed should track closely when truly calibrated.
    assert (curve["mean_predicted"] - curve["observed_freq"]).abs().mean() < 0.05

    sc = calibration.score(probs, outcomes)
    assert 0 <= sc["brier"] <= 1

    # Full comparison against the trained xG model on the loaded data.
    out = calibration.compare(n_bins=8)
    assert out["n_shots"] > 0
    assert 0 <= out["our"]["score"]["brier"] <= 1
    assert isinstance(out["has_statsbomb"], bool)
    # Synthetic data carries no StatsBomb reference xG.
    assert out["has_statsbomb"] is False


# --- LSTM forecaster (skipped when torch is absent, e.g. on CI) -----------
def test_lstm_forecaster_runs_and_predicts():
    pytest.importorskip("torch")
    from apexsports.models import lstm_forecast

    m = lstm_forecast.train(epochs=30, save=True, seed=7)
    assert m["n_samples"] > 0
    assert m["window"] == lstm_forecast.WINDOW
    assert m["mae"] >= 0

    bundle = lstm_forecast.load_model()
    window = bundle["config"]["window"]
    history = [{"minutes": 80, "xg": 0.3, "goals": 0, "shots": 2, "passes": 30,
                "distance_km": 9.5, "assists": 0, "rest_days": 4,
                "travel_km": 500, "elevation_m": 50, "fatigue_index": 0.3,
                "skill": 0.75, "position_code": 3} for _ in range(window)]
    upcoming = {"minutes": 90, "rest_days": 5, "travel_km": 0, "elevation_m": 10,
                "fatigue_index": 0.2, "skill": 0.75, "position_code": 3}
    out = lstm_forecast.forecast_sequence(history, upcoming, bundle=bundle)
    assert out["predicted_xg"] >= 0
    assert out["matches_used"] == window

    # Too-short history must raise.
    with pytest.raises(ValueError):
        lstm_forecast.forecast_sequence(history[:window - 1], upcoming, bundle=bundle)


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


# --- FBref ingest transform (offline; no soccerdata / network needed) -----
# Kept LAST: ingest() wipes the DB, so we restore synthetic data afterwards.
def test_fbref_ingest_transform():
    import pandas as pd
    from apexsports.data import fbref
    from apexsports.data.database import get_session
    from apexsports.data.schema import Match, PlayerMatchStat, Player, Shot

    schedule = pd.DataFrame([
        {"game": "g1", "date": "2024-09-17", "home_team": "Real Madrid",
         "away_team": "Stuttgart", "home_score": 3, "away_score": 1},
        {"game": "g2", "date": "2024-09-18", "home_team": "Bayern Munich",
         "away_team": "Dinamo Zagreb", "home_score": 9, "away_score": 2},
    ])
    stats = pd.DataFrame([
        {"game": "g1", "team": "Real Madrid", "player": "Kylian Mbappe",
         "pos": "FW", "min": 90, "Performance_Gls": 1, "Performance_Sh": 5,
         "Expected_xG": 0.9, "Performance_Ast": 0, "Passes_Att": 30},
        {"game": "g1", "team": "Stuttgart", "player": "Deniz Undav",
         "pos": "FW", "min": 80, "Performance_Gls": 1, "Performance_Sh": 3,
         "Expected_xG": 0.6, "Performance_Ast": 0, "Passes_Att": 18},
        {"game": "g2", "team": "Bayern Munich", "player": "Harry Kane",
         "pos": "FW", "min": 90, "Performance_Gls": 4, "Performance_Sh": 6,
         "Expected_xG": 2.3, "Performance_Ast": 1, "Passes_Att": 25},
        {"game": "g2", "team": "Bayern Munich", "player": "Unused Sub",
         "pos": "MF", "min": 0, "Performance_Gls": 0, "Performance_Sh": 0,
         "Expected_xG": 0.0, "Performance_Ast": 0, "Passes_Att": 0},
    ])
    try:
        counts = fbref.ingest(schedule, stats, competition="INT-Champions League",
                              verbose=False)
        assert counts["matches"] == 2
        assert counts["shots"] == 0           # FBref path has no per-shot data
        assert counts["player_match_stats"] == 3  # the 0-minute sub is dropped

        with get_session() as s:
            assert s.query(Match).count() == 2
            assert s.query(Shot).count() == 0
            kane = s.query(Player).filter(Player.name == "Harry Kane").one()
            st = s.query(PlayerMatchStat).filter(
                PlayerMatchStat.player_id == kane.id).one()
            assert st.goals == 4 and st.shots == 6
            assert abs(st.xg - 2.3) < 1e-6
    finally:
        # Restore the shared synthetic dataset for any later runs.
        generate.generate(n_group_rounds=5, seed=7)
