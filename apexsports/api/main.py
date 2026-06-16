"""FastAPI backend exposing the ApexSports predictive engine.

Run:  uvicorn apexsports.api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from apexsports.data.database import get_session
from apexsports.data.schema import Team, Player, Match
from apexsports.models import xg, poisson, forecast, calibration
from apexsports.sim.montecarlo import TeamState, simulate, optimize_substitution

app = FastAPI(
    title="ApexSports Analytics API",
    version="0.1.0",
    description="Live, tournament-driven predictive insights for elite teams.",
)


# --- Lazy-loaded model artifacts (cached across requests) -----------------
@lru_cache(maxsize=1)
def _xg_model():
    return xg.load_model()


@lru_cache(maxsize=1)
def _forecast_bundle():
    return forecast.load_model()


@lru_cache(maxsize=1)
def _poisson_ratings():
    return poisson._load_ratings()


@lru_cache(maxsize=1)
def _lstm_bundle():
    # Imported lazily so the API runs even if torch isn't installed.
    from apexsports.models import lstm_forecast
    return lstm_forecast.load_model()


# --- Request schemas ------------------------------------------------------
class ShotIn(BaseModel):
    x: float = Field(..., ge=0, le=120, description="StatsBomb pitch x (goal at 120)")
    y: float = Field(..., ge=0, le=80)
    is_header: bool = False
    under_pressure: bool = False
    big_chance: bool = False


class ForecastIn(BaseModel):
    skill: float = 0.6
    position_code: int = Field(3, ge=0, le=3, description="0=GK 1=DEF 2=MID 3=FWD")
    rest_days: int = 4
    travel_km: float = 0.0
    elevation_m: float = 0.0
    fatigue_index: float = 0.3
    form_xg3: float = 0.2
    form_minutes3: float = 80.0
    career_xg90: float = 0.3


class PoissonIn(BaseModel):
    player_id: int
    opponent_team_id: int
    expected_minutes: int = Field(90, ge=1, le=120)


class SequenceForecastIn(BaseModel):
    match_history: list[dict] = Field(
        ..., description="Past-match feature dicts, oldest->newest (>= window)")
    upcoming: dict = Field(
        default_factory=dict,
        description="Known pre-match context: minutes, rest_days, travel_km, "
                    "elevation_m, fatigue_index, skill, position_code")


class TeamIn(BaseModel):
    name: str = "Home"
    attack: float = Field(1.4, gt=0)
    defence: float = Field(1.2, gt=0)
    fatigue: float = Field(0.3, ge=0, le=1)
    mentality: str = "balanced"


class SimIn(BaseModel):
    home: TeamIn
    away: TeamIn
    start_minute: int = 0
    home_goals: int = 0
    away_goals: int = 0
    n_sims: int = Field(20000, ge=1000, le=100000)


class SubIn(BaseModel):
    home: TeamIn
    away: TeamIn
    minute: int = 75
    home_goals: int = 1
    away_goals: int = 0
    objective: str = Field("hold", description="win | hold | comeback")


# --- Routes ---------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "apexsports"}


@app.get("/teams")
def list_teams():
    with get_session() as s:
        return [{"id": t.id, "name": t.name,
                 "attack_strength": t.attack_strength,
                 "defence_strength": t.defence_strength}
                for t in s.query(Team).order_by(Team.name).all()]


@app.get("/teams/{team_id}/players")
def team_players(team_id: int):
    with get_session() as s:
        players = s.query(Player).filter(Player.team_id == team_id).all()
        if not players:
            raise HTTPException(404, f"No players for team {team_id}")
        return [{"id": p.id, "name": p.name, "position": p.position,
                 "skill": round(p.skill, 3)} for p in players]


@app.post("/xg")
def expected_goals(shot: ShotIn):
    return xg.predict_xg(shot.x, shot.y, shot.is_header, shot.under_pressure,
                         shot.big_chance, model=_xg_model())


@app.post("/forecast")
def forecast_output(payload: ForecastIn):
    return forecast.forecast_player(payload.model_dump(), bundle=_forecast_bundle())


@app.post("/forecast/sequence")
def forecast_sequence(payload: SequenceForecastIn):
    """LSTM forecast from a sequence of recent matches + upcoming context."""
    try:
        from apexsports.models import lstm_forecast
    except ImportError:
        raise HTTPException(503, "LSTM unavailable — torch not installed.")
    try:
        return lstm_forecast.forecast_sequence(
            payload.match_history, payload.upcoming, bundle=_lstm_bundle())
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))


@app.post("/poisson/player-goals")
def poisson_player_goals(payload: PoissonIn):
    try:
        return poisson.player_goal_distribution(
            payload.player_id, payload.opponent_team_id,
            payload.expected_minutes, ratings=_poisson_ratings())
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/calibration")
def xg_calibration(n_bins: int = 10):
    """Reliability curves + scores for our xG model vs StatsBomb's."""
    return calibration.compare(n_bins=n_bins, model=_xg_model())


def _to_state(t: TeamIn) -> TeamState:
    return TeamState(t.name, t.attack, t.defence, t.fatigue, t.mentality)


@app.post("/simulate")
def simulate_match(payload: SimIn):
    return simulate(_to_state(payload.home), _to_state(payload.away),
                    payload.start_minute, payload.home_goals,
                    payload.away_goals, payload.n_sims)


@app.post("/optimize/substitution")
def optimize_sub(payload: SubIn):
    return optimize_substitution(
        _to_state(payload.home), _to_state(payload.away), payload.minute,
        payload.home_goals, payload.away_goals, payload.objective)
