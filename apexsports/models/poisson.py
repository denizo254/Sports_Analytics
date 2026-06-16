"""Poisson player-goal model.

P(X = k) = (lambda^k * e^-lambda) / k!

lambda (a player's expected goals in a fixture) is derived dynamically by
weighting the player's historical scoring rate against the opponent's defensive
strength, then scaled by expected minutes. Returns the full goal distribution
and derived markets (P(>=1), most-likely scoreline contribution, etc.).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.stats import poisson

from config import POISSON_PARAMS_PATH
from apexsports.data.database import get_session
from apexsports.data.schema import Player, PlayerMatchStat, Team

# League-average goals per 90 — Poisson base rate, refined per player below.
LEAGUE_AVG_G90 = 0.38


def build_ratings(save: bool = True) -> dict:
    """Compute each player's goals-per-90 and each team's defensive factor."""
    with get_session() as s:
        stat_rows = s.query(
            PlayerMatchStat.player_id, PlayerMatchStat.goals,
            PlayerMatchStat.minutes,
        ).all()
        players = {p.id: (p.name, p.team_id, p.position, p.skill)
                   for p in s.query(Player).all()}
        teams = {t.id: (t.name, t.defence_strength)
                 for t in s.query(Team).all()}

    df = pd.DataFrame(stat_rows, columns=["player_id", "goals", "minutes"])
    agg = df.groupby("player_id").agg(
        goals=("goals", "sum"), minutes=("minutes", "sum")).reset_index()

    # Shrunk goals-per-90 (Bayesian prior toward league avg to tame small N).
    prior_minutes = 180.0
    ratings = {}
    for _, r in agg.iterrows():
        pid = int(r.player_id)
        g90 = (r.goals + LEAGUE_AVG_G90 * prior_minutes / 90) / \
              ((r.minutes + prior_minutes) / 90)
        name, team_id, pos, skill = players[pid]
        ratings[pid] = {"name": name, "team_id": team_id, "position": pos,
                        "g90": round(float(g90), 4), "skill": round(skill, 3)}

    league_def = float(np.mean([d for _, d in teams.values()]))
    defence = {tid: {"name": n, "factor": round(d / league_def, 4)}
               for tid, (n, d) in teams.items()}

    out = {"players": ratings, "defence": defence,
           "league_avg_def": round(league_def, 4)}
    if save:
        POISSON_PARAMS_PATH.write_text(json.dumps(out, indent=2))
    return out


def _load_ratings() -> dict:
    if not POISSON_PARAMS_PATH.exists():
        raise FileNotFoundError("Poisson ratings missing. Run scripts/build_all.py")
    return json.loads(POISSON_PARAMS_PATH.read_text())


def player_goal_distribution(player_id: int, opponent_team_id: int,
                             expected_minutes: int = 90,
                             max_goals: int = 5, ratings: dict | None = None) -> dict:
    """Return the Poisson goal distribution for a player vs a given opponent."""
    ratings = ratings or _load_ratings()
    p = ratings["players"].get(str(player_id)) or ratings["players"].get(player_id)
    if p is None:
        raise KeyError(f"Unknown player_id {player_id}")

    # Opponent defensive factor (>1 = leakier defence => higher lambda).
    dfn = ratings["defence"].get(str(opponent_team_id)) or \
        ratings["defence"].get(opponent_team_id)
    opp_factor = dfn["factor"] if dfn else 1.0

    lam = p["g90"] * (expected_minutes / 90.0) * opp_factor
    lam = max(1e-4, lam)

    ks = list(range(max_goals + 1))
    pmf = [float(poisson.pmf(k, lam)) for k in ks]
    return {
        "player_id": player_id,
        "player": p["name"],
        "lambda": round(lam, 4),
        "expected_minutes": expected_minutes,
        "opponent_defence_factor": round(opp_factor, 4),
        "distribution": {str(k): round(v, 4) for k, v in zip(ks, pmf)},
        "p_at_least_1": round(1 - math.exp(-lam), 4),
        "p_brace_plus": round(float(1 - poisson.cdf(1, lam)), 4),
    }


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in build_ratings().items()
                      if k != "players"}, indent=2))
