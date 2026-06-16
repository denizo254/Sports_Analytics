"""Monte Carlo match simulator + in-game substitution optimizer.

Models the remainder of a match as competing Poisson goal processes whose
minute-by-minute intensity is modulated by team strength, current game state,
and player fatigue. Used to (a) produce win/draw/loss + scoreline
probabilities, and (b) recommend the substitution window that maximises the
desired objective (e.g. protecting a 1-0 lead at the 75th minute).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TeamState:
    name: str
    attack: float          # goals/90 baseline when fresh
    defence: float         # >1 = stronger defence, suppresses opponent
    fatigue: float = 0.0   # 0..1, scales attack down late in games
    mentality: str = "balanced"  # "attack" | "balanced" | "defend"


MENTALITY_ATK = {"attack": 1.25, "balanced": 1.0, "defend": 0.72}
MENTALITY_DEF = {"attack": 0.85, "balanced": 1.0, "defend": 1.30}


def _intensities(team: TeamState, opp: TeamState, minute: int):
    """Per-minute scoring intensity for `team` given game context."""
    fatigue_penalty = 1.0 - 0.30 * team.fatigue * (minute / 90.0)
    atk = (team.attack / 90.0) * MENTALITY_ATK[team.mentality] * fatigue_penalty
    suppression = opp.defence * MENTALITY_DEF[opp.mentality]
    return max(1e-5, atk / suppression)


def simulate(home: TeamState, away: TeamState, start_minute: int = 0,
             home_goals: int = 0, away_goals: int = 0,
             n_sims: int = 20000, seed: int = 42) -> dict:
    """Monte Carlo the remaining minutes; return outcome probabilities."""
    rng = np.random.default_rng(seed)
    minutes_left = max(0, 90 - start_minute)

    h_final = np.full(n_sims, home_goals, dtype=int)
    a_final = np.full(n_sims, away_goals, dtype=int)

    # Vectorised minute-by-minute Bernoulli goal draws.
    for minute in range(start_minute, 90):
        lam_h = _intensities(home, away, minute)
        lam_a = _intensities(away, home, minute)
        h_final += (rng.random(n_sims) < lam_h).astype(int)
        a_final += (rng.random(n_sims) < lam_a).astype(int)

    home_win = float(np.mean(h_final > a_final))
    draw = float(np.mean(h_final == a_final))
    away_win = float(np.mean(h_final < a_final))

    # Top scorelines.
    pairs, counts = np.unique(
        np.stack([h_final, a_final], axis=1), axis=0, return_counts=True)
    order = np.argsort(-counts)[:5]
    top = [{"score": f"{int(pairs[i][0])}-{int(pairs[i][1])}",
            "prob": round(float(counts[i] / n_sims), 4)} for i in order]

    return {
        "minutes_simulated": minutes_left,
        "home_win": round(home_win, 4),
        "draw": round(draw, 4),
        "away_win": round(away_win, 4),
        "exp_home_goals": round(float(h_final.mean()), 3),
        "exp_away_goals": round(float(a_final.mean()), 3),
        "top_scorelines": top,
    }


def optimize_substitution(home: TeamState, away: TeamState, minute: int,
                          home_goals: int, away_goals: int,
                          objective: str = "win", n_sims: int = 15000) -> dict:
    """Evaluate mentality switches for the home team and rank by objective.

    objective: "win" maximises P(win); "hold" maximises P(not losing) — the
    classic 'defend a 1-0 lead' scenario; "comeback" maximises P(win|behind).
    """
    options = []
    base_mentality = home.mentality
    for mentality in ("attack", "balanced", "defend"):
        trial = TeamState(home.name, home.attack, home.defence,
                          fatigue=max(0.0, home.fatigue - 0.15),  # fresh legs
                          mentality=mentality)
        res = simulate(trial, away, start_minute=minute,
                       home_goals=home_goals, away_goals=away_goals,
                       n_sims=n_sims)
        if objective == "hold":
            score = res["home_win"] + res["draw"]
        elif objective == "comeback":
            score = res["home_win"]
        else:
            score = res["home_win"]
        options.append({"mentality": mentality, "objective_score": round(score, 4),
                        "home_win": res["home_win"], "draw": res["draw"],
                        "away_win": res["away_win"]})

    options.sort(key=lambda o: o["objective_score"], reverse=True)
    best = options[0]
    return {
        "minute": minute, "score": f"{home_goals}-{away_goals}",
        "objective": objective,
        "recommendation": best["mentality"],
        "rationale": (f"Switching to '{best['mentality']}' with fresh legs "
                      f"maximises the '{objective}' objective at "
                      f"{best['objective_score']:.1%}."),
        "options": options,
        "previous_mentality": base_mentality,
    }


if __name__ == "__main__":
    import json
    h = TeamState("Brazil", attack=1.7, defence=1.25, fatigue=0.5)
    a = TeamState("Croatia", attack=1.1, defence=1.2, fatigue=0.3)
    print(json.dumps(simulate(h, a), indent=2))
    print(json.dumps(optimize_substitution(h, a, 75, 1, 0, objective="hold"), indent=2))
