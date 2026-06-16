"""Synthetic data generator mirroring the StatsBomb event grain.

Generates a 2026-World-Cup-shaped dataset (teams, squads, fixtures across the
real host cities, shot events, and per-player match stats) with a *known*
ground-truth goal-probability surface. Because the data is generated from a
logistic function of distance/angle/header/pressure, the xG model trained on
it provably recovers that surface — useful for validating the pipeline end to
end before plugging in real StatsBomb feeds.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import numpy as np

from config import RANDOM_SEED, HOST_CITIES, PITCH_LENGTH, PITCH_WIDTH
from apexsports.data.database import get_session, init_db
from apexsports.data.schema import (
    Team, Player, Match, Shot, PlayerMatchStat,
)
from apexsports.utils import shot_geometry, city_distance_km

# 16 national teams with rough attack/defence strengths (1.0 = average).
TEAMS = [
    ("Brazil", 1.45, 1.30), ("France", 1.42, 1.32), ("Argentina", 1.44, 1.28),
    ("Spain", 1.40, 1.25), ("England", 1.35, 1.30), ("Germany", 1.33, 1.22),
    ("Portugal", 1.34, 1.18), ("Netherlands", 1.30, 1.24), ("Mexico", 1.10, 1.05),
    ("USA", 1.08, 1.02), ("Croatia", 1.12, 1.15), ("Morocco", 1.05, 1.18),
    ("Japan", 1.06, 1.00), ("Canada", 1.00, 0.95), ("Senegal", 1.07, 1.03),
    ("South Korea", 1.02, 0.98),
]

POSITIONS = (["GK"] * 3 + ["DEF"] * 7 + ["MID"] * 7 + ["FWD"] * 6)  # 23-man squad
CITIES = list(HOST_CITIES.keys())

# Ground-truth xG coefficients (logit space). The xG model should approximate
# these after training. distance lowers p, angle raises p, headers lower p.
TRUE_BETA = {
    "intercept": 0.55,
    "distance": -0.11,
    "angle": 1.9,
    "header": -0.45,
    "pressure": -0.50,
    "big_chance": 1.30,
}


def _true_goal_prob(distance, angle, header, pressure, big_chance, skill):
    z = (TRUE_BETA["intercept"]
         + TRUE_BETA["distance"] * distance
         + TRUE_BETA["angle"] * angle
         + TRUE_BETA["header"] * header
         + TRUE_BETA["pressure"] * pressure
         + TRUE_BETA["big_chance"] * big_chance
         + 1.1 * (skill - 0.5))  # finishing skill nudges conversion
    return 1.0 / (1.0 + math.exp(-z))


def _sample_shot_location(rng: random.Random):
    """Sample a plausible shot location, biased toward the box."""
    if rng.random() < 0.65:  # inside/around the box
        x = rng.uniform(100, 119)
        y = rng.uniform(24, 56)
    else:                    # long-range
        x = rng.uniform(78, 100)
        y = rng.uniform(14, 66)
    return x, y


def generate(n_group_rounds: int = 7, seed: int = RANDOM_SEED) -> dict:
    """Populate the database with a full synthetic tournament. Returns counts."""
    rng = random.Random(seed)
    np.random.seed(seed)

    init_db()
    with get_session() as s:
        # Wipe any prior run for idempotency.
        for model in (Shot, PlayerMatchStat, Match, Player, Team):
            s.query(model).delete()
        s.flush()

        # --- Teams & squads ------------------------------------------------
        teams: list[Team] = []
        for name, atk, dfc in TEAMS:
            t = Team(name=name, attack_strength=atk, defence_strength=dfc)
            s.add(t)
            teams.append(t)
        s.flush()

        players_by_team: dict[int, list[Player]] = {}
        for t in teams:
            squad = []
            for i, pos in enumerate(POSITIONS):
                base = {"GK": 0.05, "DEF": 0.20, "MID": 0.45, "FWD": 0.72}[pos]
                skill = float(np.clip(rng.gauss(base, 0.10), 0.02, 0.97))
                p = Player(name=f"{t.name} {pos}{i+1}", team_id=t.id,
                           position=pos, skill=skill)
                s.add(p)
                squad.append(p)
            players_by_team[t.id] = squad
        s.flush()

        # --- Fixtures: round-robin-ish group games + knockout flavour ------
        start = datetime(2026, 6, 11)
        matches: list[Match] = []
        last_city: dict[int, str] = {}
        last_date: dict[int, datetime] = {}

        pairings = []
        for rnd in range(n_group_rounds):
            shuffled = teams[:]
            rng.shuffle(shuffled)
            for i in range(0, len(shuffled), 2):
                pairings.append((shuffled[i], shuffled[i + 1], rnd, "Group"))

        match_day = start
        for idx, (home, away, rnd, stage) in enumerate(pairings):
            city = rng.choice(CITIES)
            match_day = start + timedelta(days=rnd * 4 + (idx % 4))
            m = Match(date=match_day, stage=stage, home_team_id=home.id,
                      away_team_id=away.id, city=city)
            s.add(m)
            s.flush()
            matches.append(m)

            _simulate_match(s, rng, m, home, away, players_by_team,
                            last_city, last_date)

        counts = {
            "teams": len(teams),
            "players": sum(len(v) for v in players_by_team.values()),
            "matches": len(matches),
            "shots": s.query(Shot).count(),
            "player_match_stats": s.query(PlayerMatchStat).count(),
        }
    return counts


def _simulate_match(s, rng, m, home, away, players_by_team,
                    last_city, last_date):
    """Simulate shots + player stats for one match and write the score."""
    score = {home.id: 0, away.id: 0}

    for team, opp in ((home, away), (away, home)):
        squad = players_by_team[team.id]
        shooters = [p for p in squad if p.position in ("FWD", "MID", "DEF")]

        # Expected shot volume from attack vs opponent defence.
        lam_shots = 11.0 * team.attack_strength / opp.defence_strength
        n_shots = max(1, np.random.poisson(lam_shots))

        # Tournament context for this team's players.
        prev_city = last_city.get(team.id, m.city)
        travel = city_distance_km(prev_city, m.city)
        rest = max(2, (m.date - last_date[team.id]).days) if team.id in last_date else 5
        elevation = HOST_CITIES[m.city]["elevation_m"]
        last_city[team.id] = m.city
        last_date[team.id] = m.date

        per_player = {p.id: {"shots": 0, "goals": 0, "xg": 0.0} for p in squad}

        # Decide minutes & fatigue up front so they can drive shot allocation:
        # a fresher player with more minutes earns proportionally more chances.
        # This makes fatigue/travel/rest genuine predictors of output, which is
        # exactly what the XGBoost forecaster needs to learn real signal.
        ctx = {}
        for p in squad:
            minutes = 90 if p.position == "GK" else rng.choice(
                [90, 90, 90, 78, 65, 30, 0])
            dist_km = round(max(0.0, rng.gauss(10.5, 1.3)) * minutes / 90, 2)
            fatigue = round(
                0.4 * (dist_km / 11.0)
                + 0.3 * (travel / 4000.0)
                + 0.2 * (1.0 if rest <= 3 else 0.4)
                + 0.1 * (elevation / 2240.0), 4)
            ctx[p.id] = {"minutes": minutes, "dist_km": dist_km, "fatigue": fatigue}

        # Shot-allocation weight: skill * share-of-match * freshness.
        def _weight(pl):
            c = ctx[pl.id]
            return (max(0.05, pl.skill)
                    * (c["minutes"] / 90.0)
                    * (1.0 - 0.45 * c["fatigue"]))

        shooter_weights = [_weight(p) for p in shooters]
        if sum(shooter_weights) <= 0:
            shooter_weights = [1.0] * len(shooters)

        for _ in range(int(n_shots)):
            shooter = rng.choices(shooters, weights=shooter_weights)[0]
            x, y = _sample_shot_location(rng)
            distance, angle = shot_geometry(x, y)
            is_header = rng.random() < (0.22 if shooter.position != "GK" else 0)
            pressure = rng.random() < 0.45
            big_chance = (distance < 12 and angle > 0.5 and rng.random() < 0.5)

            p_goal = _true_goal_prob(distance, angle, int(is_header),
                                     int(pressure), int(big_chance), shooter.skill)
            is_goal = rng.random() < p_goal

            s.add(Shot(match_id=m.id, player_id=shooter.id, team_id=team.id,
                       minute=rng.randint(1, 95), x=x, y=y, distance=distance,
                       angle=angle, is_header=is_header, under_pressure=pressure,
                       big_chance=big_chance, is_goal=is_goal))
            per_player[shooter.id]["shots"] += 1
            per_player[shooter.id]["xg"] += p_goal
            if is_goal:
                per_player[shooter.id]["goals"] += 1
                score[team.id] += 1

        # Per-player match stats (minutes/fatigue computed above in `ctx`).
        for p in squad:
            c = ctx[p.id]
            minutes, dist_km, fatigue = c["minutes"], c["dist_km"], c["fatigue"]
            st = per_player[p.id]
            s.add(PlayerMatchStat(
                match_id=m.id, player_id=p.id, minutes=minutes,
                shots=st["shots"], goals=st["goals"], xg=round(st["xg"], 3),
                assists=np.random.poisson(0.12 if p.position in ("MID", "FWD") else 0.03),
                passes=int(max(0, rng.gauss(45 if p.position == "MID" else 28, 12)
                               * minutes / 90)),
                distance_km=dist_km, rest_days=rest, travel_km=round(travel, 1),
                elevation_m=elevation, fatigue_index=fatigue))

    m.home_goals = score[home.id]
    m.away_goals = score[away.id]


if __name__ == "__main__":
    print("Generating synthetic tournament data...")
    print(generate())
