"""Loader for real StatsBomb open data.

This is the production data path. It populates the SAME schema as the synthetic
generator (teams / players / matches / shots / player_match_stats), so every
downstream model and the dashboard work unchanged on real World Cup data.

What it derives from the raw event feed:
  * shots          — location -> distance/angle geometry + goal outcome
  * minutes        — from Starting XI + Substitution events
  * goals / xG      — xG uses StatsBomb's own shot model (shot_statsbomb_xg)
  * passes / assists— pass counts and goal assists
  * rest_days      — from each team's fixture schedule
  * player skill    — career xG-per-shot proxy (used as a forecaster feature)

Not available from this source (set to 0 and documented): travel_km,
elevation_m, distance_km. The synthetic generator remains the way to exercise
those tournament-context features.

Requires: pip install statsbombpy   (free open data, non-commercial licence:
https://github.com/statsbomb/open-data)

Usage:
    from apexsports.data.statsbomb import load_competition
    load_competition(competition_id=43, season_id=106)   # 2022 FIFA World Cup
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from apexsports.data.database import get_session, init_db
from apexsports.data.schema import (
    Team, Player, Match, Shot, PlayerMatchStat,
)
from apexsports.utils import shot_geometry

POS_PRIOR = {"GK": 0.05, "DEF": 0.20, "MID": 0.45, "FWD": 0.72}


def load_competition(competition_id: int, season_id: int,
                     max_matches: int | None = None, verbose: bool = True) -> dict:
    """Load one StatsBomb competition-season into the ApexSports schema."""
    try:
        from statsbombpy import sb
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "statsbombpy not installed. Run `pip install statsbombpy`.") from e

    init_db()
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    matches = matches.sort_values("match_date").reset_index(drop=True)
    if max_matches:
        matches = matches.head(max_matches)

    team_ids: dict[str, int] = {}
    player_ids: dict[str, int] = {}
    player_career: dict[int, dict] = defaultdict(lambda: {"shots": 0, "xg": 0.0})
    last_date: dict[int, datetime] = {}
    n_shots = 0

    with get_session() as s:
        # Wipe any prior run (synthetic or real) for a clean reload.
        for model in (Shot, PlayerMatchStat, Match, Player, Team):
            s.query(model).delete()
        s.flush()

        def _team(name: str) -> int:
            if name not in team_ids:
                t = Team(name=name)
                s.add(t)
                s.flush()
                team_ids[name] = t.id
            return team_ids[name]

        def _player(name: str, team_id: int, pos: str) -> int:
            key = f"{name}@{team_id}"
            if key not in player_ids:
                p = Player(name=name, team_id=team_id, position=pos,
                           skill=POS_PRIOR[pos])
                s.add(p)
                s.flush()
                player_ids[key] = p.id
            return player_ids[key]

        for n, (_, mrow) in enumerate(matches.iterrows(), 1):
            mid = int(mrow["match_id"])
            home_name, away_name = mrow["home_team"], mrow["away_team"]
            home_id, away_id = _team(home_name), _team(away_name)
            mdate = _parse_date(mrow.get("match_date"))

            match = Match(
                date=mdate,
                stage=str(mrow.get("competition_stage", "Unknown"))[:30],
                home_team_id=home_id, away_team_id=away_id,
                city=str(mrow.get("stadium", "Unknown"))[:40],
                home_goals=int(mrow.get("home_score", 0) or 0),
                away_goals=int(mrow.get("away_score", 0) or 0))
            s.add(match)
            s.flush()

            events = sb.events(match_id=mid)
            team_of = {home_name: home_id, away_name: away_id}

            # Per-player modal position + team across this match's events.
            pos_of, tname_of = _player_positions_teams(events)

            minutes = _player_minutes(events)
            rest = {tid: _rest_days(last_date, tid, mdate)
                    for tid in (home_id, away_id)}
            last_date[home_id] = last_date[away_id] = mdate

            # --- shots -----------------------------------------------------
            shots = events[events["type"] == "Shot"]
            agg = defaultdict(lambda: {"shots": 0, "goals": 0, "xg": 0.0})
            for _, ev in shots.iterrows():
                loc = ev.get("location")
                if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                    continue
                x, y = float(loc[0]), float(loc[1])
                distance, angle = shot_geometry(x, y)
                tname = ev["team"]
                team_id = team_of[tname]
                pos = _norm_pos(pos_of.get(ev["player"], "Center Midfield"))
                pid = _player(ev["player"], team_id, pos)
                is_goal = str(ev.get("shot_outcome", "")) == "Goal"
                sb_xg = float(ev.get("shot_statsbomb_xg") or 0.0)
                s.add(Shot(
                    match_id=match.id, player_id=pid, team_id=team_id,
                    minute=int(ev.get("minute", 0) or 0), x=x, y=y,
                    distance=distance, angle=angle,
                    is_header=str(ev.get("shot_body_part", "")) == "Head",
                    under_pressure=bool(ev.get("under_pressure", False)),
                    big_chance=sb_xg >= 0.35, is_goal=is_goal,
                    sb_xg=round(sb_xg, 4)))
                n_shots += 1
                a = agg[pid]
                a["shots"] += 1
                a["xg"] += sb_xg
                a["goals"] += int(is_goal)
                player_career[pid]["shots"] += 1
                player_career[pid]["xg"] += sb_xg

            # --- passes + assists per player ------------------------------
            passes = events[events["type"] == "Pass"]
            pass_ct = passes["player"].value_counts().to_dict()
            if "pass_goal_assist" in passes.columns:
                assisters = passes[passes["pass_goal_assist"] == True]
                assist_ct = assisters["player"].value_counts().to_dict()
            else:
                assist_ct = {}

            # --- write player-match stats for everyone who played ---------
            for pname, mins in minutes.items():
                if mins <= 0:
                    continue
                tname = tname_of.get(pname)
                if tname not in team_of:
                    continue
                team_id = team_of[tname]
                pos = _norm_pos(pos_of.get(pname, "Center Midfield"))
                pid = _player(pname, team_id, pos)
                a = agg[pid]
                rd = rest[team_id]
                s.add(PlayerMatchStat(
                    match_id=match.id, player_id=pid, minutes=int(mins),
                    shots=a["shots"], goals=a["goals"], xg=round(a["xg"], 3),
                    assists=int(assist_ct.get(pname, 0)),
                    passes=int(pass_ct.get(pname, 0)),
                    distance_km=0.0,                 # not in open data
                    rest_days=rd, travel_km=0.0, elevation_m=0.0,
                    fatigue_index=round(0.6 if rd <= 3 else 0.35, 3),
                    notes="statsbomb"))

            if verbose:
                print(f"  [{n}/{len(matches)}] {home_name} {match.home_goals}-"
                      f"{match.away_goals} {away_name}  ({len(shots)} shots)")

        # --- second pass: per-player skill = career xG-per-shot proxy -----
        for pid, c in player_career.items():
            if c["shots"] >= 3:
                xgps = c["xg"] / c["shots"]
                skill = float(np.clip(0.15 + 2.2 * xgps, 0.05, 0.95))
                s.query(Player).filter(Player.id == pid).update({"skill": skill})

        counts = {
            "competition_id": competition_id, "season_id": season_id,
            "teams": len(team_ids), "players": len(player_ids),
            "matches": len(matches), "shots": n_shots,
            "player_match_stats": s.query(PlayerMatchStat).count(),
        }
    return counts


def _player_positions_teams(events):
    """Modal position + team per player name across a match's events."""
    pos, team = defaultdict(Counter), defaultdict(Counter)
    for _, ev in events.iterrows():
        p = ev.get("player")
        if not isinstance(p, str):
            continue
        if isinstance(ev.get("position"), str):
            pos[p][ev["position"]] += 1
        if isinstance(ev.get("team"), str):
            team[p][ev["team"]] += 1
    pos_of = {p: c.most_common(1)[0][0] for p, c in pos.items()}
    team_of = {p: c.most_common(1)[0][0] for p, c in team.items()}
    return pos_of, team_of


def _player_minutes(events) -> dict:
    """Approximate minutes played per player from Starting XI + Substitutions."""
    end_min = int(events["minute"].max() or 90) + 1
    on = {}   # player -> minute came on
    off = {}  # player -> minute went off

    for _, ev in events[events["type"] == "Starting XI"].iterrows():
        tactics = ev.get("tactics")
        if isinstance(tactics, dict):
            for item in tactics.get("lineup", []):
                name = item.get("player", {}).get("name")
                if name:
                    on[name] = 0

    for _, ev in events[events["type"] == "Substitution"].iterrows():
        minute = int(ev.get("minute", 0) or 0)
        if isinstance(ev.get("player"), str):
            off[ev["player"]] = minute
        repl = ev.get("substitution_replacement")
        if isinstance(repl, str):
            on[repl] = minute

    minutes = {}
    for name, start in on.items():
        minutes[name] = max(0, off.get(name, end_min) - start)
    return minutes


def _rest_days(last_date: dict, team_id: int, mdate: datetime) -> int:
    if team_id not in last_date:
        return 5
    return max(2, (mdate - last_date[team_id]).days)


def _norm_pos(sb_position: str) -> str:
    p = (sb_position or "").lower()
    if "goalkeeper" in p:
        return "GK"
    if "back" in p or "defen" in p:
        return "DEF"
    if "forward" in p or "striker" in p or "wing" in p:
        return "FWD"
    return "MID"


def _parse_date(value) -> datetime:
    try:
        return datetime.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return datetime(2022, 11, 20)


if __name__ == "__main__":
    import json
    print(json.dumps(load_competition(competition_id=43, season_id=106), indent=2))
