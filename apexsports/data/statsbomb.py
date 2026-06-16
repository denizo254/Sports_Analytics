"""Optional loader for real StatsBomb open data.

This is the production upgrade path from synthetic data. It populates the SAME
schema (teams / players / matches / shots), so every downstream model and the
dashboard work unchanged once real data is loaded.

Requires the free StatsBomb open-data package and network access:

    pip install statsbombpy

Usage:
    from apexsports.data.statsbomb import load_competition
    load_competition(competition_id=43, season_id=106)  # e.g. a World Cup season

StatsBomb open data is free for non-commercial use under their user agreement;
see https://github.com/statsbomb/open-data for the licence and competition IDs.
"""
from __future__ import annotations

from datetime import datetime

from apexsports.data.database import get_session, init_db
from apexsports.data.schema import Team, Player, Match, Shot
from apexsports.utils import shot_geometry


def load_competition(competition_id: int, season_id: int,
                     max_matches: int | None = None) -> dict:
    """Load one StatsBomb competition-season into the ApexSports schema."""
    try:
        from statsbombpy import sb
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "statsbombpy not installed. Run `pip install statsbombpy` to use "
            "the real StatsBomb loader.") from e

    init_db()
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    if max_matches:
        matches = matches.head(max_matches)

    team_ids: dict[str, int] = {}
    player_ids: dict[str, int] = {}
    n_shots = 0

    with get_session() as s:
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
                p = Player(name=name, team_id=team_id,
                           position=_norm_pos(pos), skill=0.5)
                s.add(p)
                s.flush()
                player_ids[key] = p.id
            return player_ids[key]

        for _, mrow in matches.iterrows():
            home_id = _team(mrow["home_team"])
            away_id = _team(mrow["away_team"])
            match = Match(
                date=_parse_date(mrow.get("match_date")),
                stage=str(mrow.get("competition_stage", "Unknown")),
                home_team_id=home_id, away_team_id=away_id,
                city=str(mrow.get("stadium", "Unknown"))[:40],
                home_goals=int(mrow.get("home_score", 0)),
                away_goals=int(mrow.get("away_score", 0)))
            s.add(match)
            s.flush()

            events = sb.events(match_id=int(mrow["match_id"]))
            shots = events[events["type"] == "Shot"]
            for _, ev in shots.iterrows():
                loc = ev.get("location")
                if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                    continue
                x, y = float(loc[0]), float(loc[1])
                distance, angle = shot_geometry(x, y)
                team_id = _team(ev["team"])
                pid = _player(ev["player"], team_id, ev.get("position", "MID"))
                outcome = str(ev.get("shot_outcome", ""))
                s.add(Shot(
                    match_id=match.id, player_id=pid, team_id=team_id,
                    minute=int(ev.get("minute", 0)), x=x, y=y,
                    distance=distance, angle=angle,
                    is_header=str(ev.get("shot_body_part", "")) == "Head",
                    under_pressure=bool(ev.get("under_pressure", False)),
                    big_chance=False,
                    is_goal=(outcome == "Goal")))
                n_shots += 1

    return {"teams": len(team_ids), "players": len(player_ids),
            "matches": len(matches), "shots": n_shots}


def _norm_pos(sb_position: str) -> str:
    p = (sb_position or "").lower()
    if "goalkeeper" in p:
        return "GK"
    if "back" in p or "defend" in p:
        return "DEF"
    if "forward" in p or "striker" in p or "wing" in p:
        return "FWD"
    return "MID"


def _parse_date(value) -> datetime:
    try:
        return datetime.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return datetime(2026, 6, 11)
