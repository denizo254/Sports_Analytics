"""Loader for FBref player-match data via the `soccerdata` package.

Use this for competitions/seasons that StatsBomb's free open data doesn't
cover — e.g. recent UEFA Champions League seasons. `soccerdata`'s FBref reader
(v1.9) exposes per-player-per-match *aggregates* (minutes, goals, shots, xG,
assists, passes) but NOT per-shot events, so this populates Match +
PlayerMatchStat only. That is enough for the Poisson and forecaster models;
the per-shot xG model and calibration panel need StatsBomb-style shot data and
are simply skipped on FBref data.

IMPORTANT — run this on your own machine. FBref aggressively blocks datacenter
/ headless access (CAPTCHA / IP block), so the scrape typically fails from
cloud sandboxes but works (slowly, rate-limited) from a normal connection.

Setup (one-time): register the Champions League as a custom soccerdata league
in ~/soccerdata/config/league_dict.json — `ensure_league_config()` does this.

    pip install soccerdata
    python scripts/build_all.py --source fbref --season 2024-2025 2025-2026
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from apexsports.data.database import get_session, init_db
from apexsports.data.schema import Team, Player, Match, PlayerMatchStat, Shot

UCL_LEAGUE = "INT-Champions League"
POS_PRIOR = {"GK": 0.05, "DEF": 0.20, "MID": 0.45, "FWD": 0.72}


def ensure_league_config() -> None:
    """Register the Champions League as a custom FBref league for soccerdata."""
    cfg_dir = os.path.expanduser("~/soccerdata/config")
    os.makedirs(cfg_dir, exist_ok=True)
    path = os.path.join(cfg_dir, "league_dict.json")
    existing = {}
    if os.path.exists(path):
        try:
            existing = json.loads(open(path).read())
        except (ValueError, OSError):
            existing = {}
    if UCL_LEAGUE not in existing:
        existing[UCL_LEAGUE] = {"FBref": "Champions League",
                                "season_start": "Aug", "season_end": "Jun"}
        open(path, "w").write(json.dumps(existing, indent=2))


def load_fbref(seasons, competition: str = UCL_LEAGUE,
               verbose: bool = True) -> dict:
    """Scrape FBref player-match stats for the given seasons into the schema."""
    try:
        import soccerdata as sd
    except ImportError as e:  # pragma: no cover
        raise ImportError("soccerdata not installed. Run `pip install soccerdata`.") from e

    if isinstance(seasons, str):
        seasons = [seasons]
    if competition == UCL_LEAGUE:
        ensure_league_config()

    fb = sd.FBref(leagues=competition, seasons=seasons)
    schedule = _normalize(fb.read_schedule())
    stats = _normalize(fb.read_player_match_stats(stat_type="summary"))
    return ingest(schedule, stats, competition=competition, verbose=verbose)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Reset the index and flatten any MultiIndex columns to flat strings."""
    df = df.reset_index()
    flat = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(c) for c in col if str(c) not in ("", "nan", "None")]
            flat.append("_".join(parts))
        else:
            flat.append(str(col))
    df = df.copy()
    df.columns = flat
    return df


def _col(df: pd.DataFrame, *candidates: str):
    """Return the first column whose name matches a candidate (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # Fall back to a substring match on the last path segment.
    for cand in candidates:
        for c in df.columns:
            if c.lower().endswith(cand.lower()):
                return c
    return None


def _num(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _norm_pos_fbref(pos: str) -> str:
    p = (pos or "").upper()
    if p.startswith("GK"):
        return "GK"
    if "DF" in p:
        return "DEF"
    if "FW" in p:
        return "FWD"
    return "MID"


def _parse_date(value) -> datetime:
    try:
        return datetime.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return datetime(2024, 9, 1)


def ingest(schedule: pd.DataFrame, stats: pd.DataFrame,
           competition: str = UCL_LEAGUE, verbose: bool = True) -> dict:
    """Populate Match + PlayerMatchStat from normalized FBref frames.

    Kept separate from the scrape so the transform is unit-testable offline.
    """
    g_col = _col(schedule, "game")
    date_col = _col(schedule, "date")
    home_col = _col(schedule, "home_team", "home")
    away_col = _col(schedule, "away_team", "away")
    hs_col = _col(schedule, "home_score", "home_g")
    as_col = _col(schedule, "away_score", "away_g")
    score_col = _col(schedule, "score")

    s_game = _col(stats, "game")
    s_team = _col(stats, "team")
    s_player = _col(stats, "player")
    s_pos = _col(stats, "pos", "position")
    c_min = _col(stats, "min", "minutes")
    c_gls = _col(stats, "Performance_Gls", "Gls", "goals")
    c_sh = _col(stats, "Performance_Sh", "Sh", "shots")
    c_xg = _col(stats, "Expected_xG", "xG", "xg")
    c_ast = _col(stats, "Performance_Ast", "Ast", "assists")
    c_pass = _col(stats, "Passes_Att", "Passes_Cmp", "Total_Att", "passes")

    init_db()
    team_ids: dict[str, int] = {}
    player_ids: dict[str, int] = {}
    career: dict[int, dict] = defaultdict(lambda: {"shots": 0, "xg": 0.0})
    last_date: dict[int, datetime] = {}

    def _team(s, name):
        if name not in team_ids:
            t = Team(name=str(name)[:80])
            s.add(t); s.flush()
            team_ids[name] = t.id
        return team_ids[name]

    def _player(s, name, team_id, pos):
        key = f"{name}@{team_id}"
        if key not in player_ids:
            p = Player(name=str(name)[:80], team_id=team_id, position=pos,
                       skill=POS_PRIOR[pos])
            s.add(p); s.flush()
            player_ids[key] = p.id
        return player_ids[key]

    # game -> (date, home, away, hg, ag), in chronological order.
    sched = schedule.sort_values(date_col) if date_col else schedule
    match_meta = {}
    order = []
    for _, r in sched.iterrows():
        game = r[g_col]
        hg, ag = _scoreline(r, hs_col, as_col, score_col)
        match_meta[game] = {
            "date": _parse_date(r[date_col]) if date_col else datetime(2024, 9, 1),
            "home": r[home_col], "away": r[away_col], "hg": hg, "ag": ag}
        order.append(game)

    with get_session() as s:
        for model in (Shot, PlayerMatchStat, Match, Player, Team):
            s.query(model).delete()
        s.flush()

        match_row = {}
        for game in order:
            m = match_meta[game]
            home_id, away_id = _team(s, m["home"]), _team(s, m["away"])
            match = Match(date=m["date"], stage=competition[:30],
                          home_team_id=home_id, away_team_id=away_id,
                          city="Unknown", home_goals=m["hg"], away_goals=m["ag"])
            s.add(match); s.flush()
            rest = {tid: _rest_days(last_date, tid, m["date"])
                    for tid in (home_id, away_id)}
            last_date[home_id] = last_date[away_id] = m["date"]
            match_row[game] = (match.id, {m["home"]: home_id, m["away"]: away_id}, rest)

        n_stats = 0
        for _, r in stats.iterrows():
            game = r[s_game]
            if game not in match_row:
                continue
            match_id, teams_in_game, rest = match_row[game]
            tname = r[s_team]
            team_id = teams_in_game.get(tname) or _team(s, tname)
            pos = _norm_pos_fbref(str(r[s_pos]) if s_pos else "")
            pid = _player(s, r[s_player], team_id, pos)
            minutes = int(_to_num(r[c_min])) if c_min else 0
            if minutes <= 0:
                continue
            xg = float(_to_num(r[c_xg])) if c_xg else 0.0
            shots = int(_to_num(r[c_sh])) if c_sh else 0
            rd = rest.get(team_id, 5)
            s.add(PlayerMatchStat(
                match_id=match_id, player_id=pid, minutes=minutes,
                shots=shots, goals=int(_to_num(r[c_gls])) if c_gls else 0,
                xg=round(xg, 3),
                assists=int(_to_num(r[c_ast])) if c_ast else 0,
                passes=int(_to_num(r[c_pass])) if c_pass else 0,
                distance_km=0.0, rest_days=rd, travel_km=0.0, elevation_m=0.0,
                fatigue_index=round(0.6 if rd <= 3 else 0.35, 3), notes="fbref"))
            career[pid]["shots"] += shots
            career[pid]["xg"] += xg
            n_stats += 1

        for pid, c in career.items():
            if c["shots"] >= 5:
                skill = float(np.clip(0.15 + 2.2 * c["xg"] / c["shots"], 0.05, 0.95))
                s.query(Player).filter(Player.id == pid).update({"skill": skill})

        counts = {
            "source": "fbref", "competition": competition,
            "teams": len(team_ids), "players": len(player_ids),
            "matches": len(order), "shots": 0, "player_match_stats": n_stats,
        }
    if verbose:
        print(json.dumps(counts, indent=2))
    return counts


def _to_num(v) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _scoreline(row, hs_col, as_col, score_col):
    if hs_col and as_col and pd.notna(row.get(hs_col)) and pd.notna(row.get(as_col)):
        return int(_to_num(row[hs_col])), int(_to_num(row[as_col]))
    if score_col and isinstance(row.get(score_col), str):
        # e.g. "2–1" (en dash) or "2-1".
        for sep in ("–", "-", ":"):
            if sep in row[score_col]:
                a, b = row[score_col].split(sep)[:2]
                return int(_to_num(a)), int(_to_num(b))
    return 0, 0


def _rest_days(last_date: dict, team_id: int, mdate: datetime) -> int:
    if team_id not in last_date:
        return 5
    return min(21, max(2, (mdate - last_date[team_id]).days))


if __name__ == "__main__":
    print(load_fbref(["2024-2025", "2025-2026"]))
