"""SQLAlchemy ORM schema for ApexSports Analytics.

Modelled on the StatsBomb event-data grain so the synthetic generator and a
real StatsBomb loader populate identical tables. Spatial coordinates use the
StatsBomb 120x80 pitch convention.

To target PostgreSQL + TimescaleDB instead of SQLite, point APEX_DATABASE_URL
at Postgres and (optionally) run `SELECT create_hypertable('shots','minute')`
on the time-series tables — the schema itself is unchanged.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, String, Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    # Rough attack/defence strengths used to seed the synthetic generator.
    attack_strength: Mapped[float] = mapped_column(Float, default=1.0)
    defence_strength: Mapped[float] = mapped_column(Float, default=1.0)
    players: Mapped[list["Player"]] = relationship(back_populates="team")


class Player(Base):
    __tablename__ = "players"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    position: Mapped[str] = mapped_column(String(20))  # GK/DEF/MID/FWD
    skill: Mapped[float] = mapped_column(Float, default=0.5)  # 0..1 finishing
    team: Mapped["Team"] = relationship(back_populates="players")


class Match(Base):
    __tablename__ = "matches"
    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[datetime] = mapped_column(DateTime)
    stage: Mapped[str] = mapped_column(String(30))  # Group/RO16/QF/SF/Final
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    city: Mapped[str] = mapped_column(String(40))
    home_goals: Mapped[int] = mapped_column(Integer, default=0)
    away_goals: Mapped[int] = mapped_column(Integer, default=0)


class Shot(Base):
    """One row per shot — the training grain for the xG model."""
    __tablename__ = "shots"
    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    minute: Mapped[int] = mapped_column(Integer)
    x: Mapped[float] = mapped_column(Float)   # StatsBomb location
    y: Mapped[float] = mapped_column(Float)
    distance: Mapped[float] = mapped_column(Float)  # to goal centre
    angle: Mapped[float] = mapped_column(Float)     # radians, goal-mouth angle
    is_header: Mapped[bool] = mapped_column(Boolean, default=False)
    under_pressure: Mapped[bool] = mapped_column(Boolean, default=False)
    big_chance: Mapped[bool] = mapped_column(Boolean, default=False)
    is_goal: Mapped[bool] = mapped_column(Boolean, default=False)
    # Reference xG from the data source (StatsBomb's own model); 0.0 for
    # synthetic shots. Enables calibrating our xG model against StatsBomb's.
    sb_xg: Mapped[float] = mapped_column(Float, default=0.0)


class PlayerMatchStat(Base):
    """Per-player per-match aggregates — training grain for forecasting."""
    __tablename__ = "player_match_stats"
    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    minutes: Mapped[int] = mapped_column(Integer)
    shots: Mapped[int] = mapped_column(Integer, default=0)
    goals: Mapped[int] = mapped_column(Integer, default=0)
    xg: Mapped[float] = mapped_column(Float, default=0.0)
    assists: Mapped[int] = mapped_column(Integer, default=0)
    passes: Mapped[int] = mapped_column(Integer, default=0)
    distance_km: Mapped[float] = mapped_column(Float, default=0.0)  # running load
    # Tournament-context features.
    rest_days: Mapped[int] = mapped_column(Integer, default=4)
    travel_km: Mapped[float] = mapped_column(Float, default=0.0)
    elevation_m: Mapped[float] = mapped_column(Float, default=0.0)
    fatigue_index: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[str] = mapped_column(Text, default="")
