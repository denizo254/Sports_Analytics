"""Engine / session factory. Import `get_session` everywhere DB access is needed."""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import DATABASE_URL
from apexsports.data.schema import Base

# check_same_thread=False lets FastAPI (threaded) share the SQLite engine.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create all tables if they do not yet exist."""
    Base.metadata.create_all(engine)


@contextmanager
def get_session():
    """Context-managed session with commit/rollback handling."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
