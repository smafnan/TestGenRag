"""
Persistence for human-approved, e-signed test cases.

Uses PostgreSQL when DATABASE_URL is set (e.g.
postgresql+psycopg://user:pass@host:5432/db), otherwise falls back to a local
SQLite file so the app runs with zero external services. SQLAlchemy keeps the
model identical across both.
"""

import datetime as _dt
import json
import os
from typing import List

from sqlalchemy import String, Text, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///approved_testcases.db")

_engine = create_engine(DATABASE_URL, echo=False, future=True)


class Base(DeclarativeBase):
    pass


class ApprovedTestCase(Base):
    __tablename__ = "approved_test_cases"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    requirement: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String(512))
    priority: Mapped[str] = mapped_column(String(16))
    test_case_json: Mapped[str] = mapped_column(Text)
    signed_by: Mapped[str] = mapped_column(String(128), default="reviewer")
    signed_at: Mapped[_dt.datetime] = mapped_column(
        DateTime, default=lambda: _dt.datetime.now(_dt.timezone.utc)
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "requirement": self.requirement,
            "title": self.title,
            "priority": self.priority,
            "test_case": json.loads(self.test_case_json),
            "signed_by": self.signed_by,
            "signed_at": self.signed_at.isoformat(),
        }


def init_db() -> None:
    """Create tables if they do not exist. Safe to call on every startup."""
    Base.metadata.create_all(_engine)


def save_approved(requirement: str, test_case: dict, signed_by: str) -> dict:
    with Session(_engine) as session:
        row = ApprovedTestCase(
            requirement=requirement,
            title=test_case.get("title", "Untitled"),
            priority=test_case.get("priority", "Medium"),
            test_case_json=json.dumps(test_case, ensure_ascii=False),
            signed_by=signed_by,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.to_dict()


def list_approved() -> List[dict]:
    with Session(_engine) as session:
        rows = session.query(ApprovedTestCase).order_by(
            ApprovedTestCase.signed_at.desc()
        ).all()
        return [r.to_dict() for r in rows]


def backend_name() -> str:
    return _engine.url.get_backend_name()
