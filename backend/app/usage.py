from __future__ import annotations

import sqlite3
import threading
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

COUNTERS: Final = (
    "backend_starts",
    "live_sessions_admitted",
    "live_sessions_finished",
    "live_sessions_rejected",
    "live_session_seconds",
    "live_frames",
    "live_frame_bytes",
    "peak_compute_active",
    "render_requests",
    "render_completed",
    "render_failed",
    "render_cancelled",
    "render_rejected",
    "render_seconds",
    "time_studies_started",
    "time_studies_finished",
    "time_studies_rejected",
    "numerical_failures",
    "internal_errors",
)


class UsageStore:
    """Restart-safe daily aggregates containing no visitor-level data."""

    def __init__(self, data_dir: str | Path, timezone_name: str):
        self.timezone = ZoneInfo(timezone_name)
        path = Path(data_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        self._database = sqlite3.connect(
            path / "usage.sqlite3", check_same_thread=False, timeout=10
        )
        self._database.row_factory = sqlite3.Row
        self._database.execute("PRAGMA journal_mode=WAL")
        self._database.execute("PRAGMA synchronous=FULL")
        self._lock = threading.Lock()
        self._initialize()

    def _initialize(self) -> None:
        columns = ",\n".join(f"{name} REAL NOT NULL DEFAULT 0" for name in COUNTERS)
        with self._lock, self._database:
            self._database.execute(
                f"""
                CREATE TABLE IF NOT EXISTS daily_usage (
                    day TEXT PRIMARY KEY,
                    {columns}
                )
                """  # noqa: S608 - columns are a module-owned allowlist
            )
            self._database.execute(
                """
                CREATE TABLE IF NOT EXISTS report_deliveries (
                    day TEXT PRIMARY KEY,
                    claimed_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    finished_at TEXT,
                    detail TEXT
                )
                """
            )

    def day_for(self, at: datetime | None = None) -> date:
        instant = at or datetime.now(UTC)
        if instant.tzinfo is None:
            raise ValueError("usage timestamps must be timezone-aware")
        return instant.astimezone(self.timezone).date()

    def increment(
        self, name: str, amount: int | float = 1, *, at: datetime | None = None
    ) -> None:
        self.increment_many({name: amount}, at=at)

    def increment_many(
        self,
        values: Mapping[str, int | float],
        *,
        at: datetime | None = None,
    ) -> None:
        if not values:
            return
        unknown = set(values) - set(COUNTERS)
        if unknown:
            raise ValueError(f"unknown usage counters: {sorted(unknown)}")
        day = self.day_for(at).isoformat()
        assignments = ", ".join(f"{name} = {name} + ?" for name in values)
        with self._lock, self._database:
            self._database.execute(
                "INSERT OR IGNORE INTO daily_usage (day) VALUES (?)", (day,)
            )
            self._database.execute(
                f"UPDATE daily_usage SET {assignments} WHERE day = ?",  # noqa: S608
                (*values.values(), day),
            )

    def record_peak(
        self, name: str, value: int | float, *, at: datetime | None = None
    ) -> None:
        if name != "peak_compute_active":
            raise ValueError(f"{name} is not a peak counter")
        day = self.day_for(at).isoformat()
        with self._lock, self._database:
            self._database.execute(
                "INSERT OR IGNORE INTO daily_usage (day) VALUES (?)", (day,)
            )
            self._database.execute(
                f"UPDATE daily_usage SET {name} = MAX({name}, ?) WHERE day = ?",  # noqa: S608
                (value, day),
            )

    def snapshot(self, day: date) -> dict[str, int | float | str]:
        with self._lock:
            row = self._database.execute(
                "SELECT * FROM daily_usage WHERE day = ?", (day.isoformat(),)
            ).fetchone()
        if row is None:
            return {"day": day.isoformat(), **dict.fromkeys(COUNTERS, 0)}
        return dict(row)

    def claim_delivery(self, day: date) -> bool:
        """Claim the sole delivery attempt for a day before contacting SMTP."""
        now = datetime.now(UTC).isoformat()
        with self._lock, self._database:
            cursor = self._database.execute(
                """
                INSERT OR IGNORE INTO report_deliveries
                    (day, claimed_at, status)
                VALUES (?, ?, 'claimed')
                """,
                (day.isoformat(), now),
            )
        return cursor.rowcount == 1

    def finish_delivery(self, day: date, status: str, detail: str = "") -> None:
        if status not in {"sent", "failed"}:
            raise ValueError("delivery status must be sent or failed")
        with self._lock, self._database:
            self._database.execute(
                """
                UPDATE report_deliveries
                SET status = ?, finished_at = ?, detail = ?
                WHERE day = ?
                """,
                (
                    status,
                    datetime.now(UTC).isoformat(),
                    detail[:500],
                    day.isoformat(),
                ),
            )

    def delivery(self, day: date) -> dict[str, str | None] | None:
        with self._lock:
            row = self._database.execute(
                "SELECT * FROM report_deliveries WHERE day = ?", (day.isoformat(),)
            ).fetchone()
        return None if row is None else dict(row)

    def close(self) -> None:
        with self._lock:
            self._database.close()


__all__ = ["COUNTERS", "UsageStore"]
