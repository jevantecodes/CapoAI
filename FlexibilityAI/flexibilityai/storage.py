from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any


class FlexibilityStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS athletes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE,
                    full_name TEXT NOT NULL,
                    email TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    athlete_id INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    goal TEXT,
                    filename TEXT,
                    predicted_movement TEXT NOT NULL,
                    movement_confidence REAL NOT NULL,
                    quality_score REAL NOT NULL,
                    poorness_score REAL NOT NULL,
                    readiness_score REAL,
                    is_ready INTEGER,
                    breakdown_json TEXT,
                    movement_ranking_json TEXT,
                    known_scores_json TEXT,
                    goal_feedback_json TEXT,
                    raw_result_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(athlete_id) REFERENCES athletes(id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_performance_athlete_created
                ON performance_records(athlete_id, created_at)
                """
            )
            conn.commit()

    def create_athlete(
        self,
        *,
        full_name: str,
        user_id: str | None = None,
        email: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        now = self._utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO athletes (user_id, full_name, email, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, full_name, email, notes, now, now),
            )
            conn.commit()
            athlete_id = int(cursor.lastrowid)
        athlete = self.get_athlete(athlete_id)
        if athlete is None:
            raise RuntimeError("Failed to create athlete record.")
        return athlete

    def list_athletes(self, *, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, user_id, full_name, email, notes, created_at, updated_at
                FROM athletes
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_athlete(self, athlete_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, user_id, full_name, email, notes, created_at, updated_at
                FROM athletes
                WHERE id = ?
                """,
                (int(athlete_id),),
            ).fetchone()
        return dict(row) if row else None

    def update_athlete(
        self,
        athlete_id: int,
        *,
        full_name: str | None = None,
        user_id: str | None = None,
        email: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_athlete(athlete_id)
        if current is None:
            raise ValueError(f"Athlete id {athlete_id} not found.")

        merged = {
            "full_name": full_name if full_name is not None else current["full_name"],
            "user_id": user_id if user_id is not None else current["user_id"],
            "email": email if email is not None else current["email"],
            "notes": notes if notes is not None else current["notes"],
        }

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE athletes
                SET user_id = ?, full_name = ?, email = ?, notes = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    merged["user_id"],
                    merged["full_name"],
                    merged["email"],
                    merged["notes"],
                    self._utc_now(),
                    int(athlete_id),
                ),
            )
            conn.commit()

        updated = self.get_athlete(athlete_id)
        if updated is None:
            raise RuntimeError("Athlete update failed unexpectedly.")
        return updated

    def add_performance_record(
        self,
        *,
        athlete_id: int,
        result: dict[str, Any],
        source_type: str,
        goal: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        if self.get_athlete(athlete_id) is None:
            raise ValueError(f"Athlete id {athlete_id} not found.")

        goal_feedback = result.get("goal_feedback") if isinstance(result.get("goal_feedback"), dict) else None
        readiness_score = None
        is_ready = None
        if goal_feedback is not None:
            readiness_score = goal_feedback.get("readiness_score")
            is_ready = 1 if bool(goal_feedback.get("is_ready")) else 0

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO performance_records (
                    athlete_id,
                    source_type,
                    goal,
                    filename,
                    predicted_movement,
                    movement_confidence,
                    quality_score,
                    poorness_score,
                    readiness_score,
                    is_ready,
                    breakdown_json,
                    movement_ranking_json,
                    known_scores_json,
                    goal_feedback_json,
                    raw_result_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(athlete_id),
                    source_type,
                    goal,
                    filename,
                    result.get("predicted_movement", ""),
                    float(result.get("movement_confidence", 0.0)),
                    float(result.get("quality_score", 0.0)),
                    float(result.get("poorness_score", 100.0)),
                    float(readiness_score) if readiness_score is not None else None,
                    is_ready,
                    json.dumps(result.get("breakdown", {})),
                    json.dumps(result.get("movement_ranking", [])),
                    json.dumps(result.get("known_scores", {})),
                    json.dumps(goal_feedback or {}),
                    json.dumps(result),
                    self._utc_now(),
                ),
            )
            conn.commit()
            record_id = int(cursor.lastrowid)
        record = self.get_performance_record(record_id)
        if record is None:
            raise RuntimeError("Failed to persist performance record.")
        return record

    def get_performance_record(self, record_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM performance_records
                WHERE id = ?
                """,
                (int(record_id),),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_performance_records(self, athlete_id: int, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM performance_records
                WHERE athlete_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(athlete_id), max(1, int(limit))),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_latest_record(self, athlete_id: int) -> dict[str, Any] | None:
        rows = self.list_performance_records(athlete_id, limit=1)
        return rows[0] if rows else None

    def athlete_analytics(self, athlete_id: int) -> dict[str, Any]:
        athlete = self.get_athlete(athlete_id)
        if athlete is None:
            raise ValueError(f"Athlete id {athlete_id} not found.")

        records = self.list_performance_records(athlete_id, limit=500)
        if not records:
            return {
                "athlete": athlete,
                "total_sessions": 0,
                "quality": {},
                "readiness": {},
                "movement_breakdown": {},
                "latest_session": None,
            }

        records_chrono = list(reversed(records))
        quality_values = [float(r["quality_score"]) for r in records_chrono]
        readiness_values = [float(r["readiness_score"]) for r in records_chrono if r["readiness_score"] is not None]

        n = len(quality_values)
        baseline_window = quality_values[: min(5, n)]
        recent_window = quality_values[max(0, n - 5) :]
        quality_improvement = (sum(recent_window) / len(recent_window)) - (sum(baseline_window) / len(baseline_window))

        readiness_improvement = None
        if len(readiness_values) >= 2:
            r_n = len(readiness_values)
            r_base = readiness_values[: min(5, r_n)]
            r_recent = readiness_values[max(0, r_n - 5) :]
            readiness_improvement = (sum(r_recent) / len(r_recent)) - (sum(r_base) / len(r_base))

        per_movement: dict[str, list[float]] = {}
        for row in records_chrono:
            movement = str(row.get("predicted_movement") or "unknown")
            per_movement.setdefault(movement, []).append(float(row["quality_score"]))

        movement_breakdown: dict[str, Any] = {}
        for movement, vals in per_movement.items():
            movement_breakdown[movement] = {
                "sessions": len(vals),
                "avg_quality": round(sum(vals) / len(vals), 2),
                "best_quality": round(max(vals), 2),
                "latest_quality": round(vals[-1], 2),
            }

        return {
            "athlete": athlete,
            "total_sessions": len(records_chrono),
            "quality": {
                "avg": round(sum(quality_values) / len(quality_values), 2),
                "best": round(max(quality_values), 2),
                "latest": round(quality_values[-1], 2),
                "improvement_recent_vs_baseline": round(quality_improvement, 2),
            },
            "readiness": {
                "avg": round(sum(readiness_values) / len(readiness_values), 2) if readiness_values else None,
                "best": round(max(readiness_values), 2) if readiness_values else None,
                "latest": round(readiness_values[-1], 2) if readiness_values else None,
                "improvement_recent_vs_baseline": round(readiness_improvement, 2)
                if readiness_improvement is not None
                else None,
            },
            "movement_breakdown": movement_breakdown,
            "latest_session": records[0],
        }

    @staticmethod
    def _json_to_obj(raw: str | None) -> Any:
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def _row_to_record(self, row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["is_ready"] = None if record.get("is_ready") is None else bool(record["is_ready"])
        record["breakdown"] = self._json_to_obj(record.pop("breakdown_json", None))
        record["movement_ranking"] = self._json_to_obj(record.pop("movement_ranking_json", None))
        record["known_scores"] = self._json_to_obj(record.pop("known_scores_json", None))
        record["goal_feedback"] = self._json_to_obj(record.pop("goal_feedback_json", None))
        record["raw_result"] = self._json_to_obj(record.pop("raw_result_json", None))
        return record
