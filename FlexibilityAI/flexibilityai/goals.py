from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _clamp_score(value: float) -> float:
    return max(0.0, min(float(value), 100.0))


class GoalAdvisor:
    """Maps flexibility movement scores to capoeira goal readiness."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

    @property
    def goals(self) -> list[str]:
        return sorted(self.config.get("goals", {}).keys())

    def evaluate(self, goal: str, known_scores: dict[str, float]) -> dict[str, Any]:
        goals = self.config.get("goals", {})
        if goal not in goals:
            raise ValueError(f"Unknown goal '{goal}'. Choose one of: {', '.join(self.goals)}")

        goal_cfg = goals[goal]
        prerequisites = goal_cfg.get("prerequisites", {})
        if not prerequisites:
            raise ValueError(f"Goal '{goal}' has no prerequisites configured.")

        total_weight = 0.0
        weighted_progress_sum = 0.0
        movement_feedback: list[dict[str, Any]] = []

        for movement, cfg in prerequisites.items():
            required = float(cfg.get("required_score", 75.0))
            weight = float(cfg.get("weight", 1.0))
            current = _clamp_score(float(known_scores.get(movement, 0.0)))

            progress_ratio = min(current / max(required, 1e-6), 1.0)
            gap = max(0.0, required - current)

            movement_feedback.append(
                {
                    "movement": movement,
                    "current_score": round(current, 2),
                    "required_score": round(required, 2),
                    "gap_to_target": round(gap, 2),
                    "weight": round(weight, 3),
                }
            )

            weighted_progress_sum += progress_ratio * weight
            total_weight += weight

        readiness_score = 100.0 * (weighted_progress_sum / max(total_weight, 1e-6))
        movement_feedback.sort(key=lambda item: item["gap_to_target"], reverse=True)

        top_gaps = [item for item in movement_feedback if item["gap_to_target"] > 0][:3]
        if top_gaps:
            next_focus = [
                {
                    "movement": item["movement"],
                    "needed_points": item["gap_to_target"],
                }
                for item in top_gaps
            ]
        else:
            next_focus = []

        guidance = goal_cfg.get("guidance", "")
        ready_threshold = float(goal_cfg.get("ready_threshold", 85.0))

        return {
            "goal": goal,
            "readiness_score": round(readiness_score, 2),
            "ready_threshold": round(ready_threshold, 2),
            "is_ready": readiness_score >= ready_threshold,
            "movement_feedback": movement_feedback,
            "next_focus": next_focus,
            "guidance": guidance,
        }

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Goal config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "goals" not in data or not data["goals"]:
            raise ValueError("Goal config must include a non-empty 'goals' object.")
        return data
