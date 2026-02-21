from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dtw import dtw_distance
from .features import FEATURE_NAMES, extract_feature_sequence
from .goals import GoalAdvisor
from .pose import extract_landmark_sequence
from .templates import TemplateRecord, load_templates, safe_name, save_template


class MovementAnalyzer:
    def __init__(
        self,
        templates_dir: str | Path,
        movement_config_path: str | Path,
        goals_config_path: str | Path | None = None,
    ):
        self.templates_dir = Path(templates_dir)
        self.movement_config_path = Path(movement_config_path)
        self.feature_index = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

        self.config = self._load_config(self.movement_config_path)
        self.templates = load_templates(self.templates_dir)
        self.goal_advisor = GoalAdvisor(goals_config_path) if goals_config_path else None

    @property
    def movement_names(self) -> list[str]:
        return sorted(self.config.get("movements", {}).keys())

    @property
    def goal_names(self) -> list[str]:
        if self.goal_advisor is None:
            return []
        return self.goal_advisor.goals

    def reload_templates(self) -> None:
        self.templates = load_templates(self.templates_dir)

    def template_counts(self) -> dict[str, int]:
        return {movement: len(self.templates.get(movement, [])) for movement in self.movement_names}

    def create_template(
        self,
        *,
        movement: str,
        video_path: str | Path,
        template_name: str | None = None,
    ) -> dict[str, Any]:
        if movement not in self.config.get("movements", {}):
            raise ValueError(f"Unknown movement '{movement}'. Choose one of: {', '.join(self.movement_names)}")

        landmarks = extract_landmark_sequence(video_path)
        features = extract_feature_sequence(landmarks)

        clean_name = safe_name(template_name or f"{movement}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
        template_path = self.templates_dir / movement / f"{clean_name}.npz"
        metadata = {
            "name": clean_name,
            "movement": movement,
            "source_video": str(video_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frame_count": int(features.shape[0]),
        }

        save_template(
            template_path,
            movement=movement,
            features=features,
            metadata=metadata,
        )
        self.reload_templates()

        return {
            "ok": True,
            "movement": movement,
            "template_name": clean_name,
            "template_path": str(template_path),
            "frame_count": int(features.shape[0]),
        }

    def analyze_video(self, video_path: str | Path) -> dict[str, Any]:
        if not any(self.templates.get(m) for m in self.movement_names):
            raise ValueError(
                "No templates found. Register at least one good example with build-template or /api/template."
            )

        landmarks = extract_landmark_sequence(video_path)
        query_features = extract_feature_sequence(landmarks)

        ranking = self._rank_movements(query_features)
        top = ranking[0]

        predicted_movement = top["movement"]
        best_template = top["template"]
        quality = self._quality_breakdown(
            query_features=query_features,
            ref_features=best_template.features,
            movement=predicted_movement,
        )

        confidence = self._classification_confidence(ranking)

        return {
            "predicted_movement": predicted_movement,
            "movement_confidence": round(confidence, 2),
            "quality_score": round(quality["quality_score"], 2),
            "poorness_score": round(quality["poorness_score"], 2),
            "breakdown": {
                "similarity_score": round(quality["similarity_score"], 2),
                "rom_score": round(quality["rom_score"], 2),
                "posture_score": round(quality["posture_score"], 2),
                "best_template": best_template.name,
            },
            "movement_ranking": [
                {
                    "movement": item["movement"],
                    "distance": round(float(item["distance"]), 4),
                    "best_template": item["template"].name,
                }
                for item in ranking
            ],
            "frames_used": int(query_features.shape[0]),
        }

    def analyze_video_for_goal(
        self,
        *,
        video_path: str | Path,
        goal: str,
        known_scores: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if self.goal_advisor is None:
            raise ValueError("Goal advisor is not configured.")

        result = self.analyze_video(video_path)
        merged_scores = {k: float(v) for k, v in (known_scores or {}).items()}
        merged_scores[result["predicted_movement"]] = float(result["quality_score"])

        goal_feedback = self.goal_advisor.evaluate(goal=goal, known_scores=merged_scores)
        result["goal_feedback"] = goal_feedback
        result["known_scores"] = {k: round(float(v), 2) for k, v in merged_scores.items()}
        return result

    def _rank_movements(self, query_features: np.ndarray) -> list[dict[str, Any]]:
        ranking: list[dict[str, Any]] = []
        for movement in self.movement_names:
            movement_templates = self.templates.get(movement, [])
            if not movement_templates:
                continue

            best_template: TemplateRecord | None = None
            best_distance = float("inf")
            window = int(max(len(query_features), 30) * 0.25)

            for template in movement_templates:
                dist = dtw_distance(query_features, template.features, window=window)
                if dist < best_distance:
                    best_distance = dist
                    best_template = template

            if best_template is not None:
                ranking.append(
                    {
                        "movement": movement,
                        "distance": float(best_distance),
                        "template": best_template,
                    }
                )

        if not ranking:
            raise ValueError("Templates are present but none are valid for scoring.")

        ranking.sort(key=lambda item: item["distance"])
        return ranking

    def _quality_breakdown(
        self,
        *,
        query_features: np.ndarray,
        ref_features: np.ndarray,
        movement: str,
    ) -> dict[str, float]:
        movement_cfg = self.config["movements"][movement]
        focus_features = movement_cfg.get("focus_features", FEATURE_NAMES)
        focus_indices = [self.feature_index[name] for name in focus_features if name in self.feature_index]
        if not focus_indices:
            focus_indices = list(range(query_features.shape[1]))

        window = int(max(len(query_features), 30) * 0.25)
        dist = dtw_distance(query_features, ref_features, window=window)
        similarity_scale = float(movement_cfg.get("similarity_scale", 18.0))
        similarity_score = 100.0 * float(np.exp(-dist / max(similarity_scale, 1e-6)))

        q_range = np.ptp(query_features[:, focus_indices], axis=0)
        r_range = np.ptp(ref_features[:, focus_indices], axis=0)
        rom_ratio = np.minimum(q_range / (r_range + 1e-6), 1.0)
        rom_score = 100.0 * float(np.mean(np.clip(rom_ratio, 0.0, 1.0)))

        q_resampled = self._resample(query_features[:, focus_indices], target_len=120)
        r_resampled = self._resample(ref_features[:, focus_indices], target_len=120)
        mean_abs_error = float(np.mean(np.abs(q_resampled - r_resampled)))
        posture_tolerance = float(movement_cfg.get("posture_tolerance", 20.0))
        posture_score = 100.0 * max(0.0, 1.0 - (mean_abs_error / max(posture_tolerance, 1e-6)))

        weights = movement_cfg.get(
            "score_weights",
            {
                "similarity": 0.65,
                "rom": 0.2,
                "posture": 0.15,
            },
        )

        quality_score = (
            similarity_score * float(weights.get("similarity", 0.65))
            + rom_score * float(weights.get("rom", 0.2))
            + posture_score * float(weights.get("posture", 0.15))
        )
        quality_score = float(np.clip(quality_score, 0.0, 100.0))

        return {
            "quality_score": quality_score,
            "poorness_score": 100.0 - quality_score,
            "similarity_score": similarity_score,
            "rom_score": rom_score,
            "posture_score": posture_score,
        }

    @staticmethod
    def _resample(sequence: np.ndarray, target_len: int) -> np.ndarray:
        if len(sequence) == target_len:
            return sequence
        if len(sequence) == 1:
            return np.repeat(sequence, repeats=target_len, axis=0)

        x_old = np.linspace(0.0, 1.0, num=len(sequence), dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)

        columns: list[np.ndarray] = []
        for idx in range(sequence.shape[1]):
            columns.append(np.interp(x_new, x_old, sequence[:, idx]))
        return np.stack(columns, axis=1).astype(np.float32)

    @staticmethod
    def _classification_confidence(ranking: list[dict[str, Any]]) -> float:
        distances = np.array([entry["distance"] for entry in ranking], dtype=np.float64)
        temp = max(np.median(distances), 1e-6)
        logits = -distances / temp
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)
        return float(probs[0] * 100.0)

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Movement config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "movements" not in data or not data["movements"]:
            raise ValueError("Movement config must include a non-empty 'movements' object.")
        return data
