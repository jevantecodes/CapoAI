from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

DEFAULT_SEQUENCE_LENGTH = 50
DEFAULT_WINDOW_STRIDE = 10
DEFAULT_LABEL_MAP_PATH = "dataset/label_map.json"
MODEL_SCAN_GLOBS = ("models/*.keras", "models/*.h5")


def _load_labels(base_dir: Path, label_map_path: str = DEFAULT_LABEL_MAP_PATH) -> list[str]:
    path = base_dir / label_map_path
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    return [str(label) for label in labels]


def _load_model_labels(base_dir: Path, model_id: str, fallback_labels: list[str]) -> list[str]:
    model_label_path = base_dir / "models" / f"{model_id}_labels.json"
    if not model_label_path.exists():
        return fallback_labels
    try:
        with model_label_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        return [str(label) for label in labels]
    except Exception:
        return fallback_labels


def _model_label_path(base_dir: Path, model_id: str) -> Path:
    return base_dir / "models" / f"{model_id}_labels.json"


def _infer_architecture(stem: str) -> str:
    lower = stem.lower()
    if "conv1d" in lower or "cnn" in lower:
        return "conv1d"
    if "lstm" in lower:
        return "lstm"
    return "generic"


class CapoeiraInferenceService:
    def __init__(self, base_dir: Path, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
        self.base_dir = base_dir
        self.sequence_length = sequence_length
        self.window_stride = DEFAULT_WINDOW_STRIDE
        self.labels = _load_labels(base_dir)

        self.models: dict[str, Any] = {}
        self.model_paths: dict[str, Path] = {}
        self.model_arch: dict[str, str] = {}
        self.model_labels: dict[str, list[str]] = {}
        self.default_model_id: str | None = None
        self.model = None
        self.model_path: Path | None = None
        self.refresh_models()

        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def refresh_models(self) -> None:
        discovered_paths: list[Path] = []
        for pattern in MODEL_SCAN_GLOBS:
            discovered_paths.extend(sorted(self.base_dir.glob(pattern)))

        seen: set[Path] = set()
        unique_paths = []
        for path in discovered_paths:
            if path in seen:
                continue
            seen.add(path)
            unique_paths.append(path)

        models: dict[str, Any] = {}
        model_paths: dict[str, Path] = {}
        model_arch: dict[str, str] = {}
        model_labels: dict[str, list[str]] = {}

        for path in unique_paths:
            model_id = path.stem
            if model_id in models:
                continue
            try:
                loaded = load_model(path)
                output_dim = int(loaded.output_shape[-1])

                label_path = _model_label_path(self.base_dir, model_id)
                resolved_labels = _load_model_labels(self.base_dir, model_id, self.labels)

                # Keep only models with a trustworthy label mapping:
                # - model-specific labels file with matching output dim, OR
                # - fallback dataset labels with matching output dim.
                if len(resolved_labels) != output_dim:
                    if label_path.exists():
                        # Explicit label file exists but mismatched => skip model.
                        continue
                    if len(self.labels) == output_dim:
                        resolved_labels = self.labels
                    else:
                        # Unknown mapping (e.g. old legacy model with different output dim).
                        continue

                models[model_id] = loaded
                model_paths[model_id] = path
                model_arch[model_id] = _infer_architecture(path.stem)
                model_labels[model_id] = resolved_labels
            except Exception:
                continue

        if not models:
            raise FileNotFoundError(
                "No trained model found under models/. "
                "Train at least one model (e.g. conv1d/lstm)."
            )

        self.models = models
        self.model_paths = model_paths
        self.model_arch = model_arch
        self.model_labels = model_labels
        self.default_model_id = self._select_default_model_id()
        self.model = self.models[self.default_model_id]
        self.model_path = self.model_paths[self.default_model_id]

    def _select_default_model_id(self) -> str:
        priority = [
            "capoeira_conv1d_best",
            "capoeira_lstm_best",
            "capoeira_model_best",
            "capoeira_conv1d_final",
            "capoeira_lstm_final",
            "capoeira_model_final",
        ]
        for preferred in priority:
            if preferred in self.models:
                return preferred
        return sorted(self.models.keys())[0]

    def get_available_models(self) -> list[dict[str, Any]]:
        rows = []
        for model_id in sorted(self.models.keys()):
            path = self.model_paths[model_id]
            rows.append(
                {
                    "id": model_id,
                    "architecture": self.model_arch.get(model_id, "generic"),
                    "path": str(path.relative_to(self.base_dir)),
                    "label_count": len(self.model_labels.get(model_id, [])),
                    "default": model_id == self.default_model_id,
                }
            )
        return rows

    def _resolve_model(self, model_id: str | None) -> tuple[str, Any]:
        resolved = model_id or self.default_model_id
        if resolved not in self.models:
            raise ValueError(f"Unknown model_id '{resolved}'.")
        return resolved, self.models[resolved]

    def extract_landmarks(
        self,
        video_path: Path,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path.name}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0

        start_frame = 0
        end_frame = None
        if start_time is not None:
            start_frame = max(0, math.floor(start_time * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if end_time is not None:
            end_frame = max(0, math.ceil(end_time * fps))

        landmarks_all = []
        landmark_frame_indices = []
        frame_idx = start_frame

        while cap.isOpened():
            if end_frame is not None and frame_idx > end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                frame_landmarks = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32).flatten()
                landmarks_all.append(frame_landmarks)
                landmark_frame_indices.append(frame_idx)

            frame_idx += 1

        cap.release()
        return (
            np.array(landmarks_all, dtype=np.float32),
            np.array(landmark_frame_indices, dtype=np.int32),
            fps,
        )

    def _normalize_sequence(self, landmarks: np.ndarray) -> np.ndarray:
        if landmarks.size == 0:
            raise ValueError("No landmarks detected in this video.")

        feature_count = landmarks.shape[1]
        if len(landmarks) < self.sequence_length:
            pad = np.zeros((self.sequence_length - len(landmarks), feature_count), dtype=np.float32)
            landmarks = np.vstack([landmarks, pad])
        else:
            landmarks = landmarks[: self.sequence_length]
        return landmarks

    def _predict_with_sliding_windows(
        self,
        model: Any,
        landmarks: np.ndarray,
    ) -> tuple[np.ndarray, int, int, int]:
        frame_count = len(landmarks)
        if frame_count < self.sequence_length:
            padded = self._normalize_sequence(landmarks)
            preds = model.predict(np.expand_dims(padded, axis=0), verbose=0)[0]
            return preds, 0, max(0, frame_count - 1), 1

        start_positions = list(range(0, frame_count - self.sequence_length + 1, self.window_stride))
        last_start = frame_count - self.sequence_length
        if start_positions[-1] != last_start:
            start_positions.append(last_start)

        best_preds = None
        best_conf = -1.0
        best_start = 0

        for start in start_positions:
            end = start + self.sequence_length
            window = landmarks[start:end]
            preds = model.predict(np.expand_dims(window, axis=0), verbose=0)[0]
            conf = float(np.max(preds))
            if conf > best_conf:
                best_conf = conf
                best_preds = preds
                best_start = start

        best_end = best_start + self.sequence_length - 1
        return best_preds, best_start, best_end, len(start_positions)

    def predict_video(
        self,
        video_path: Path,
        start_time: float | None = None,
        end_time: float | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        selected_model_id, model = self._resolve_model(model_id)

        raw_landmarks, landmark_frame_indices, fps = self.extract_landmarks(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
        )
        if raw_landmarks.size == 0:
            raise ValueError("No landmarks detected in this video.")

        preds, best_start_idx, best_end_idx, windows_evaluated = self._predict_with_sliding_windows(
            model=model,
            landmarks=raw_landmarks,
        )
        predicted_index = int(np.argmax(preds))
        confidence = float(preds[predicted_index])

        selected_labels = self.model_labels.get(selected_model_id, self.labels)
        if len(selected_labels) == len(preds):
            labels = selected_labels
        else:
            labels = [f"class_{i}" for i in range(len(preds))]

        probabilities = [
            {"move": labels[idx], "confidence": float(value)}
            for idx, value in enumerate(preds)
        ]
        probabilities.sort(key=lambda item: item["confidence"], reverse=True)

        best_window_start_frame = int(landmark_frame_indices[best_start_idx])
        best_window_end_frame = int(landmark_frame_indices[best_end_idx])

        selected_start = 0.0 if start_time is None else float(start_time)
        selected_end = (
            float(landmark_frame_indices[-1] / fps)
            if end_time is None
            else float(end_time)
        )

        selected_model_path = self.model_paths[selected_model_id]

        return {
            "move": labels[predicted_index],
            "confidence": confidence,
            "probabilities": probabilities,
            "frames_with_landmarks": int(len(raw_landmarks)),
            "sequence_length": self.sequence_length,
            "windows_evaluated": windows_evaluated,
            "model_id": selected_model_id,
            "model_architecture": self.model_arch.get(selected_model_id, "generic"),
            "model_path": str(selected_model_path.relative_to(self.base_dir)),
            "labels": labels,
            "selected_segment": {
                "start_sec": selected_start,
                "end_sec": selected_end,
            },
            "best_window": {
                "start_frame": best_window_start_frame,
                "end_frame": best_window_end_frame,
                "start_sec": float(best_window_start_frame / fps),
                "end_sec": float(best_window_end_frame / fps),
            },
        }
