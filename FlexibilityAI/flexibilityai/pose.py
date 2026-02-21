from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_landmark_sequence(
    video_path: str | Path,
    *,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    max_frames: int | None = None,
) -> np.ndarray:
    """Return pose landmarks as [frames, 33, 4] (x, y, z, visibility)."""
    try:
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe import failed. Install compatible dependencies from "
            "FlexibilityAI/requirements.txt (notably numpy<2)."
        ) from exc

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    mp_pose = mp.solutions.pose
    frames: list[np.ndarray] = []

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if not results.pose_landmarks:
                continue

            landmarks = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
                dtype=np.float32,
            )
            frames.append(landmarks)

            if max_frames is not None and len(frames) >= max_frames:
                break

    cap.release()

    if not frames:
        raise ValueError("No pose landmarks detected in this clip.")

    return np.stack(frames, axis=0)
