from __future__ import annotations

import numpy as np

# MediaPipe pose landmark indices.
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28

FEATURE_NAMES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_knee_angle",
    "right_knee_angle",
    "torso_tilt_deg",
    "hip_drop_norm",
    "ankle_width_norm",
    "knee_width_norm",
    "left_wrist_to_ankle_norm",
    "right_wrist_to_ankle_norm",
]


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-8:
        return 180.0
    cos_theta = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _center(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def frame_to_features(frame_landmarks: np.ndarray) -> np.ndarray:
    """Convert one frame [33, >=3] into engineered biomechanical features."""
    xyz = frame_landmarks[:, :3]

    ls = xyz[L_SHOULDER]
    rs = xyz[R_SHOULDER]
    le = xyz[L_ELBOW]
    re = xyz[R_ELBOW]
    lw = xyz[L_WRIST]
    rw = xyz[R_WRIST]
    lh = xyz[L_HIP]
    rh = xyz[R_HIP]
    lk = xyz[L_KNEE]
    rk = xyz[R_KNEE]
    la = xyz[L_ANKLE]
    ra = xyz[R_ANKLE]

    shoulder_center = _center(ls, rs)
    hip_center = _center(lh, rh)
    trunk_scale = _dist(shoulder_center, hip_center)
    if trunk_scale < 1e-4:
        trunk_scale = 1.0

    torso_vec = hip_center - shoulder_center
    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    torso_norm = np.linalg.norm(torso_vec)
    if torso_norm < 1e-8:
        torso_tilt = 0.0
    else:
        cos_tilt = float(np.clip(abs(np.dot(torso_vec, vertical)) / torso_norm, -1.0, 1.0))
        torso_tilt = float(np.degrees(np.arccos(cos_tilt)))

    feature_vector = np.array(
        [
            _angle(ls, le, lw),
            _angle(rs, re, rw),
            _angle(le, ls, lh),
            _angle(re, rs, rh),
            _angle(ls, lh, lk),
            _angle(rs, rh, rk),
            _angle(lh, lk, la),
            _angle(rh, rk, ra),
            torso_tilt,
            (hip_center[1] - shoulder_center[1]) / trunk_scale,
            _dist(la, ra) / trunk_scale,
            _dist(lk, rk) / trunk_scale,
            _dist(lw, la) / trunk_scale,
            _dist(rw, ra) / trunk_scale,
        ],
        dtype=np.float32,
    )
    return feature_vector


def extract_feature_sequence(landmark_sequence: np.ndarray) -> np.ndarray:
    if landmark_sequence.ndim != 3 or landmark_sequence.shape[1] != 33:
        raise ValueError("Landmark sequence must have shape [frames, 33, values].")
    frames = [frame_to_features(frame) for frame in landmark_sequence]
    return np.stack(frames, axis=0)
