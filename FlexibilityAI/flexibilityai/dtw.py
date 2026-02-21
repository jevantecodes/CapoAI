from __future__ import annotations

import numpy as np


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, window: int | None = None) -> float:
    """Compute normalized DTW distance between two feature sequences."""
    if seq_a.ndim != 2 or seq_b.ndim != 2:
        raise ValueError("DTW expects 2D arrays shaped [frames, features].")
    if seq_a.shape[1] != seq_b.shape[1]:
        raise ValueError("Both sequences must have the same feature dimension.")
    if len(seq_a) == 0 or len(seq_b) == 0:
        raise ValueError("Sequences must be non-empty for DTW.")

    n, m = len(seq_a), len(seq_b)
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))

    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        a = seq_a[i - 1]
        for j in range(j_start, j_end + 1):
            cost = float(np.linalg.norm(a - seq_b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(dp[n, m] / (n + m))
