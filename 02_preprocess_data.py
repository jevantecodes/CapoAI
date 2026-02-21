from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build X/y tensors from processed landmarks."
    )
    parser.add_argument(
        "--landmark-dir",
        type=Path,
        default=Path("processed_landmarks"),
        help="Root folder that contains per-move .npy landmark files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Where to write X.npy, y.npy, label_map.json and dataset_meta.json.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=50,
        help="Frames per sample (clips are padded/truncated to this length).",
    )
    return parser.parse_args()


def to_categorical_np(labels: list[int], num_classes: int | None = None) -> np.ndarray:
    labels_np = np.array(labels, dtype=int)
    if num_classes is None:
        num_classes = int(labels_np.max()) + 1
    return np.eye(num_classes, dtype=np.float32)[labels_np]


def normalize_sequence(sample: np.ndarray, sequence_length: int) -> np.ndarray:
    if sample.ndim != 2:
        raise ValueError(f"Expected 2D array [frames, features], got shape={sample.shape}")
    if len(sample) < sequence_length:
        pad = np.zeros((sequence_length - len(sample), sample.shape[1]), dtype=np.float32)
        sample = np.vstack([sample, pad])
    else:
        sample = sample[:sequence_length]
    return sample.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    landmark_dir = args.landmark_dir
    output_dir = args.output_dir
    sequence_length = int(args.sequence_length)

    if sequence_length <= 0:
        raise ValueError("--sequence-length must be > 0.")
    if not landmark_dir.exists():
        raise FileNotFoundError(f"Landmark directory not found: {landmark_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    moves = sorted([d.name for d in landmark_dir.iterdir() if d.is_dir()])
    if not moves:
        raise ValueError(f"No move folders found in {landmark_dir}")
    print("Moves (label order):", moves)

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    sample_manifest: list[dict[str, str | int]] = []

    for label_idx, move in enumerate(moves):
        move_folder = landmark_dir / move
        npy_files = sorted(move_folder.glob("*.npy"))
        for npy_file in npy_files:
            sample = np.load(npy_file)
            sample = normalize_sequence(sample, sequence_length=sequence_length)
            x_rows.append(sample)
            y_rows.append(label_idx)
            sample_manifest.append(
                {
                    "move": move,
                    "label_index": int(label_idx),
                    "landmark_file": str(npy_file.relative_to(landmark_dir)),
                }
            )

    if not x_rows:
        raise ValueError(f"No .npy samples found under {landmark_dir}")

    x = np.array(x_rows, dtype=np.float32)
    y = to_categorical_np(y_rows)

    np.save(output_dir / "X.npy", x)
    np.save(output_dir / "y.npy", y)
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(moves, f, indent=2)

    per_move_counts: dict[str, int] = {move: 0 for move in moves}
    for row in sample_manifest:
        per_move_counts[str(row["move"])] += 1

    meta = {
        "landmark_dir": str(landmark_dir),
        "output_dir": str(output_dir),
        "sequence_length": sequence_length,
        "num_samples": int(x.shape[0]),
        "num_classes": int(y.shape[1]),
        "feature_count": int(x.shape[2]),
        "per_move_counts": per_move_counts,
    }
    with (output_dir / "dataset_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with (output_dir / "sample_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(sample_manifest, f, indent=2)

    print(f"Saved dataset to: {output_dir}")
    print("X shape:", x.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    main()
