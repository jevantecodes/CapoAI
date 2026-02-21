from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TemplateRecord:
    movement: str
    name: str
    features: np.ndarray
    metadata: dict[str, Any]
    path: Path


def safe_name(name: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "template"


def save_template(
    target_path: str | Path,
    *,
    movement: str,
    features: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    saved_metadata = dict(metadata or {})
    saved_metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    saved_metadata.setdefault("name", path.stem)
    saved_metadata.setdefault("movement", movement)

    np.savez_compressed(
        path,
        movement=np.array(movement),
        features=features.astype(np.float32),
        metadata=np.array(json.dumps(saved_metadata)),
    )
    return path


def load_templates(templates_dir: str | Path) -> dict[str, list[TemplateRecord]]:
    root = Path(templates_dir)
    if not root.exists():
        return {}

    grouped: dict[str, list[TemplateRecord]] = {}
    for path in sorted(root.rglob("*.npz")):
        try:
            data = np.load(path, allow_pickle=False)
        except Exception:
            continue

        features = data.get("features")
        if features is None or features.ndim != 2:
            continue

        movement_raw = data.get("movement")
        movement = str(movement_raw.item()) if movement_raw is not None else path.parent.name

        metadata_raw = data.get("metadata")
        metadata: dict[str, Any]
        if metadata_raw is None:
            metadata = {}
        else:
            try:
                metadata = json.loads(str(metadata_raw.item()))
            except Exception:
                metadata = {}

        name = str(metadata.get("name") or path.stem)
        record = TemplateRecord(
            movement=movement,
            name=name,
            features=features.astype(np.float32),
            metadata=metadata,
            path=path,
        )
        grouped.setdefault(movement, []).append(record)

    return grouped
