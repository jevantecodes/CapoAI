from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2


@dataclass
class ClipSpec:
    movement: str
    query: str
    clip_start: float
    clip_end: float


def _safe_name(raw: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw.strip())
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "template"


def _cut_clip_cv2(source_path: Path, target_path: Path, start_time: float, end_time: float) -> None:
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open downloaded video: {source_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Downloaded video has invalid dimensions.")

    start_frame = max(0, int(start_time * fps))
    end_frame = max(start_frame + 1, int(end_time * fps))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(target_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while cap.isOpened() and frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if not target_path.exists() or target_path.stat().st_size == 0:
        raise ValueError("Failed to generate clip from downloaded video.")


def _load_query_config(path: str | Path) -> dict[str, list[str]]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"YouTube query config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "movement_queries" not in data:
        raise ValueError("Expected 'movement_queries' object in YouTube query config.")
    movement_queries = data["movement_queries"]
    if not isinstance(movement_queries, dict):
        raise ValueError("'movement_queries' must be a JSON object.")
    return movement_queries


def bootstrap_templates_from_youtube(
    *,
    analyzer: Any,
    query_config_path: str | Path,
    downloads_dir: str | Path,
    clips_dir: str | Path,
    per_movement: int,
    clip_start: float,
    clip_end: float,
    movements: list[str] | None = None,
) -> dict[str, Any]:
    try:
        import yt_dlp
    except Exception as exc:
        raise RuntimeError(
            "yt-dlp is required for YouTube extraction. Install FlexibilityAI/requirements.txt first."
        ) from exc

    selected_movements = set(movements or analyzer.movement_names)
    movement_queries = _load_query_config(query_config_path)

    downloads_root = Path(downloads_dir)
    clips_root = Path(clips_dir)
    downloads_root.mkdir(parents=True, exist_ok=True)
    clips_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "ok": True,
        "per_movement": int(per_movement),
        "clip_window_seconds": [float(clip_start), float(clip_end)],
        "movements": {},
    }

    if per_movement <= 0:
        raise ValueError("per_movement must be >= 1.")
    if clip_end <= clip_start:
        raise ValueError("clip_end must be greater than clip_start.")

    for movement in sorted(selected_movements):
        if movement not in analyzer.movement_names:
            summary["movements"][movement] = {
                "created": 0,
                "errors": [f"Unknown movement '{movement}'"],
            }
            continue

        queries = movement_queries.get(movement) or [f"{movement} flexibility exercise"]
        movement_created = 0
        movement_errors: list[str] = []

        for query in queries:
            if movement_created >= per_movement:
                break

            try:
                info_opts = {
                    "quiet": True,
                    "skip_download": True,
                    "extract_flat": "in_playlist",
                }
                with yt_dlp.YoutubeDL(info_opts) as ydl:
                    search = ydl.extract_info(f"ytsearch{per_movement * 2}:{query}", download=False)
                entries = search.get("entries") or []
            except Exception as exc:
                movement_errors.append(f"Search failed for query '{query}': {exc}")
                continue

            for entry in entries:
                if movement_created >= per_movement:
                    break

                video_url = entry.get("url") or entry.get("webpage_url")
                if not video_url:
                    continue
                if not video_url.startswith("http"):
                    video_url = f"https://www.youtube.com/watch?v={video_url}"

                video_id = entry.get("id") or _safe_name(entry.get("title") or "video")
                title_slug = _safe_name(entry.get("title") or video_id)
                movement_download_dir = downloads_root / movement
                movement_download_dir.mkdir(parents=True, exist_ok=True)

                outtmpl = str(movement_download_dir / f"{title_slug}_{video_id}.%(ext)s")
                download_opts = {
                    "quiet": True,
                    "noplaylist": True,
                    "format": "mp4/bestvideo+bestaudio/best",
                    "outtmpl": outtmpl,
                    "merge_output_format": "mp4",
                    "retries": 2,
                }

                downloaded_path: Path | None = None
                try:
                    with yt_dlp.YoutubeDL(download_opts) as ydl:
                        result = ydl.extract_info(video_url, download=True)
                        maybe_path = ydl.prepare_filename(result)

                    base = Path(maybe_path)
                    if base.exists():
                        downloaded_path = base
                    else:
                        mp4_alt = base.with_suffix(".mp4")
                        if mp4_alt.exists():
                            downloaded_path = mp4_alt
                        else:
                            candidates = sorted(movement_download_dir.glob(f"{title_slug}_{video_id}.*"))
                            downloaded_path = candidates[0] if candidates else None

                    if downloaded_path is None:
                        raise ValueError("Could not locate downloaded video file.")

                    clip_name = f"{movement}_yt_{movement_created + 1}_{video_id}.mp4"
                    clip_path = clips_root / movement / clip_name
                    _cut_clip_cv2(downloaded_path, clip_path, clip_start, clip_end)

                    analyzer.create_template(
                        movement=movement,
                        video_path=clip_path,
                        template_name=f"yt_{title_slug}_{movement_created + 1}",
                    )
                    movement_created += 1
                except Exception as exc:
                    movement_errors.append(f"{video_url}: {exc}")

        summary["movements"][movement] = {
            "created": movement_created,
            "errors": movement_errors,
        }

    return summary
