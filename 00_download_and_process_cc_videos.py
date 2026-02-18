from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from yt_dlp import YoutubeDL

# Keep this list focused on capoeira movements you are training.
MOVE_CONFIG: dict[str, dict[str, Any]] = {
    "ginga": {
        "seed_urls": [],
        "queries": [
            "capoeira ginga tutorial",
            "capoeira ginga treino",
            "capoeira ginga drill",
        ],
    },
    "au": {
        "seed_urls": [
            "https://www.youtube.com/watch?v=TQH2-x2kv2o",
            "https://www.youtube.com/watch?v=0KVbdm9q_r4",
            "https://www.youtube.com/watch?v=6Y6iykIwRlE",
        ],
        "queries": [
            "capoeira au tutorial",
            "capoeira au training",
            "capoeira cartwheel au",
        ],
    },
    "mini_au": {
        "seed_urls": [],
        "queries": [
            "capoeira mini au tutorial",
            "capoeira mini au treino",
            "capoeira small au",
        ],
    },
    "one_hand_au": {
        "seed_urls": [
            "https://youtu.be/AECgkHdxdS4?si=xa8Ws-riioDb1gqr",
        ],
        "queries": [
            "capoeira one hand au",
            "capoeira one arm cartwheel",
            "capoeira au de uma mao",
        ],
    },
    "au_sem_maos": {
        "seed_urls": [
            "https://youtu.be/SoN6M4ga6NY?si=wRAcLLfUMUQdQYUl",
        ],
        "queries": [
            "capoeira au sem maos",
            "capoeira no hands cartwheel",
            "capoeira aerial au",
        ],
    },
    "meia_lua_de_compasso": {
        "seed_urls": [
            "https://youtu.be/G3LKd21IJu0?si=9b031dcnJDgIDMPM",
            "https://youtu.be/jBTlUxY24YY?si=8QiXhu_5pXZrhvtz",
        ],
        "queries": [
            "capoeira meia lua de compasso tutorial",
            "capoeira meia lua de compasso treino",
            "capoeira spinning kick compasso",
        ],
    },
    "queixada": {
        "seed_urls": [],
        "queries": [
            "capoeira queixada tutorial",
            "capoeira queixada treino",
            "capoeira queixada kick drill",
        ],
    },
    "meia_lua_de_frente": {
        "seed_urls": [],
        "queries": [
            "capoeira meia lua de frente tutorial",
            "capoeira meia lua de frente treino",
            "capoeira meia lua de frente drill",
        ],
    },
    "armada": {
        "seed_urls": [],
        "queries": [
            "capoeira armada tutorial",
            "capoeira armada treino",
            "capoeira armada kick drill",
        ],
    },
    "cocorinha": {
        "seed_urls": [],
        "queries": [
            "capoeira cocorinha tutorial",
            "capoeira cocorinha treino",
            "capoeira cocorinha drill",
        ],
    },
    "esquiva_lateral": {
        "seed_urls": [],
        "queries": [
            "capoeira esquiva lateral tutorial",
            "capoeira esquiva lateral treino",
            "capoeira esquiva lateral drill",
        ],
    },
    "esquiva_atras": {
        "seed_urls": [],
        "queries": [
            "capoeira esquiva atras tutorial",
            "capoeira esquiva atras treino",
            "capoeira esquiva atras drill",
        ],
    },
    "esquiva_baixa": {
        "seed_urls": [],
        "queries": [
            "capoeira esquiva baixa tutorial",
            "capoeira esquiva baixa treino",
            "capoeira esquiva baixa drill",
        ],
    },
    "negativa": {
        "seed_urls": [],
        "queries": [
            "capoeira negativa tutorial",
            "capoeira negativa treino",
            "capoeira negativa drill",
        ],
    },
}

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
VIDEO_ID_PATTERNS = [
    re.compile(r"[?&]v=([A-Za-z0-9_-]{6,})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{6,})"),
    re.compile(r"/shorts/([A-Za-z0-9_-]{6,})"),
]


@dataclass
class RunSettings:
    data_root: Path
    processed_root: Path
    report_path: Path
    download_archive: Path
    per_query: int
    max_per_move: int
    min_duration: int
    max_duration: int
    include_shorts: bool
    cc_only: bool
    download_only: bool
    extract_only: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bulk download capoeira videos from YouTube and extract MediaPipe pose landmarks. "
            "Use this for dataset expansion by movement keyword."
        )
    )
    parser.add_argument(
        "--moves",
        type=str,
        default="all",
        help="Comma-separated moves to collect (default: all configured moves).",
    )
    parser.add_argument(
        "--per-query",
        type=int,
        default=20,
        help="How many search results to request per query.",
    )
    parser.add_argument(
        "--max-per-move",
        type=int,
        default=120,
        help="Maximum candidate videos per move after dedupe.",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=2,
        help="Minimum video duration in seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=90,
        help="Maximum video duration in seconds.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory for downloaded videos.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("processed_landmarks"),
        help="Directory for output .npy files.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/collection_report.json"),
        help="Path to save a run report JSON.",
    )
    parser.add_argument(
        "--download-archive",
        type=Path,
        default=Path("data/.youtube_download_archive.txt"),
        help="Archive file used by yt-dlp to skip already downloaded videos.",
    )
    parser.add_argument(
        "--include-shorts",
        action="store_true",
        default=False,
        help="Include YouTube Shorts URLs from search results.",
    )
    parser.add_argument(
        "--cc-only",
        action="store_true",
        default=False,
        help="Try to keep only videos tagged Creative Commons in metadata.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        default=False,
        help="Download videos without extracting landmarks.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        default=False,
        help="Skip downloading and extract from videos that already exist in data/<move>/.",
    )
    return parser.parse_args()


def extract_video_id(url: str) -> str:
    for pattern in VIDEO_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    cleaned = url.strip().rstrip("/")
    return cleaned.split("/")[-1]


def dedupe_urls(urls: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for url in urls:
        if not url or not url.strip():
            continue
        video_id = extract_video_id(url)
        if video_id in seen:
            continue
        seen.add(video_id)
        deduped.append(url.strip())
    return deduped


def search_move_urls(move: str, queries: list[str], per_query: int, include_shorts: bool) -> list[str]:
    found_urls: list[str] = []
    opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "ignoreerrors": True,
        "noplaylist": True,
    }

    with YoutubeDL(opts) as ydl:
        for query in queries:
            search_expr = f"ytsearch{per_query}:{query}"
            print(f"[search] {move}: {query!r}")
            try:
                info = ydl.extract_info(search_expr, download=False)
            except Exception as exc:
                print(f"[warn] search failed for query={query!r}: {exc}")
                continue

            entries = info.get("entries") if isinstance(info, dict) else None
            if not entries:
                continue

            for entry in entries:
                if not entry:
                    continue
                video_id = entry.get("id")
                if not video_id:
                    continue
                candidate = f"https://www.youtube.com/watch?v={video_id}"
                if not include_shorts:
                    maybe_url = str(entry.get("url") or "")
                    if "/shorts/" in maybe_url:
                        continue
                found_urls.append(candidate)

    return dedupe_urls(found_urls)


def find_local_video_by_id(move_data_dir: Path, video_id: str) -> Path | None:
    for ext in VIDEO_EXTENSIONS:
        candidate = move_data_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def download_video(
    url: str,
    out_dir: Path,
    archive_file: Path,
    min_duration: int,
    max_duration: int,
    cc_only: bool,
) -> tuple[Path | None, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_file.parent.mkdir(parents=True, exist_ok=True)

    def _match_filter(info_dict: dict[str, Any], *, incomplete: bool) -> str | None:
        if incomplete:
            return None
        duration = info_dict.get("duration")
        if duration is not None and (duration < min_duration or duration > max_duration):
            return f"duration {duration}s outside [{min_duration}, {max_duration}]"
        if cc_only:
            license_name = str(info_dict.get("license") or "").lower()
            if "creative commons" not in license_name:
                return "video not tagged as Creative Commons"
        return None

    ydl_opts = {
        "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "download_archive": str(archive_file),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "match_filter": _match_filter,
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except Exception as exc:
            return None, f"download_error: {exc}"

        if info is None:
            return None, "skipped_by_filter_or_unavailable"

        video_id = str(info.get("id") or extract_video_id(url))
        local_path = find_local_video_by_id(out_dir, video_id)
        if local_path is None:
            guessed = Path(ydl.prepare_filename(info))
            if guessed.exists():
                local_path = guessed

        if local_path is None:
            return None, "downloaded_but_file_not_found"

    return local_path, "ok"


def extract_landmarks(video_path: Path, pose: Any) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    landmarks_all = []

    if not cap.isOpened():
        return np.array([])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            frame_landmarks = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32).flatten()
            landmarks_all.append(frame_landmarks)

    cap.release()
    return np.array(landmarks_all, dtype=np.float32)


def resolve_moves(arg_value: str) -> list[str]:
    if arg_value.strip().lower() == "all":
        return sorted(MOVE_CONFIG.keys())
    requested = [m.strip() for m in arg_value.split(",") if m.strip()]
    invalid = [m for m in requested if m not in MOVE_CONFIG]
    if invalid:
        valid = ", ".join(sorted(MOVE_CONFIG.keys()))
        raise ValueError(f"Unknown move(s): {', '.join(invalid)}. Valid moves: {valid}")
    return requested


def build_settings(args: argparse.Namespace) -> RunSettings:
    if args.download_only and args.extract_only:
        raise ValueError("Use either --download-only or --extract-only, not both.")
    if args.per_query <= 0:
        raise ValueError("--per-query must be > 0")
    if args.max_per_move <= 0:
        raise ValueError("--max-per-move must be > 0")
    if args.min_duration < 0 or args.max_duration <= 0:
        raise ValueError("Duration constraints must be positive.")
    if args.min_duration > args.max_duration:
        raise ValueError("--min-duration cannot be greater than --max-duration")

    return RunSettings(
        data_root=args.data_root,
        processed_root=args.processed_root,
        report_path=args.report_path,
        download_archive=args.download_archive,
        per_query=args.per_query,
        max_per_move=args.max_per_move,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        include_shorts=args.include_shorts,
        cc_only=args.cc_only,
        download_only=args.download_only,
        extract_only=args.extract_only,
    )


def main() -> None:
    args = parse_args()
    settings = build_settings(args)
    selected_moves = resolve_moves(args.moves)

    settings.data_root.mkdir(parents=True, exist_ok=True)
    settings.processed_root.mkdir(parents=True, exist_ok=True)
    settings.report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] collecting moves: {', '.join(selected_moves)}")
    print(
        f"[info] filters: min_duration={settings.min_duration}s "
        f"max_duration={settings.max_duration}s include_shorts={settings.include_shorts} cc_only={settings.cc_only}"
    )

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "moves": selected_moves,
            "per_query": settings.per_query,
            "max_per_move": settings.max_per_move,
            "min_duration": settings.min_duration,
            "max_duration": settings.max_duration,
            "include_shorts": settings.include_shorts,
            "cc_only": settings.cc_only,
            "download_only": settings.download_only,
            "extract_only": settings.extract_only,
        },
        "results": [],
    }

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        for move in selected_moves:
            move_data_dir = settings.data_root / move
            move_proc_dir = settings.processed_root / move
            move_data_dir.mkdir(parents=True, exist_ok=True)
            move_proc_dir.mkdir(parents=True, exist_ok=True)

            move_cfg = MOVE_CONFIG[move]
            seed_urls = [u for u in move_cfg.get("seed_urls", []) if u]
            queries = [q for q in move_cfg.get("queries", []) if q]

            print(f"\n[move] {move}")
            print(f"[move] seed URLs: {len(seed_urls)} | queries: {len(queries)}")

            query_urls: list[str] = []
            if not settings.extract_only and queries:
                query_urls = search_move_urls(
                    move=move,
                    queries=queries,
                    per_query=settings.per_query,
                    include_shorts=settings.include_shorts,
                )

            candidate_urls = dedupe_urls(seed_urls + query_urls)[: settings.max_per_move]
            print(f"[move] candidate URLs after dedupe+cap: {len(candidate_urls)}")

            if settings.extract_only:
                local_files = []
                for path in sorted(move_data_dir.iterdir()):
                    if path.suffix.lower() in VIDEO_EXTENSIONS:
                        local_files.append(path)
                print(f"[move] extract-only mode, local videos found: {len(local_files)}")
                for local_video in local_files:
                    video_id = local_video.stem
                    npy_path = move_proc_dir / f"{video_id}.npy"
                    if npy_path.exists():
                        report["results"].append(
                            {
                                "move": move,
                                "video_id": video_id,
                                "status": "already_processed",
                                "video_path": str(local_video),
                                "npy_path": str(npy_path),
                            }
                        )
                        continue

                    landmarks = extract_landmarks(local_video, pose)
                    if landmarks.size == 0:
                        report["results"].append(
                            {
                                "move": move,
                                "video_id": video_id,
                                "status": "no_landmarks",
                                "video_path": str(local_video),
                            }
                        )
                        continue

                    np.save(npy_path, landmarks)
                    report["results"].append(
                        {
                            "move": move,
                            "video_id": video_id,
                            "status": "processed",
                            "video_path": str(local_video),
                            "npy_path": str(npy_path),
                            "frames_with_landmarks": int(len(landmarks)),
                        }
                    )
                continue

            for url in candidate_urls:
                video_id = extract_video_id(url)
                npy_path = move_proc_dir / f"{video_id}.npy"
                if npy_path.exists():
                    report["results"].append(
                        {
                            "move": move,
                            "url": url,
                            "video_id": video_id,
                            "status": "already_processed",
                            "npy_path": str(npy_path),
                        }
                    )
                    continue

                local_path, dl_status = download_video(
                    url=url,
                    out_dir=move_data_dir,
                    archive_file=settings.download_archive,
                    min_duration=settings.min_duration,
                    max_duration=settings.max_duration,
                    cc_only=settings.cc_only,
                )
                if local_path is None:
                    report["results"].append(
                        {
                            "move": move,
                            "url": url,
                            "video_id": video_id,
                            "status": dl_status,
                        }
                    )
                    continue

                if settings.download_only:
                    report["results"].append(
                        {
                            "move": move,
                            "url": url,
                            "video_id": local_path.stem,
                            "status": "downloaded",
                            "video_path": str(local_path),
                        }
                    )
                    continue

                landmarks = extract_landmarks(local_path, pose)
                if landmarks.size == 0:
                    report["results"].append(
                        {
                            "move": move,
                            "url": url,
                            "video_id": local_path.stem,
                            "status": "no_landmarks",
                            "video_path": str(local_path),
                        }
                    )
                    continue

                save_id = local_path.stem
                save_path = move_proc_dir / f"{save_id}.npy"
                np.save(save_path, landmarks)
                report["results"].append(
                    {
                        "move": move,
                        "url": url,
                        "video_id": save_id,
                        "status": "processed",
                        "video_path": str(local_path),
                        "npy_path": str(save_path),
                        "frames_with_landmarks": int(len(landmarks)),
                    }
                )

    with settings.report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    totals = {}
    for row in report["results"]:
        totals[row["status"]] = totals.get(row["status"], 0) + 1

    print(f"\n[done] report: {settings.report_path}")
    print(f"[done] status totals: {totals}")


if __name__ == "__main__":
    # You are responsible for respecting video licensing and platform terms.
    main()
