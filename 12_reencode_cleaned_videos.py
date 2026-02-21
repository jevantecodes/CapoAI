from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-encode cleaned clips to browser-friendly H.264/AAC MP4. "
            "Useful when VS Code/video players cannot preview generated clips."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/cleaned_archived_data"),
        help="Root directory containing cleaned clips.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace files in place. Default writes to sibling folder <root>_h264.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "--glob",
        default="*.mp4",
        help=(
            "Filename glob filter relative to --root (example: '*_auto_*.mp4'). "
            "Default: '*.mp4'."
        ),
    )
    return parser.parse_args()


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def list_videos(root: Path, pattern: str) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob(pattern):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            files.append(p)
    files.sort()
    return files


def reencode(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".__tmp__.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        snippet = err[-1] if err else "ffmpeg failed"
        raise RuntimeError(snippet)
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced empty output")

    if dst.exists():
        dst.unlink()
    tmp.replace(dst)


def main() -> None:
    args = parse_args()
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found. Install first (e.g. `brew install ffmpeg`).")

    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    if args.in_place:
        out_root = root
    else:
        out_root = root.parent / f"{root.name}_h264"

    videos = list_videos(root, args.glob)
    print(f"[scan] videos={len(videos)} source={root} out={out_root} glob={args.glob}")

    ok = 0
    skip = 0
    fail = 0
    failures: list[dict[str, str]] = []

    for i, src in enumerate(videos, start=1):
        rel = src.relative_to(root)
        dst = out_root / rel.with_suffix(".mp4")
        if dst.exists() and not args.overwrite and not args.in_place:
            skip += 1
            continue
        if dst.exists() and not args.overwrite and args.in_place:
            skip += 1
            continue

        try:
            reencode(src, dst)
            ok += 1
        except Exception as exc:
            fail += 1
            failures.append({"file": str(rel), "error": str(exc)})

        if i % 50 == 0:
            print(f"[progress] {i}/{len(videos)}")

    report = {
        "source_root": str(root),
        "output_root": str(out_root),
        "total": len(videos),
        "ok": ok,
        "skipped": skip,
        "failed": fail,
        "in_place": bool(args.in_place),
        "failures": failures,
    }
    report_path = out_root / "reencode_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[done] ok={ok} skipped={skip} failed={fail} report={report_path}")


if __name__ == "__main__":
    main()
