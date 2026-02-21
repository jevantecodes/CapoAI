from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2


@dataclass
class BuildResult:
    source: str
    destination: str
    status: str
    reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build data/cleaned_archived_data from auto_trim_report.json using "
            "start_sec/end_sec windows."
        )
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/auto_trim_report.json"),
        help="Path to auto trim report JSON.",
    )
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=Path("data/cleaned_archived_data"),
        help="Target root for cleaned clips.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "--use-existing-output",
        action="store_true",
        help=(
            "Copy the prebuilt file from report.results[].output when it exists. "
            "Default behavior is to trim from archive source using start_sec/end_sec."
        ),
    )
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="Disable ffmpeg and use OpenCV-only trimming.",
    )
    return parser.parse_args()


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def trim_by_seconds_ffmpeg(src: Path, dst: Path, start_sec: float, end_sec: float) -> None:
    duration = max(0.01, float(end_sec) - float(start_sec))
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
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
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        snippet = (proc.stderr or "").strip().splitlines()[-1:] or ["ffmpeg failed"]
        raise ValueError(f"ffmpeg trim failed: {snippet[0]}")
    if not dst.exists() or dst.stat().st_size == 0:
        raise ValueError(f"ffmpeg wrote empty output: {dst}")


def trim_by_seconds_opencv(src: Path, dst: Path, start_sec: float, end_sec: float) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"Could not open source video: {src}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Could not read dimensions for: {src}")

    start_frame = max(0, int(start_sec * fps))
    end_frame = max(start_frame, int(end_sec * fps))

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while cap.isOpened() and frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if not dst.exists() or dst.stat().st_size == 0:
        raise ValueError(f"Failed to write trimmed output: {dst}")


def trim_by_seconds(
    src: Path,
    dst: Path,
    start_sec: float,
    end_sec: float,
    prefer_ffmpeg: bool = True,
) -> None:
    if prefer_ffmpeg:
        if not _ffmpeg_available():
            raise RuntimeError(
                "ffmpeg is required for browser-compatible clips. "
                "Install ffmpeg or run with --no-ffmpeg (not recommended)."
            )
        trim_by_seconds_ffmpeg(src, dst, start_sec, end_sec)
        return
    trim_by_seconds_opencv(src, dst, start_sec, end_sec)


def resolve_existing_trimmed_path(project_root: Path, output_field: str | None) -> Path | None:
    if not output_field:
        return None
    p = Path(output_field)
    if not p.is_absolute():
        p = project_root / p
    return p


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()

    report_path = args.report_path
    if not report_path.is_absolute():
        report_path = project_root / report_path
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    archive_root = Path(report.get("archive_root", "data/_raw_youtube_archive"))
    if not archive_root.is_absolute():
        archive_root = project_root / archive_root

    cleaned_root = args.cleaned_root
    if not cleaned_root.is_absolute():
        cleaned_root = project_root / cleaned_root
    cleaned_root.mkdir(parents=True, exist_ok=True)

    rows = report.get("results", [])
    print(f"[scan] rows={len(rows)} report={report_path}")
    print(
        f"[config] use_existing_output={bool(args.use_existing_output)} "
        f"ffmpeg_available={_ffmpeg_available()} no_ffmpeg={bool(args.no_ffmpeg)}"
    )

    copied = 0
    trimmed = 0
    skipped = 0
    failed = 0
    results: list[BuildResult] = []

    for idx, row in enumerate(rows, start=1):
        source_rel = row.get("source")
        if not source_rel:
            failed += 1
            results.append(
                BuildResult(
                    source="",
                    destination="",
                    status="error",
                    reason="missing source field",
                )
            )
            continue

        src_rel_path = Path(source_rel)
        src_original = archive_root / src_rel_path
        dst = cleaned_root / src_rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not args.overwrite:
            skipped += 1
            results.append(
                BuildResult(
                    source=source_rel,
                    destination=str(dst.relative_to(project_root)),
                    status="skipped_exists",
                )
            )
            continue

        existing_trimmed = resolve_existing_trimmed_path(project_root, row.get("output"))
        try:
            if args.use_existing_output and existing_trimmed is not None and existing_trimmed.exists():
                shutil.copy2(existing_trimmed, dst)
                copied += 1
                results.append(
                    BuildResult(
                        source=source_rel,
                        destination=str(dst.relative_to(project_root)),
                        status="ok_copied_existing_trim",
                    )
                )
            else:
                start_sec = row.get("start_sec")
                end_sec = row.get("end_sec")
                if start_sec is None or end_sec is None:
                    raise ValueError("missing start_sec or end_sec in report row")
                if not src_original.exists():
                    raise FileNotFoundError(f"source not found: {src_original}")
                trim_by_seconds(
                    src=src_original,
                    dst=dst,
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    prefer_ffmpeg=not args.no_ffmpeg,
                )
                trimmed += 1
                results.append(
                    BuildResult(
                        source=source_rel,
                        destination=str(dst.relative_to(project_root)),
                        status="ok_trimmed_from_source",
                    )
                )
        except Exception as exc:
            failed += 1
            results.append(
                BuildResult(
                    source=source_rel,
                    destination=str(dst.relative_to(project_root)),
                    status="error",
                    reason=str(exc),
                )
            )

        if idx % 50 == 0:
            print(f"[progress] {idx}/{len(rows)}")

    summary = {
        "source_report": str(report_path),
        "cleaned_root": str(cleaned_root),
        "total_rows": len(rows),
        "copied_existing_trim": copied,
        "trimmed_from_source": trimmed,
        "skipped_exists": skipped,
        "failed": failed,
        "results": [asdict(r) for r in results],
    }
    output_report = cleaned_root / "build_report.json"
    with output_report.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[done] copied={copied} trimmed={trimmed} skipped={skipped} failed={failed} "
        f"report={output_report}"
    )


if __name__ == "__main__":
    main()
