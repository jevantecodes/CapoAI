from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class ClipResult:
    source: str
    output: str | None
    status: str
    fps: float | None = None
    frame_count: int | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-trim raw videos from data/_raw_youtube_archive by detecting active movement "
            "and writing one clipped sample per source video to data/<move>/."
        )
    )
    parser.add_argument("--archive-root", type=Path, default=Path("data/_raw_youtube_archive"))
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--report-path", type=Path, default=Path("data/auto_trim_report.json"))
    parser.add_argument("--min-clip-sec", type=float, default=1.4)
    parser.add_argument("--max-clip-sec", type=float, default=4.0)
    parser.add_argument("--pad-sec", type=float, default=0.35)
    parser.add_argument("--z-threshold", type=float, default=1.8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _has_video_backend() -> bool:
    info = cv2.getBuildInformation()
    ffmpeg_ok = "FFMPEG:                      YES" in info
    gst_ok = "GStreamer:                   YES" in info
    return ffmpeg_ok or gst_ok


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def list_videos(root: Path) -> list[Path]:
    videos: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            videos.append(path)
    videos.sort()
    return videos


def compute_motion_scores(video_path: Path) -> tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_gray = None
    scores: list[float] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (192, 108), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            scores.append(float(np.mean(diff)))
        prev_gray = gray

    cap.release()

    if not scores:
        return np.zeros((0,), dtype=np.float32), fps, frame_count
    return np.array(scores, dtype=np.float32), fps, frame_count


def smooth_signal(values: np.ndarray, win_size: int) -> np.ndarray:
    if values.size == 0:
        return values
    win_size = max(1, win_size)
    kernel = np.ones((win_size,), dtype=np.float32) / float(win_size)
    return np.convolve(values, kernel, mode="same")


def detect_clip_window(
    motion_scores: np.ndarray,
    fps: float,
    frame_count: int,
    min_clip_sec: float,
    max_clip_sec: float,
    pad_sec: float,
    z_threshold: float,
) -> tuple[int, int, str]:
    if frame_count <= 1:
        return 0, 0, "single_frame_video"

    if motion_scores.size == 0:
        return 0, frame_count - 1, "no_motion_scores"

    smooth_win = max(3, int(round(fps * 0.20)))
    smooth = smooth_signal(motion_scores, smooth_win)

    baseline = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - baseline))) + 1e-6
    robust_sigma = 1.4826 * mad
    z = (smooth - baseline) / robust_sigma

    active = z >= z_threshold

    min_len = max(1, int(round(min_clip_sec * fps)))
    max_len = max(min_len, int(round(max_clip_sec * fps)))
    pad_frames = max(0, int(round(pad_sec * fps)))

    best_start = 0
    best_end = min(frame_count - 1, max_len - 1)
    best_score = -1.0

    i = 0
    n = len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and active[j + 1]:
            j += 1

        # motion index maps to transition t->t+1, so frame window is [i, j+1]
        start = max(0, i - pad_frames)
        end = min(frame_count - 1, (j + 1) + pad_frames)

        seg_len = end - start + 1
        if seg_len > max_len:
            center = (start + end) // 2
            half = max_len // 2
            start = max(0, center - half)
            end = min(frame_count - 1, start + max_len - 1)
            start = max(0, end - max_len + 1)
            seg_len = end - start + 1

        if seg_len < min_len:
            add = min_len - seg_len
            left = add // 2
            right = add - left
            start = max(0, start - left)
            end = min(frame_count - 1, end + right)
            if end - start + 1 < min_len:
                start = max(0, end - min_len + 1)

        score = float(np.sum(smooth[max(i, 0) : min(j + 1, len(smooth))]))
        if score > best_score:
            best_score = score
            best_start = start
            best_end = end

        i = j + 1

    if best_score < 0:
        # fallback: highest-energy fixed window
        win = min(max_len, max(min_len, frame_count))
        if len(smooth) < 1:
            return 0, min(frame_count - 1, win - 1), "fallback_empty_smooth"

        # frame_count = len(smooth)+1 approximately
        max_start = max(0, frame_count - win)
        best_sum = -1.0
        best_idx = 0
        for start in range(0, max_start + 1):
            s0 = max(0, start)
            s1 = min(len(smooth), start + win - 1)
            val = float(np.sum(smooth[s0:s1]))
            if val > best_sum:
                best_sum = val
                best_idx = start
        return best_idx, min(frame_count - 1, best_idx + win - 1), "fallback_peak_window"

    return best_start, best_end, "active_motion_segment"


def write_trimmed_clip_ffmpeg(src: Path, dst: Path, start_sec: float, end_sec: float) -> None:
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
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        snippet = err[-1] if err else "ffmpeg failed"
        raise ValueError(f"ffmpeg trim failed: {snippet}")
    if not dst.exists() or dst.stat().st_size == 0:
        raise ValueError("Failed to write trimmed clip with ffmpeg.")


def write_trimmed_clip_opencv(src: Path, dst: Path, start_frame: int, end_frame: int) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError("Could not open source video for writing.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Could not read video dimensions.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    frame_idx = max(0, start_frame)
    while cap.isOpened() and frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if not dst.exists() or dst.stat().st_size == 0:
        raise ValueError("Failed to write trimmed clip.")


def write_trimmed_clip(src: Path, dst: Path, start_frame: int, end_frame: int, fps: float) -> None:
    if _ffmpeg_available():
        write_trimmed_clip_ffmpeg(
            src=src,
            dst=dst,
            start_sec=float(start_frame / fps),
            end_sec=float(end_frame / fps),
        )
        return
    write_trimmed_clip_opencv(src, dst, start_frame, end_frame)


def output_path_for(source_path: Path, archive_root: Path, output_root: Path) -> Path:
    rel = source_path.relative_to(archive_root)
    move = rel.parts[0] if len(rel.parts) > 1 else "unknown_move"
    src_stem = source_path.stem
    name = f"{move}_auto_{src_stem}.mp4"
    return output_root / move / name


def main() -> None:
    args = parse_args()
    archive_root = args.archive_root.resolve()
    output_root = args.output_root.resolve()
    report_path = args.report_path.resolve()

    if not _has_video_backend():
        raise RuntimeError(
            "OpenCV in this environment has no video backend enabled "
            "(FFMPEG/GStreamer). Install an OpenCV build with FFMPEG support."
        )

    if not archive_root.exists():
        raise FileNotFoundError(f"Archive root not found: {archive_root}")

    videos = list_videos(archive_root)
    print(f"[scan] found {len(videos)} source videos in {archive_root}")

    results: list[ClipResult] = []
    written = 0
    skipped = 0
    failed = 0

    for idx, src in enumerate(videos, start=1):
        dst = output_path_for(src, archive_root, output_root)
        rel_src = str(src.relative_to(archive_root))
        rel_dst = str(dst.relative_to(output_root.parent))

        if dst.exists() and not args.overwrite:
            results.append(
                ClipResult(
                    source=rel_src,
                    output=rel_dst,
                    status="skipped_exists",
                    reason="output already exists",
                )
            )
            skipped += 1
            if idx % 25 == 0:
                print(f"[progress] {idx}/{len(videos)} processed")
            continue

        try:
            motion_scores, fps, frame_count = compute_motion_scores(src)
            start_frame, end_frame, reason = detect_clip_window(
                motion_scores=motion_scores,
                fps=fps,
                frame_count=frame_count,
                min_clip_sec=args.min_clip_sec,
                max_clip_sec=args.max_clip_sec,
                pad_sec=args.pad_sec,
                z_threshold=args.z_threshold,
            )
            if end_frame < start_frame:
                end_frame = start_frame

            if not args.dry_run:
                write_trimmed_clip(src, dst, start_frame, end_frame, fps=fps)

            results.append(
                ClipResult(
                    source=rel_src,
                    output=rel_dst,
                    status="ok",
                    fps=fps,
                    frame_count=frame_count,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_sec=float(start_frame / fps) if fps > 0 else None,
                    end_sec=float(end_frame / fps) if fps > 0 else None,
                    reason=reason,
                )
            )
            written += 1
        except Exception as exc:
            results.append(
                ClipResult(
                    source=rel_src,
                    output=rel_dst,
                    status="error",
                    reason=str(exc),
                )
            )
            failed += 1

        if idx % 25 == 0:
            print(f"[progress] {idx}/{len(videos)} processed")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "archive_root": str(archive_root),
        "output_root": str(output_root),
        "total_sources": len(videos),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "dry_run": bool(args.dry_run),
        "settings": {
            "min_clip_sec": args.min_clip_sec,
            "max_clip_sec": args.max_clip_sec,
            "pad_sec": args.pad_sec,
            "z_threshold": args.z_threshold,
            "overwrite": bool(args.overwrite),
        },
        "results": [asdict(row) for row in results],
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"[done] written={written} skipped={skipped} failed={failed} "
        f"report={report_path}"
    )


if __name__ == "__main__":
    main()
