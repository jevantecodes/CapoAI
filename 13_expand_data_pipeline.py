from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MOVES = "mini_au,esquiva_lateral,negativa,ginga,au,meia_lua_de_compasso,meia_lua_de_frente"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download more capoeira videos, auto-detect clip windows, build cleaned clips, "
            "and extract landmarks into processed_landmarks."
        )
    )
    parser.add_argument(
        "--moves",
        default=DEFAULT_MOVES,
        help=(
            "Comma-separated moves to expand. Use 'all' for every configured move. "
            f"Default: {DEFAULT_MOVES}"
        ),
    )
    parser.add_argument("--per-query", type=int, default=40, help="Search results requested per query.")
    parser.add_argument("--max-per-move", type=int, default=300, help="Max candidate videos per move.")
    parser.add_argument("--min-duration", type=int, default=2, help="Minimum video duration (seconds).")
    parser.add_argument("--max-duration", type=int, default=90, help="Maximum video duration (seconds).")
    parser.add_argument(
        "--run-tag",
        default="",
        help=(
            "Optional run id for versioned outputs. If empty, uses UTC timestamp "
            "(YYYYMMDD_HHMMSS)."
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("data/pipeline_runs"),
        help="Parent directory for versioned dataset runs.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Where raw downloaded videos are stored. Default: <run_dir>/_raw_youtube_archive",
    )
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=None,
        help="Where cleaned trimmed clips are stored. Default: <run_dir>/cleaned_archived_data",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Where extracted .npy files are stored. Default: <run_dir>/processed_landmarks",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Where final dataset tensors are written. Default: dataset/<run_tag>",
    )
    parser.add_argument(
        "--download-archive",
        type=Path,
        default=None,
        help="yt-dlp archive file. Default: <run_dir>/.youtube_download_archive.txt",
    )
    parser.add_argument(
        "--auto-trim-report",
        type=Path,
        default=None,
        help="Output JSON for detected trim windows. Default: <run_dir>/auto_trim_report.json",
    )
    parser.add_argument(
        "--collection-report",
        type=Path,
        default=None,
        help="Download step report path. Default: <run_dir>/collection_report.json",
    )
    parser.add_argument(
        "--extract-report",
        type=Path,
        default=None,
        help="Extract step report path. Default: <run_dir>/cleaned_extract_report.json",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Run 02_preprocess_data.py after landmark extraction.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip the download step.")
    parser.add_argument("--skip-trim-report", action="store_true", help="Skip trim-window detection step.")
    parser.add_argument("--skip-build-cleaned", action="store_true", help="Skip cleaned clip build step.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip landmark extraction step.")
    return parser.parse_args()


def run_step(cmd: list[str], label: str) -> None:
    print(f"\n[step] {label}")
    print("[cmd] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def resolve_paths(args: argparse.Namespace) -> tuple[str, dict[str, Path]]:
    run_tag = args.run_tag.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.runs_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    archive_root = args.archive_root or (run_dir / "_raw_youtube_archive")
    cleaned_root = args.cleaned_root or (run_dir / "cleaned_archived_data")
    processed_root = args.processed_root or (run_dir / "processed_landmarks")
    dataset_root = args.dataset_root or (Path("dataset") / run_tag)
    download_archive = args.download_archive or (run_dir / ".youtube_download_archive.txt")
    auto_trim_report = args.auto_trim_report or (run_dir / "auto_trim_report.json")
    collection_report = args.collection_report or (run_dir / "collection_report.json")
    extract_report = args.extract_report or (run_dir / "cleaned_extract_report.json")

    return run_tag, {
        "run_dir": run_dir,
        "archive_root": archive_root,
        "cleaned_root": cleaned_root,
        "processed_root": processed_root,
        "dataset_root": dataset_root,
        "download_archive": download_archive,
        "auto_trim_report": auto_trim_report,
        "collection_report": collection_report,
        "extract_report": extract_report,
    }


def main() -> None:
    args = parse_args()
    py = sys.executable
    run_tag, paths = resolve_paths(args)

    print("\n[run] " + run_tag)
    print("[run_dir] " + str(paths["run_dir"]))
    print("[archive_root] " + str(paths["archive_root"]))
    print("[cleaned_root] " + str(paths["cleaned_root"]))
    print("[processed_root] " + str(paths["processed_root"]))
    print("[dataset_root] " + str(paths["dataset_root"]))

    if not args.skip_download:
        run_step(
            [
                py,
                "00_download_and_process_cc_videos.py",
                "--moves",
                args.moves,
                "--per-query",
                str(args.per_query),
                "--max-per-move",
                str(args.max_per_move),
                "--min-duration",
                str(args.min_duration),
                "--max-duration",
                str(args.max_duration),
                "--download-only",
                "--data-root",
                str(paths["archive_root"]),
                "--processed-root",
                str(paths["processed_root"]),
                "--report-path",
                str(paths["collection_report"]),
                "--download-archive",
                str(paths["download_archive"]),
            ],
            "Download more raw videos",
        )

    if not args.skip_trim_report:
        run_step(
            [
                py,
                "10_auto_trim_archive_clips.py",
                "--archive-root",
                str(paths["archive_root"]),
                "--output-root",
                "data",
                "--report-path",
                str(paths["auto_trim_report"]),
                "--dry-run",
                "--overwrite",
            ],
            "Detect trim windows (report only)",
        )

    if not args.skip_build_cleaned:
        run_step(
            [
                py,
                "11_build_cleaned_archived_data.py",
                "--report-path",
                str(paths["auto_trim_report"]),
                "--cleaned-root",
                str(paths["cleaned_root"]),
                "--overwrite",
            ],
            "Build cleaned clips from trim report",
        )

    if not args.skip_extract:
        run_step(
            [
                py,
                "00_download_and_process_cc_videos.py",
                "--moves",
                args.moves,
                "--extract-only",
                "--data-root",
                str(paths["cleaned_root"]),
                "--processed-root",
                str(paths["processed_root"]),
                "--report-path",
                str(paths["extract_report"]),
            ],
            "Extract landmarks from cleaned clips",
        )

    if args.rebuild_dataset:
        run_step(
            [
                py,
                "02_preprocess_data.py",
                "--landmark-dir",
                str(paths["processed_root"]),
                "--output-dir",
                str(paths["dataset_root"]),
            ],
            "Rebuild dataset tensors",
        )

    manifest = {
        "run_tag": run_tag,
        "run_dir": str(paths["run_dir"]),
        "moves": args.moves,
        "per_query": int(args.per_query),
        "max_per_move": int(args.max_per_move),
        "archive_root": str(paths["archive_root"]),
        "cleaned_root": str(paths["cleaned_root"]),
        "processed_root": str(paths["processed_root"]),
        "dataset_root": str(paths["dataset_root"]),
        "download_archive": str(paths["download_archive"]),
        "collection_report": str(paths["collection_report"]),
        "auto_trim_report": str(paths["auto_trim_report"]),
        "extract_report": str(paths["extract_report"]),
        "rebuild_dataset": bool(args.rebuild_dataset),
        "skip_download": bool(args.skip_download),
        "skip_trim_report": bool(args.skip_trim_report),
        "skip_build_cleaned": bool(args.skip_build_cleaned),
        "skip_extract": bool(args.skip_extract),
    }
    manifest_path = paths["run_dir"] / "pipeline_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n[done] Expansion pipeline complete.")
    print("[manifest] " + str(manifest_path))
    if args.rebuild_dataset:
        print("[dataset] " + str(paths["dataset_root"]))


if __name__ == "__main__":
    main()
