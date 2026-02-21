import argparse
import shutil
from pathlib import Path
import re


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move raw YouTube-ID-named videos out of data/<move>/ to an archive folder. "
            "Only targets files like 9bPA_7myDag.mp4."
        )
    )
    parser.add_argument("--data-dir", default="data", help="Root data directory.")
    parser.add_argument(
        "--archive-dir",
        default="data/_raw_youtube_archive",
        help="Where matched files are moved.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, runs as dry-run.",
    )
    return parser.parse_args()


def is_youtube_named_video(path: Path) -> bool:
    if path.suffix.lower() not in VIDEO_EXTS:
        return False
    return bool(YOUTUBE_ID_RE.match(path.stem))


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    idx = 1
    while True:
        candidate = path.with_name(f"{stem}_dup{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    archive_dir = Path(args.archive_dir).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    candidates: list[Path] = []
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if archive_dir in path.parents:
            continue
        if is_youtube_named_video(path):
            candidates.append(path)

    candidates.sort()
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] Found {len(candidates)} YouTube-ID-style video files.")

    moved = 0
    for src in candidates:
        rel = src.relative_to(data_dir)
        dst = unique_destination(archive_dir / rel)
        print(f"{rel} -> {dst.relative_to(data_dir.parent)}")
        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved += 1

    if args.apply:
        print(f"[DONE] Moved {moved} files to {archive_dir}.")
    else:
        print("[DONE] Dry-run only. Re-run with --apply to perform moves.")


if __name__ == "__main__":
    main()
