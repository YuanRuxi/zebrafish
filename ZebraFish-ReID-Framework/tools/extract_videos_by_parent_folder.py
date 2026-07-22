#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract high-quality zebrafish frames from videos organized by fish-ID folders.

Expected source layout:

    SOURCE/
      1/*.mov
      2/*.mov
      ...

The parent folder number is used as fish ID, so timestamp-named videos are fine.
Outputs use the project convention before side labeling:

    NNNN_s1_ZZZZ.png

Run tools/side_geometry_label.py afterwards to rename selected frames to c1/c2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FRAMEWORK = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = FRAMEWORK / "src" / "preprocessing"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

import video_extractor as ve


DEFAULT_SRC = Path(r"C:\Users\JiangYao\Desktop\26_Medical\7.14斑马鱼照片+视频")
DEFAULT_RAW = FRAMEWORK / "database" / "video_queries" / "raw"
DEFAULT_ENH = FRAMEWORK / "database" / "video_queries" / "enhanced"
DEFAULT_REPORT = FRAMEWORK / "database" / "video_queries" / "video_crop_report.jsonl"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def natural_key(path: Path) -> tuple:
    import re

    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)


def fish_id_from_folder(path: Path) -> str | None:
    digits = "".join(ch for ch in path.name if ch.isdigit())
    if not digits:
        return None
    value = int(digits)
    if value <= 0:
        return None
    return f"{value:04d}"


def collect_videos(src: Path, fish_start: int | None, fish_end: int | None) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for folder in sorted(src.iterdir(), key=natural_key):
        if not folder.is_dir():
            continue
        fish_id = fish_id_from_folder(folder)
        if fish_id is None:
            continue
        fish_no = int(fish_id)
        if fish_start is not None and fish_no < fish_start:
            continue
        if fish_end is not None and fish_no > fish_end:
            continue
        videos = [p for p in sorted(folder.iterdir(), key=natural_key) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        for video in videos:
            items.append((video, fish_id))
    return items


def clear_outputs(raw_dir: Path, enh_dir: Path, report: Path) -> None:
    for folder in (raw_dir, enh_dir):
        folder.mkdir(parents=True, exist_ok=True)
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()
    if report.exists():
        report.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract video frames using parent folder as fish ID.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out-raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-enh", type=Path, default=DEFAULT_ENH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--fish-start", type=int, default=1)
    parser.add_argument("--fish-end", type=int, default=24)
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=80, help="Maximum accepted frames per fish.")
    parser.add_argument("--dedup", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.22)
    parser.add_argument("--blur", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep", action="store_true", help="Do not clear previous outputs first.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = args.src.resolve()
    if not src.exists() or not src.is_dir():
        print(f"[ERROR] Source directory not found: {src}")
        return 1

    videos = collect_videos(src, args.fish_start, args.fish_end)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"[ERROR] No videos found in {src}")
        return 1

    print(f"[SOURCE] {src}")
    print(f"[VIDEOS] {len(videos)}")
    by_fish = {}
    for video, fish_id in videos:
        by_fish.setdefault(fish_id, 0)
        by_fish[fish_id] += 1
    for fish_id in sorted(by_fish):
        print(f"  fish {fish_id}: {by_fish[fish_id]} video(s)")
    print(f"[OUT RAW] {args.out_raw}")
    print(f"[OUT ENH] {args.out_enh}")
    print(f"[PARAM] conf={args.conf}, imgsz={args.imgsz}, stride={args.stride}, "
          f"max_frames={args.max_frames}, dedup={args.dedup}, margin={args.margin}, blur={args.blur}")

    if args.dry_run:
        print("\nDry run only.")
        return 0

    if not args.keep:
        clear_outputs(args.out_raw, args.out_enh, args.report)
    else:
        args.out_raw.mkdir(parents=True, exist_ok=True)
        args.out_enh.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)

    cfg = ve.load_config()
    cfg.update(
        {
            "conf_side": args.conf,
            "imgsz": args.imgsz,
            "sample_stride": args.stride,
            "max_frames_per_fish": args.max_frames,
            "dedup_mae": args.dedup,
            "margin_ratio": args.margin,
            "blur_thresh": args.blur,
            "use_enhance": True,
        }
    )
    yolo_path = ve.resolve_yolo(cfg["yolo_model"])
    print(f"[YOLO] {yolo_path}")
    model = ve.load_yolo(yolo_path)

    counter: dict[str, int] = {}
    total = 0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as rf:
        for video, fish_id in videos:
            result = ve.process_video(
                str(video),
                model,
                cfg,
                args.blur,
                str(args.out_raw),
                str(args.out_enh),
                rf,
                counter,
                args.max_frames,
                False,
                True,
                fish_id=fish_id,
            )
            accepted = int(result.get("accepted", 0))
            total += accepted
            print(f"  [ok] fish {fish_id} {video.name}: accepted {accepted}")

    summary_path = args.report.parent / "video_extract_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"total": total, "per_fish": counter}, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] accepted frames: {total}")
    print(f"[REPORT] {args.report}")
    print(f"[SUMMARY] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
