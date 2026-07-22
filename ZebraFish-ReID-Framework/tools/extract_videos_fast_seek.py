#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast extraction of high-quality zebrafish side-view frames from videos.

Unlike src/preprocessing/video_extractor.py, this script jumps to evenly spaced
candidate frame indices instead of decoding every frame sequentially. It is meant
for evaluation runs where each fish video only needs a moderate number of clear,
mostly side-view frames.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


FRAMEWORK = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = FRAMEWORK / "src" / "preprocessing"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

import video_extractor as ve
from enhance import enhance_clahe
from quality import compute_blur_score


DEFAULT_SRC = Path(r"C:\Users\JiangYao\Desktop\26_Medical\7.14斑马鱼照片+视频")
DEFAULT_RAW = FRAMEWORK / "database" / "video_queries" / "raw"
DEFAULT_ENH = FRAMEWORK / "database" / "video_queries" / "enhanced"
DEFAULT_REPORT = FRAMEWORK / "database" / "video_queries" / "video_crop_report.jsonl"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)


def fish_id_from_folder(path: Path) -> str | None:
    digits = "".join(ch for ch in path.name if ch.isdigit())
    if not digits:
        return None
    return f"{int(digits):04d}"


def collect_videos(src: Path, fish_start: int, fish_end: int) -> list[tuple[Path, str]]:
    out = []
    for folder in sorted(src.iterdir(), key=natural_key):
        if not folder.is_dir():
            continue
        fish_id = fish_id_from_folder(folder)
        if fish_id is None:
            continue
        fish_no = int(fish_id)
        if not (fish_start <= fish_no <= fish_end):
            continue
        for video in sorted(folder.iterdir(), key=natural_key):
            if video.is_file() and video.suffix.lower() in VIDEO_EXTS:
                out.append((video, fish_id))
    return out


def clear_outputs(*paths: Path) -> None:
    for path in paths:
        if path.suffix:
            if path.exists():
                path.unlink()
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
            for child in path.iterdir():
                if child.is_file():
                    child.unlink()


def frame_mae(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ga = cv2.resize(ga, (64, 64))
    gb = cv2.resize(gb, (64, 64))
    return float(np.mean(np.abs(ga.astype(np.float32) - gb.astype(np.float32)) / 255.0))


def is_side_like(crop: np.ndarray, min_aspect: float, max_aspect: float) -> tuple[bool, float]:
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return False, 0.0
    aspect = max(w, h) / max(1, min(w, h))
    return min_aspect <= aspect <= max_aspect, float(aspect)


def candidate_indices(frame_count: int, samples: int) -> list[int]:
    if frame_count <= 0:
        return []
    # Avoid the very first/last moments, which often include handling or empty frames.
    start = int(frame_count * 0.03)
    end = max(start + 1, int(frame_count * 0.97))
    if samples >= end - start:
        return list(range(start, end))
    return sorted(set(int(x) for x in np.linspace(start, end - 1, samples)))


def extract_video(
    video: Path,
    fish_id: str,
    model,
    counter: dict[str, int],
    raw_dir: Path,
    enh_dir: Path,
    report,
    max_frames: int,
    samples_per_video: int,
    conf: float,
    imgsz: int,
    margin: float,
    blur_thresh: float,
    dedup: float,
    min_aspect: float,
    max_aspect: float,
) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  [warn] cannot open {video}")
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = candidate_indices(frame_count, samples_per_video)
    accepted = 0
    last_crop = None
    for idx in indices:
        if counter.get(fish_id, 0) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        det = ve.detect_fish(frame, model, conf, imgsz)
        if det is None:
            continue
        box5, yolo_conf = det
        crop = ve.crop_with_margin(frame, box5, margin)
        if crop is None or crop.size == 0:
            continue
        side_ok, aspect = is_side_like(crop, min_aspect, max_aspect)
        if not side_ok:
            continue
        blur = compute_blur_score(crop)
        if blur < blur_thresh:
            continue
        if last_crop is not None and frame_mae(crop, last_crop) < dedup:
            continue
        last_crop = crop

        counter[fish_id] = counter.get(fish_id, 0) + 1
        frame_no = counter[fish_id]
        fname = f"{fish_id}_s1_{frame_no:04d}.png"
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(raw_dir / fname)
        enh = enhance_clahe(crop, clip_limit=2.0)
        Image.fromarray(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)).save(enh_dir / fname)
        report.write(
            json.dumps(
                {
                    "file": fname,
                    "status": "accepted",
                    "fish_id": fish_id,
                    "video": video.name,
                    "frame_idx": int(idx),
                    "conf": round(float(yolo_conf), 4),
                    "blur_score": round(float(blur), 2),
                    "aspect": round(float(aspect), 3),
                    "cam": None,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        accepted += 1
    cap.release()
    return accepted


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast seek-based zebrafish video frame extraction.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out-raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-enh", type=Path, default=DEFAULT_ENH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--fish-start", type=int, default=1)
    parser.add_argument("--fish-end", type=int, default=24)
    parser.add_argument("--samples-per-video", type=int, default=80)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--margin", type=float, default=0.22)
    parser.add_argument("--blur", type=float, default=3.0)
    parser.add_argument("--dedup", type=float, default=0.12)
    parser.add_argument("--min-aspect", type=float, default=1.45)
    parser.add_argument("--max-aspect", type=float, default=9.0)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = args.src.resolve()
    videos = collect_videos(src, args.fish_start, args.fish_end)
    if not videos:
        print(f"[ERROR] No videos found in {src}")
        return 1
    print(f"[SOURCE] {src}")
    print(f"[VIDEOS] {len(videos)}")
    per_fish_videos: dict[str, int] = {}
    for _video, fish_id in videos:
        per_fish_videos[fish_id] = per_fish_videos.get(fish_id, 0) + 1
    for fish_id in sorted(per_fish_videos):
        print(f"  fish {fish_id}: {per_fish_videos[fish_id]} video(s)")
    print(
        f"[PARAM] samples/video={args.samples_per_video}, max_frames/fish={args.max_frames}, "
        f"conf={args.conf}, imgsz={args.imgsz}, aspect=[{args.min_aspect},{args.max_aspect}]"
    )
    if args.dry_run:
        return 0

    if not args.keep:
        clear_outputs(args.out_raw, args.out_enh, args.report)
    else:
        args.out_raw.mkdir(parents=True, exist_ok=True)
        args.out_enh.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)

    model = ve.load_yolo(ve.resolve_yolo("models/yolov8_zebrafish.pt"))
    counter: dict[str, int] = {}
    total = 0
    with open(args.report, "w", encoding="utf-8") as report:
        for video, fish_id in videos:
            n = extract_video(
                video,
                fish_id,
                model,
                counter,
                args.out_raw,
                args.out_enh,
                report,
                args.max_frames,
                args.samples_per_video,
                args.conf,
                args.imgsz,
                args.margin,
                args.blur,
                args.dedup,
                args.min_aspect,
                args.max_aspect,
            )
            total += n
            print(f"  [ok] fish {fish_id} {video.name}: accepted {n}")

    summary = {"total": total, "per_fish": counter}
    summary_path = args.report.parent / "video_extract_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] accepted {total} frames")
    print(f"[SUMMARY] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
