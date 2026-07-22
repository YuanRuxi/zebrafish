#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter non-fish crops from extracted video query frames.

The YOLO detector can occasionally crop high-contrast tank edges or background
objects as fish. This script performs a second, image-only sanity check on the
already-cropped PNGs and moves rejected pairs out of the active query folders.

Inputs:
    database/video_queries/enhanced/NNNN_s1_ZZZZ.png
    database/video_queries/raw/NNNN_s1_ZZZZ.png

Outputs when --apply is used:
    database/video_queries/rejected/<timestamp>/enhanced/*.png
    database/video_queries/rejected/<timestamp>/raw/*.png
    database/video_queries/filter_report.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


FRAMEWORK = Path(__file__).resolve().parents[1]
DEFAULT_ENH = FRAMEWORK / "database" / "video_queries" / "enhanced"
DEFAULT_RAW = FRAMEWORK / "database" / "video_queries" / "raw"
DEFAULT_REJECT = FRAMEWORK / "database" / "video_queries" / "rejected"
DEFAULT_REPORT = FRAMEWORK / "database" / "video_queries" / "filter_report.jsonl"
NAME_RE = re.compile(r"^(\d{4})_(?:c[12])?s\d_\d{4}\.png$", re.I)


def fish_id_from_name(name: str) -> str:
    m = NAME_RE.match(name)
    return m.group(1) if m else "unknown"


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)


def pca_aspect(points: np.ndarray) -> float:
    if len(points) < 10:
        return 0.0
    centered = points.astype(np.float32) - points.mean(axis=0)
    cov = np.cov(centered.T)
    if cov.shape != (2, 2) or np.isnan(cov).any():
        return 0.0
    eigvals = np.linalg.eigvalsh(cov)
    return float(math.sqrt(max(eigvals[-1], 1e-6) / max(eigvals[0], 1e-6)))


def best_fish_component(bgr: np.ndarray) -> dict:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    chroma = np.linalg.norm(lab[:, :, 1:3].astype(np.float32) - np.array([128, 128]), axis=2)

    # Fish body and fins usually contain low-value stripes, yellow/green fins,
    # or mild chroma differences. Pure tank walls and waterlines are mostly
    # low-saturation and extremely thin.
    mask = (
        ((sat > 35) & (val < 252))
        | ((chroma > 18) & (val < 250))
        | ((gray < 110) & (sat > 10) & (val < 235))
    ).astype(np.uint8)

    border = max(4, min(h, w) // 45)
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    open_k = max(3, min(h, w) // 240)
    close_k = max(7, min(h, w) // 70)
    mask = cv2.morphologyEx(
        mask * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
        iterations=1,
    )

    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    img_area = h * w
    best = {
        "score": 0.0,
        "area_ratio": 0.0,
        "bbox_aspect": 0.0,
        "pca_aspect": 0.0,
        "fill": 0.0,
        "touches_border": 0,
    }

    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_ratio = area / img_area
        if area_ratio < 0.004 or area_ratio > 0.35:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if cw <= 0 or ch <= 0:
            continue
        bbox_aspect = max(cw, ch) / max(1, min(cw, ch))
        if bbox_aspect > 35.0:
            continue
        fill = area / max(1, cw * ch)
        if fill < 0.015:
            continue
        touches = int(x <= border * 2) + int(y <= border * 2)
        touches += int(x + cw >= w - border * 2) + int(y + ch >= h - border * 2)
        if touches >= 3:
            continue

        ys, xs = np.where(labels == label)
        aspect = pca_aspect(np.column_stack([xs, ys]))
        cx = (x + cw / 2) / w
        cy = (y + ch / 2) / h
        centered = 1.0 - min(1.0, math.hypot(cx - 0.5, cy - 0.5))
        score = area_ratio * min(aspect, 12.0) * (0.5 + centered)
        if 0.015 <= area_ratio <= 0.20:
            score *= 1.4
        if aspect >= 3.0:
            score *= 1.4
        if score > best["score"]:
            best = {
                "score": float(score),
                "area_ratio": float(area_ratio),
                "bbox_aspect": float(bbox_aspect),
                "pca_aspect": float(aspect),
                "fill": float(fill),
                "touches_border": int(touches),
            }
    return best


def classify(path: Path, min_pca_aspect: float, min_score: float) -> dict:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"file": path.name, "status": "reject", "reason": "unreadable"}

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    dark_ratio = float(np.mean(gray < 80))
    dark_sat_ratio = float(np.mean((gray < 135) & (sat > 15)))
    component = best_fish_component(bgr)

    reason = "accepted"
    status = "accept"
    if dark_ratio > 0.22 or dark_sat_ratio > 0.32:
        status, reason = "reject", "background_too_dark_or_skin_toned"
    elif component["area_ratio"] <= 0:
        status, reason = "reject", "no_fishlike_component"
    elif component["pca_aspect"] < min_pca_aspect:
        status, reason = "reject", "component_not_elongated"
    elif component["score"] < min_score:
        status, reason = "reject", "low_fishlike_score"

    return {
        "file": path.name,
        "fish_id": fish_id_from_name(path.name),
        "status": status,
        "reason": reason,
        "width": int(w),
        "height": int(h),
        "dark_ratio": round(dark_ratio, 4),
        "dark_sat_ratio": round(dark_sat_ratio, 4),
        "component": {
            "score": round(component["score"], 5),
            "area_ratio": round(component["area_ratio"], 4),
            "bbox_aspect": round(component["bbox_aspect"], 3),
            "pca_aspect": round(component["pca_aspect"], 3),
            "fill": round(component["fill"], 4),
            "touches_border": component["touches_border"],
        },
    }


def move_pair(name: str, enh_dir: Path, raw_dir: Path, reject_enh: Path, reject_raw: Path) -> None:
    for src_dir, dst_dir in ((enh_dir, reject_enh), (raw_dir, reject_raw)):
        src = src_dir / name
        if not src.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / name
        if dst.exists():
            stem, suffix = dst.stem, dst.suffix
            i = 1
            while (dst_dir / f"{stem}_{i}{suffix}").exists():
                i += 1
            dst = dst_dir / f"{stem}_{i}{suffix}"
        shutil.move(str(src), str(dst))


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter non-fish video query crops.")
    parser.add_argument("--enhanced-dir", type=Path, default=DEFAULT_ENH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--reject-dir", type=Path, default=DEFAULT_REJECT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-pca-aspect", type=float, default=2.6)
    parser.add_argument("--min-score", type=float, default=0.035)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    files = sorted(
        [p for p in args.enhanced_dir.glob("*.png") if p.is_file() and NAME_RE.match(p.name)],
        key=natural_key,
    )
    if not files:
        print(f"[WARN] No PNG query frames found in {args.enhanced_dir}")
        return 0

    rows = [classify(p, args.min_pca_aspect, args.min_score) for p in files]
    status_counts = Counter(r["status"] for r in rows)
    reason_counts = Counter(r["reason"] for r in rows if r["status"] == "reject")
    per_fish = defaultdict(lambda: Counter({"accept": 0, "reject": 0}))
    for row in rows:
        per_fish[row["fish_id"]][row["status"]] += 1

    print(f"[SCAN] {len(rows)} frame(s)")
    print(f"[KEEP] {status_counts['accept']}")
    print(f"[REJECT] {status_counts['reject']}")
    if reason_counts:
        print("[REJECT REASONS]")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")
    print("[PER FISH]")
    for fish_id in sorted(per_fish):
        c = per_fish[fish_id]
        print(f"  {fish_id}: keep={c['accept']}, reject={c['reject']}")

    if args.apply:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        reject_enh = args.reject_dir / ts / "enhanced"
        reject_raw = args.reject_dir / ts / "raw"
        for row in rows:
            if row["status"] == "reject":
                move_pair(row["file"], args.enhanced_dir, args.raw_dir, reject_enh, reject_raw)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[MOVED] rejected frames -> {args.reject_dir / ts}")
        print(f"[REPORT] {args.report}")
    else:
        print("[DRY-RUN] Add --apply to move rejected frames and write the report.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
