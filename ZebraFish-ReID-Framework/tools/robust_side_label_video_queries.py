#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustly assign c1/c2 side labels to video query crops.

This is a conservative replacement for the older geometry-only labeling pass.
It uses a color-aware fish mask and scores multiple eye candidates, instead of
using the single darkest pixel, which can be a tail stripe.
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
DEFAULT_RAW = FRAMEWORK / "database" / "video_queries" / "relabel_work" / "raw_sonly"
DEFAULT_ENH = FRAMEWORK / "database" / "video_queries" / "relabel_work" / "enhanced_sonly"
DEFAULT_REPORT = FRAMEWORK / "database" / "video_queries" / "robust_side_report.jsonl"
DEFAULT_BACKUP = FRAMEWORK / "database" / "video_queries" / "robust_side_backup"
NAME_RE = re.compile(r"^(\d{4})_(?:c([12]))?s(\d)_(\d{4})\.png$", re.I)


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)


def fish_mask(rgb: np.ndarray) -> np.ndarray | None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    chroma = np.linalg.norm(lab[:, :, 1:3].astype(np.float32) - np.array([128, 128]), axis=2)

    mask = (
        ((sat > 30) & (val < 252))
        | ((chroma > 16) & (val < 250))
        | ((gray < 125) & (sat > 8) & (val < 245))
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

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None

    best_label = None
    best_score = -1.0
    img_area = h * w
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_ratio = area / img_area
        if area_ratio < 0.004 or area_ratio > 0.35:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox_aspect = max(cw, ch) / max(1, min(cw, ch))
        fill = area / max(1, cw * ch)
        if bbox_aspect > 35.0 or fill < 0.015:
            continue
        ys, xs = np.where(labels == label)
        coords = np.column_stack([xs, ys]).astype(np.float32)
        if len(coords) < 20:
            continue
        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered.T)
        if cov.shape != (2, 2) or np.isnan(cov).any():
            continue
        eigvals = np.linalg.eigvalsh(cov)
        pca_aspect = math.sqrt(max(eigvals[-1], 1e-6) / max(eigvals[0], 1e-6))
        score = area_ratio * min(pca_aspect, 12.0)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None
    out = labels == best_label
    dilate = max(5, min(h, w) // 100)
    out = cv2.dilate(
        out.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)),
        iterations=1,
    ).astype(bool)
    return out


def pca_axis(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return None
    coords = np.column_stack([xs, ys]).astype(np.float32)
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    if cov.shape != (2, 2) or np.isnan(cov).any():
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argsort(eigvals)[-1]]
    proj = centered @ principal
    p_min, p_max = float(proj.min()), float(proj.max())
    if p_max - p_min < 1e-3:
        return None
    angle = math.degrees(math.atan2(abs(principal[1]), abs(principal[0])))
    return coords, mean, principal, p_min, p_max, angle


def classify_side(rgb: np.ndarray, min_conf: float) -> dict:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = fish_mask(rgb)
    if mask is None:
        return {"head_dir": "unknown", "confidence": 0.0, "reason": "no_fish_mask", "angle": 0.0}

    axis = pca_axis(mask)
    if axis is None:
        return {"head_dir": "unknown", "confidence": 0.0, "reason": "bad_axis", "angle": 0.0}
    _coords, mean, principal, p_min, p_max, angle = axis
    if angle > 55:
        return {"head_dir": "unknown", "confidence": 0.0, "reason": "too_vertical", "angle": angle}

    candidates = []
    max_eye_area = max(600, int(mask.sum() * 0.035))
    for thresh in range(65, 136, 10):
        dark = ((gray < thresh) & mask).astype(np.uint8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
        for label in range(1, n):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 8 or area > max_eye_area:
                continue
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            aspect = max(bw, bh) / max(1, min(bw, bh))
            fill = area / max(1, bw * bh)
            if aspect > 3.2 or fill < 0.25:
                continue
            cx, cy = centroids[label]
            proj_eye = (np.array([cx, cy]) - mean) @ principal
            u_eye = (proj_eye - p_min) / (p_max - p_min)
            endness = abs(float(u_eye) - 0.5) * 2.0
            if endness < 0.48:
                continue
            score = area * endness * (0.5 + fill) / aspect
            candidates.append(
                {
                    "score": float(score),
                    "u": float(u_eye),
                    "endness": float(endness),
                    "cx": float(cx),
                    "cy": float(cy),
                    "area": area,
                    "aspect": float(aspect),
                    "fill": float(fill),
                }
            )

    if not candidates:
        return {"head_dir": "unknown", "confidence": 0.0, "reason": "no_eye_candidate", "angle": angle}

    candidates.sort(key=lambda row: row["score"], reverse=True)
    best = candidates[0]
    opposite = [c for c in candidates[1:8] if (c["u"] < 0.5) != (best["u"] < 0.5)]
    if opposite and opposite[0]["score"] > best["score"] * 0.75:
        return {
            "head_dir": "unknown",
            "confidence": round(best["endness"] * 0.5, 4),
            "reason": "conflicting_eye_candidates",
            "angle": angle,
            "eye": best,
        }

    end0 = mean + principal * p_min
    end1 = mean + principal * p_max
    head_at_end0 = best["u"] < 0.5
    head_end = end0 if head_at_end0 else end1
    tail_end = end1 if head_at_end0 else end0
    head_dir = "left" if head_end[0] < tail_end[0] else "right"
    confidence = min(1.0, best["endness"])
    if confidence < min_conf:
        return {
            "head_dir": "unknown",
            "confidence": round(confidence, 4),
            "reason": "low_head_conf",
            "angle": angle,
            "eye": best,
        }

    return {
        "head_dir": head_dir,
        "confidence": round(confidence, 4),
        "reason": "eye_candidate",
        "angle": angle,
        "eye": best,
    }


def cam_from_head(head_dir: str, flip_map: bool) -> str | None:
    if head_dir == "unknown":
        return None
    if head_dir == "left":
        return "c2" if flip_map else "c1"
    return "c1" if flip_map else "c2"


def move_or_copy_pair(old: str, new: str, raw_dir: Path, enh_dir: Path, backup_dir: Path | None, apply: bool) -> None:
    if not apply:
        return
    for label, folder in (("raw", raw_dir), ("enhanced", enh_dir)):
        src = folder / old
        if not src.exists():
            continue
        if backup_dir is not None:
            dst_backup = backup_dir / label / old
            dst_backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_backup)
        if new != old:
            src.rename(folder / new)


def main() -> int:
    parser = argparse.ArgumentParser(description="Robust c1/c2 side labeling for video query frames.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--enhanced-dir", type=Path, default=DEFAULT_ENH)
    parser.add_argument("--decision-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--min-head-conf", type=float, default=0.35)
    parser.add_argument("--flip-map", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    decision_dir = args.decision_dir or args.raw_dir
    files = sorted([p for p in decision_dir.glob("*.png") if NAME_RE.match(p.name)], key=natural_key)
    if not files:
        print(f"[WARN] no matching PNG files in {decision_dir}")
        return 0

    backup_sub = None
    if args.apply:
        backup_sub = args.backup_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_sub.mkdir(parents=True, exist_ok=True)
    else:
        print("[DRY-RUN] no files will be renamed")

    rows = []
    stats = Counter()
    per_fish = defaultdict(Counter)
    for path in files:
        m = NAME_RE.match(path.name)
        if not m:
            continue
        fish_id, old_cam_num, seg, frame = m.groups()
        rgb = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        result = classify_side(rgb, args.min_head_conf)
        new_cam = cam_from_head(result["head_dir"], args.flip_map)
        if new_cam is None:
            new_name = f"{fish_id}_s{seg}_{frame}.png"
            stats["unknown"] += 1
            per_fish[fish_id]["s_only"] += 1
        else:
            new_name = f"{fish_id}_{new_cam}s{seg}_{frame}.png"
            stats[new_cam] += 1
            per_fish[fish_id][new_cam] += 1

        old_cam = f"c{old_cam_num}" if old_cam_num else None
        rows.append(
            {
                "old_file": path.name,
                "new_file": new_name,
                "fish_id": fish_id,
                "old_cam": old_cam,
                "new_cam": new_cam,
                "head_dir": result["head_dir"],
                "confidence": result["confidence"],
                "angle_deg": round(float(result["angle"]), 2),
                "reason": result["reason"],
                "eye": result.get("eye"),
            }
        )
        move_or_copy_pair(path.name, new_name, args.raw_dir, args.enhanced_dir, backup_sub, args.apply)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[TOTAL] {len(rows)}")
    print(f"[c1] {stats['c1']}")
    print(f"[c2] {stats['c2']}")
    print(f"[s-only/unknown] {stats['unknown']}")
    print("[PER FISH]")
    for fish_id in sorted(per_fish):
        c = per_fish[fish_id]
        missing = "" if c["c1"] and c["c2"] else "  <-- missing side"
        print(f"  {fish_id}: c1={c['c1']:2d}, c2={c['c2']:2d}, s_only={c['s_only']:2d}{missing}")
    print(f"[REPORT] {args.report}")
    if args.apply:
        print(f"[BACKUP] {backup_sub}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
