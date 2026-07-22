#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalize imported zebrafish photos for ReID feature extraction.

Input:
    database/photos/NNNN_cXs1_ZZZZ.png

Output:
    database/photos_processed/NNNN_cXs1_ZZZZ.png

Processing:
  - segment the fish body from the bright tank background
  - rotate the long body axis to horizontal
  - crop tightly with a conservative margin, preserving the whole fish
  - flip horizontally when the detected head is on the right
  - optionally apply mild CLAHE enhancement

The script never overwrites source photos.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


FRAMEWORK = Path(__file__).resolve().parents[1]
SRC_DEFAULT = FRAMEWORK / "database" / "photos"
DST_DEFAULT = FRAMEWORK / "database" / "photos_processed"
REJECT_DEFAULT = FRAMEWORK / "database" / "photos_rejected"
REPORT_DEFAULT = FRAMEWORK / "database" / "photos_processed_report.jsonl"
YOLO_DEFAULT = FRAMEWORK / "models" / "yolov8_zebrafish.pt"

NAME_RE = re.compile(r"^\d{4}_c[12]s\d_\d{4}\.(png|jpg|jpeg|bmp|tif|tiff|webp)$", re.I)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)


def estimate_fish_mask(rgb: np.ndarray) -> tuple[np.ndarray, str]:
    """Return a conservative fish mask; prefer keeping extra fish pixels over trimming."""
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    border = np.zeros((h, w), dtype=bool)
    bw = max(8, min(h, w) // 30)
    border[:bw, :] = True
    border[-bw:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True
    bg_gray = float(np.median(gray[border]))
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    chroma = np.linalg.norm(lab[:, :, 1:3].astype(np.float32) - np.array([128, 128]), axis=2)

    # Tank walls and water marks are often different from the border color, but
    # they are still low-saturation. Fish body/fins/stripes carry stronger color
    # or localized dark texture, so prefer chroma/saturation cues here.
    mask = (
        ((sat > 35) & (val < 252))
        | ((chroma > 18) & (val < 250))
        | ((gray < max(95, bg_gray - 55)) & (sat > 14) & (val < 235))
    )

    # Ignore obvious border objects; the fish should not touch the image boundary.
    mask[:bw, :] = False
    mask[-bw:, :] = False
    mask[:, :bw] = False
    mask[:, -bw:] = False

    mask_u8 = (mask.astype(np.uint8) * 255)
    small = max(3, min(h, w) // 280)
    close = max(9, min(h, w) // 85)
    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small, small)),
        iterations=1,
    )
    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)),
        iterations=2,
    )

    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    if n <= 1:
        return np.ones((h, w), dtype=bool), "fallback_full_image"

    best_label = None
    best_score = -1.0
    img_area = h * w
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < img_area * 0.0002 or area > img_area * 0.20:
            continue
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        cw = stats[label, cv2.CC_STAT_WIDTH]
        ch = stats[label, cv2.CC_STAT_HEIGHT]
        aspect = max(cw, ch) / max(1, min(cw, ch))
        fill = area / max(1, cw * ch)
        # Water lines/tank rims are extremely long and thin. Zebrafish are long,
        # but the full body plus fins still has meaningful height.
        if aspect > 18.0:
            continue
        if cw > w * 0.45 and ch < h * 0.10:
            continue
        if fill < 0.015:
            continue
        cx = (x + cw / 2) / w
        cy = (y + ch / 2) / h
        centered = 1.0 - min(1.0, math.hypot(cx - 0.5, cy - 0.5))
        comp = labels == label
        dark = ((gray < 70) & comp).astype(np.uint8)
        dn, dlabels, dstats, _ = cv2.connectedComponentsWithStats(dark, 8)
        eye_score = 0.0
        for dlabel in range(1, dn):
            da = int(dstats[dlabel, cv2.CC_STAT_AREA])
            if da < 8 or da > max(3000, area * 0.08):
                continue
            dbw = dstats[dlabel, cv2.CC_STAT_WIDTH]
            dbh = dstats[dlabel, cv2.CC_STAT_HEIGHT]
            daspect = max(dbw, dbh) / max(1, min(dbw, dbh))
            if daspect <= 3.5:
                eye_score = max(eye_score, da / daspect)
        sat_mean = float(np.mean(sat[comp])) if area else 0.0
        score = area * min(aspect, 10.0) * (0.5 + centered) * (1.0 + sat_mean / 80.0)
        if eye_score > 0:
            score *= 4.0 + min(4.0, eye_score / 20.0)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        reason = "largest_component"
    else:
        reason = "scored_component"

    fish = labels == best_label
    # Dilate slightly so fins and transparent edges are not clipped.
    dilate = max(5, min(h, w) // 110)
    fish = cv2.dilate(
        fish.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)),
        iterations=1,
    ).astype(bool)
    return fish, reason


def pca_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return 0.0
    coords = np.column_stack([xs, ys]).astype(np.float32)
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argsort(eigvals)[-1]]
    angle = math.degrees(math.atan2(principal[1], principal[0]))
    # Rotating by -angle makes the principal axis horizontal.
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return float(angle)


def find_eye_center(rgb: np.ndarray, mask: np.ndarray) -> tuple[tuple[float, float] | None, float]:
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return None, 0.0
    x_min, x_max = int(xs.min()), int(xs.max())
    width = max(1, x_max - x_min + 1)
    gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    dark = ((gray < 70) & mask).astype(np.uint8)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    best = None
    best_score = -1.0
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 8 or area > max(5000, width * h * 0.012):
            continue
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 4.0:
            continue
        cx, cy = centroids[label]
        endness = abs((cx - x_min) / width - 0.5) * 2.0
        if endness < 0.20:
            continue
        score = area * endness / aspect
        if score > best_score:
            best_score = score
            best = (float(cx), float(cy))
    if best is None:
        return None, 0.0
    return best, float(best_score)


def body_axis_angle(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, str, float]:
    """Estimate rotation angle from head-eye to tail; fallback to PCA."""
    eye, eye_score = find_eye_center(rgb, mask)
    if eye is None:
        return pca_angle(mask), "pca", 0.0
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return 0.0, "empty_mask", 0.0
    ex, ey = eye
    dist2 = (xs.astype(np.float32) - ex) ** 2 + (ys.astype(np.float32) - ey) ** 2
    # Average the farthest 1% of mask pixels; using one pixel is too noisy at fin edges.
    k = max(10, int(len(dist2) * 0.01))
    far_idx = np.argpartition(dist2, -k)[-k:]
    tx = float(np.mean(xs[far_idx]))
    ty = float(np.mean(ys[far_idx]))
    angle = math.degrees(math.atan2(ty - ey, tx - ex))
    return float(angle), "eye_to_tail", eye_score


def rotate_image_and_mask(rgb: np.ndarray, mask: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, angle=angle, scale=1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    mat[0, 2] += new_w / 2 - center[0]
    mat[1, 2] += new_h / 2 - center[1]
    rotated_rgb = cv2.warpAffine(
        rgb,
        mat,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    rotated_mask = cv2.warpAffine(
        mask.astype(np.uint8) * 255,
        mat,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    return rotated_rgb, rotated_mask


def crop_to_mask(rgb: np.ndarray, mask: np.ndarray, margin_ratio: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = rgb.shape[:2]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return rgb, (0, 0, w, h)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw = x2 - x1
    bh = y2 - y1
    mx = max(16, int(bw * margin_ratio))
    my = max(16, int(bh * margin_ratio * 1.4))
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    return rgb[y1:y2, x1:x2], (x1, y1, x2, y2)


def crop_with_yolo(
    rgb: np.ndarray,
    model,
    imgsz: int,
    conf: float,
    margin_ratio: float,
) -> tuple[np.ndarray, dict]:
    h, w = rgb.shape[:2]
    result = model.predict(source=rgb, imgsz=imgsz, conf=conf, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return rgb, {"used": False, "reason": "no_detection"}

    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = int(np.argmax(scores * np.sqrt(np.maximum(areas, 1.0))))
    x1, y1, x2, y2 = xyxy[idx]
    bw = x2 - x1
    bh = y2 - y1
    mx = max(16, int(bw * margin_ratio))
    my = max(16, int(bh * margin_ratio * 1.4))
    ix1 = max(0, int(math.floor(x1)) - mx)
    iy1 = max(0, int(math.floor(y1)) - my)
    ix2 = min(w, int(math.ceil(x2)) + mx)
    iy2 = min(h, int(math.ceil(y2)) + my)
    return rgb[iy1:iy2, ix1:ix2], {
        "used": True,
        "conf": float(scores[idx]),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "crop": [ix1, iy1, ix2, iy2],
    }


def find_global_eye(rgb: np.ndarray) -> tuple[tuple[float, float] | None, float]:
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    chroma = np.linalg.norm(lab[:, :, 1:3].astype(np.float32) - np.array([128, 128]), axis=2)

    dark = (gray < 58).astype(np.uint8) * 255
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    best = None
    best_score = -1.0
    border = max(20, min(h, w) // 40)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 12 or area > max(4500, h * w * 0.002):
            continue
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        if x < border or y < border or x + bw > w - border or y + bh > h - border:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 3.0:
            continue
        cx, cy = centroids[label]
        r = max(80, int(max(bw, bh) * 5))
        x1, y1 = max(0, int(cx) - r), max(0, int(cy) - r)
        x2, y2 = min(w, int(cx) + r), min(h, int(cy) + r)
        local_sat = hsv[y1:y2, x1:x2, 1]
        local_chroma = chroma[y1:y2, x1:x2]
        color_ratio = float(((local_sat > 25) | (local_chroma > 14)).mean())
        rr = max(12, int(max(bw, bh) * 2.4))
        rx1, ry1 = max(0, int(cx) - rr), max(0, int(cy) - rr)
        rx2, ry2 = min(w, int(cx) + rr + 1), min(h, int(cy) + rr + 1)
        patch_gray = gray[ry1:ry2, rx1:rx2]
        yy, xx = np.mgrid[ry1:ry2, rx1:rx2]
        dist = np.sqrt((xx.astype(np.float32) - cx) ** 2 + (yy.astype(np.float32) - cy) ** 2)
        ring = (dist > max(bw, bh) * 0.65) & (dist < rr)
        bright_ring = float(((patch_gray > 145) & ring).sum() / max(1, ring.sum()))
        # The eye sits in a colored head region; tiny dirt specks on the tank do not.
        score = area / aspect * (1.0 + 6.0 * color_ratio) * (0.25 + 8.0 * bright_ring)
        if score > best_score:
            best_score = score
            best = (float(cx), float(cy))
    if best is None:
        return None, 0.0
    return best, float(best_score)


def crop_with_eye_guidance(rgb: np.ndarray, margin_ratio: float) -> tuple[np.ndarray, dict]:
    h, w = rgb.shape[:2]
    eye, eye_score = find_global_eye(rgb)
    if eye is None:
        return rgb, {"used": False, "reason": "no_eye"}

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    chroma = np.linalg.norm(lab[:, :, 1:3].astype(np.float32) - np.array([128, 128]), axis=2)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    ex, ey = eye

    candidate = (
        ((sat > 28) & (val < 252))
        | ((chroma > 16) & (val < 252))
        | ((gray < 92) & (sat > 10) & (val < 240))
    )
    border = max(16, min(h, w) // 45)
    candidate[:border, :] = False
    candidate[-border:, :] = False
    candidate[:, :border] = False
    candidate[:, -border:] = False
    yy, xx = np.where(candidate)
    if len(xx) < 50:
        return rgb, {"used": False, "reason": "weak_body_pixels", "eye": [ex, ey], "score": eye_score}

    # Keep plausible fish-body pixels around the eye's horizontal neighborhood.
    band = np.abs(yy.astype(np.float32) - ey) < max(260, h * 0.34)
    xx2, yy2 = xx[band], yy[band]
    if len(xx2) < 50:
        xx2, yy2 = xx, yy
    dist2 = (xx2.astype(np.float32) - ex) ** 2 + (yy2.astype(np.float32) - ey) ** 2
    k = max(20, int(len(dist2) * 0.01))
    far = np.argpartition(dist2, -k)[-k:]
    tx = float(np.mean(xx2[far]))
    ty = float(np.mean(yy2[far]))

    vx, vy = tx - ex, ty - ey
    length = max(1.0, math.hypot(vx, vy))
    px = xx.astype(np.float32) - ex
    py = yy.astype(np.float32) - ey
    proj = (px * vx + py * vy) / length
    perp = np.abs(px * vy - py * vx) / length
    corridor = (proj > -0.22 * length) & (proj < 1.22 * length) & (
        perp < max(140, length * 0.20)
    )
    cx = xx[corridor]
    cy = yy[corridor]
    if len(cx) < 50:
        cx, cy = xx2, yy2
    all_x = np.concatenate([cx, np.array([ex, tx])])
    all_y = np.concatenate([cy, np.array([ey, ty])])
    x1, x2 = float(all_x.min()), float(all_x.max())
    y1, y2 = float(all_y.min()), float(all_y.max())
    bw = x2 - x1
    bh = y2 - y1
    mx = max(80, int(bw * margin_ratio))
    my = max(80, int(max(bh, bw * 0.18) * margin_ratio * 1.2))
    ix1 = max(0, int(math.floor(x1)) - mx)
    iy1 = max(0, int(math.floor(y1)) - my)
    ix2 = min(w, int(math.ceil(x2)) + mx)
    iy2 = min(h, int(math.ceil(y2)) + my)
    return rgb[iy1:iy2, ix1:ix2], {
        "used": True,
        "eye": [ex, ey],
        "tail": [tx, ty],
        "score": eye_score,
        "crop": [ix1, iy1, ix2, iy2],
    }


def head_is_right(rgb: np.ndarray, mask: np.ndarray) -> tuple[bool, float, str]:
    """Estimate whether fish head is on the right after horizontal alignment."""
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return False, 0.0, "no_mask"

    x_min, x_max = int(xs.min()), int(xs.max())
    width = max(1, x_max - x_min + 1)
    gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

    # Primary cue: the eye is a compact, very dark component near one end.
    dark = ((gray < 65) & mask).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    best = None
    best_score = -1.0
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 8 or area > max(4000, width * h * 0.01):
            continue
        bx = stats[label, cv2.CC_STAT_LEFT]
        by = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 4.0:
            continue
        cx, cy = centroids[label]
        endness = abs((cx - x_min) / width - 0.5) * 2.0
        if endness < 0.25:
            continue
        score = area * endness / aspect
        if score > best_score:
            best_score = score
            best = (float(cx), float(cy), area)
    if best is not None:
        eye_x = best[0]
        confidence = min(1.0, abs((eye_x - x_min) / width - 0.5) * 2.0)
        return eye_x > (x_min + x_max) / 2, float(confidence), "eye_blob"

    left = mask[:, x_min : x_min + max(1, int(width * 0.28))]
    right = mask[:, x_max - max(1, int(width * 0.28)) + 1 : x_max + 1]
    left_gray = gray[:, x_min : x_min + left.shape[1]]
    right_gray = gray[:, x_max - right.shape[1] + 1 : x_max + 1]

    # Fallback cue: head is usually thicker and darker than the tail end.
    left_dark = float(np.mean((left_gray < 90) & left)) if left.any() else 0.0
    right_dark = float(np.mean((right_gray < 90) & right)) if right.any() else 0.0

    col_counts = mask.sum(axis=0).astype(np.float32)
    left_thick = float(np.mean(col_counts[x_min : x_min + max(1, int(width * 0.30))]))
    right_thick = float(np.mean(col_counts[x_max - max(1, int(width * 0.30)) + 1 : x_max + 1]))

    left_score = left_dark * 2.0 + left_thick / max(1.0, h)
    right_score = right_dark * 2.0 + right_thick / max(1.0, h)
    confidence = abs(right_score - left_score)
    return right_score > left_score, float(confidence), "darkness_thickness"


def enhance_clahe(rgb: np.ndarray, clip_limit: float) -> np.ndarray:
    if clip_limit <= 0:
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.merge([l2, a, b])
    return cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)


def validate_processed(rgb: np.ndarray) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    h, w = rgb.shape[:2]
    if w < h * 1.15:
        reasons.append("not_horizontal")
    mask, _reason = estimate_fish_mask(rgb)
    ys, xs = np.where(mask)
    if len(xs) < 80:
        reasons.append("weak_mask")
        return False, reasons
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw = x2 - x1
    bh = y2 - y1
    edge_tol = max(4, min(h, w) // 160)
    if x1 <= edge_tol or y1 <= edge_tol or x2 >= w - edge_tol or y2 >= h - edge_tol:
        reasons.append("fish_touches_crop_edge")
    eye, eye_score = find_eye_center(rgb, mask)
    if eye is None or eye_score < 8:
        reasons.append("no_reliable_eye")
    return not reasons, reasons


def process_one(
    path: Path,
    dst: Path,
    margin_ratio: float,
    clip_limit: float,
    overwrite: bool,
    reject_dir: Path,
    yolo_model=None,
    yolo_imgsz: int = 1920,
    yolo_conf: float = 0.25,
) -> dict:
    if dst.exists() and not overwrite:
        return {"file": path.name, "status": "exists", "out": str(dst)}
    rgb = load_rgb(path)
    work_rgb, eye_crop_info = crop_with_eye_guidance(rgb, margin_ratio=margin_ratio)
    if eye_crop_info.get("used"):
        yolo_info = {"used": False, "reason": "eye_guidance_preferred"}
    elif yolo_model is not None:
        work_rgb, yolo_info = crop_with_yolo(
            rgb, yolo_model, imgsz=yolo_imgsz, conf=yolo_conf, margin_ratio=margin_ratio
        )
    else:
        work_rgb, yolo_info = rgb, {"used": False, "reason": "disabled"}

    mask, mask_reason = estimate_fish_mask(work_rgb)
    angle, angle_reason, angle_conf = body_axis_angle(work_rgb, mask)
    rotated, rotated_mask = rotate_image_and_mask(work_rgb, mask, angle)
    flip, head_conf, head_reason = head_is_right(rotated, rotated_mask)
    if flip:
        rotated = np.ascontiguousarray(rotated[:, ::-1])
        rotated_mask = np.ascontiguousarray(rotated_mask[:, ::-1])
    cropped, crop_box = crop_to_mask(rotated, rotated_mask, margin_ratio)
    crop_fallback = False
    out_h, out_w = cropped.shape[:2]
    if min(out_w, out_h) < 250 or max(out_w, out_h) < 700:
        # When segmentation latches onto a small reflection or stripe fragment,
        # keep the full rotated detector crop. ReID tolerates extra background
        # much better than a missing head or tail.
        cropped = rotated
        crop_box = (0, 0, int(rotated.shape[1]), int(rotated.shape[0]))
        crop_fallback = True
    enhanced = enhance_clahe(cropped, clip_limit)
    valid, reject_reasons = validate_processed(enhanced)
    if crop_fallback:
        valid = False
        reject_reasons.append("crop_fallback")
    if not valid:
        reject_path = reject_dir / path.name
        reject_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(enhanced).save(reject_path, format="PNG")
        if dst.exists():
            dst.unlink()
        return {
            "file": path.name,
            "status": "rejected",
            "out": str(reject_path),
            "reject_reasons": reject_reasons,
            "mask_reason": mask_reason,
            "eye_crop": eye_crop_info,
            "yolo": yolo_info,
            "angle_deg": angle,
            "angle_reason": angle_reason,
            "angle_conf": angle_conf,
            "flipped": flip,
            "head_conf": head_conf,
            "head_reason": head_reason,
            "crop_box": crop_box,
            "crop_fallback": crop_fallback,
            "input_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "output_size": [int(enhanced.shape[1]), int(enhanced.shape[0])],
        }
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(enhanced).save(dst, format="PNG")
    return {
        "file": path.name,
        "status": "processed",
        "out": str(dst),
        "mask_reason": mask_reason,
        "eye_crop": eye_crop_info,
        "yolo": yolo_info,
        "angle_deg": angle,
        "angle_reason": angle_reason,
        "angle_conf": angle_conf,
        "flipped": flip,
        "head_conf": head_conf,
        "head_reason": head_reason,
        "crop_box": crop_box,
        "crop_fallback": crop_fallback,
        "input_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "output_size": [int(enhanced.shape[1]), int(enhanced.shape[0])],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize database/photos for ReID.")
    parser.add_argument("--src", type=Path, default=SRC_DEFAULT)
    parser.add_argument("--dst", type=Path, default=DST_DEFAULT)
    parser.add_argument("--reject-dir", type=Path, default=REJECT_DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--apply", action="store_true", help="Write processed images.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.45, help="Crop margin relative to fish bbox.")
    parser.add_argument("--clip-limit", type=float, default=1.5, help="CLAHE strength; 0 disables enhancement.")
    parser.add_argument("--yolo", type=Path, default=YOLO_DEFAULT, help="YOLO zebrafish detector; use 'none' to disable.")
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO detection confidence.")
    parser.add_argument("--imgsz", type=int, default=1920, help="YOLO inference image size.")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    if not src.exists():
        print(f"[ERROR] Source directory does not exist: {src}")
        return 1

    files = [
        p
        for p in sorted(src.iterdir(), key=natural_key)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and NAME_RE.match(p.name)
    ]
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"[ERROR] No named images found in {src}")
        return 1

    print(f"[SOURCE] {src}")
    print(f"[DEST]   {dst}")
    print(f"[PLAN]   {len(files)} images")
    print(f"[MODE]   {'apply' if args.apply else 'dry-run'}")
    print(f"[PARAM]  margin={args.margin}, clip_limit={args.clip_limit}")
    yolo_model = None
    yolo_arg = str(args.yolo)
    if yolo_arg.lower() != "none":
        if not args.yolo.exists():
            print(f"[ERROR] YOLO model not found: {args.yolo}")
            return 1
        if args.apply:
            from ultralytics import YOLO

            yolo_model = YOLO(str(args.yolo))
        print(f"[YOLO]   {args.yolo} conf={args.conf} imgsz={args.imgsz}")
    for p in files[:10]:
        print(f"  {p.name} -> {dst / p.name}")
    if len(files) > 10:
        print(f"  ... {len(files) - 10} more")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write processed images.")
        return 0

    args.report.parent.mkdir(parents=True, exist_ok=True)
    processed = skipped = failed = 0
    with open(args.report, "w", encoding="utf-8") as report:
        for p in files:
            try:
                row = process_one(
                    p,
                    dst / p.name,
                    margin_ratio=args.margin,
                    clip_limit=args.clip_limit,
                    overwrite=args.overwrite,
                    reject_dir=args.reject_dir.resolve(),
                    yolo_model=yolo_model,
                    yolo_imgsz=args.imgsz,
                    yolo_conf=args.conf,
                )
            except Exception as exc:
                row = {"file": p.name, "status": "failed", "reason": str(exc)}
                failed += 1
            else:
                if row["status"] == "processed":
                    processed += 1
                elif row["status"] == "rejected":
                    skipped += 1
                else:
                    skipped += 1
            report.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[DONE] processed={processed}, skipped={skipped}, failed={failed}")
    print(f"[REPORT] {args.report}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
