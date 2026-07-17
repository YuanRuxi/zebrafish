"""
quality.py — 帧质量评估（方案2：仅做筛选，不改尺寸/不归一化）

基于文献：
  - Nature 2025 斑马鱼论文：丢弃约 10% 低质量帧（模糊/异常体位）
  - AutoFish (NLDL 2026)：视角不一致性是 Re-ID 最大破坏因素
    （本数据集为纯侧面视频，c1/c2 均侧面，故侧面权重恒为 1.0）

指标：
  - blur_score : 灰度图 Laplacian 方差（越大越清晰）
  - fish_ratio : 非近黑像素占比（检测裁剪后鱼体应占满画面）
  - side_weight: 侧面权重（本数据恒 1.0，预留扩展正面/背面）
"""
import os
import json
import numpy as np
import cv2
from PIL import Image


def load_bgr(path):
    """用 PIL 读取（绕开本环境 OpenCV 的 PNG 解码 bug），返回 BGR numpy 数组。"""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def compute_blur_score(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_fish_ratio(bgr, black_thresh=30):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray > black_thresh) / (gray.shape[0] * gray.shape[1]))


def side_view_weight(fish_id_meta):
    """本数据集为纯侧面，权重恒 1.0；预留接口用于正面/背面降权。"""
    return 1.0


def analyze_blur_threshold(src_dir, quantile=0.10, sample=None):
    """
    扫描数据集，用分位数自适应确定模糊阈值（默认丢弃最模糊的 10%）。
    返回 (blur_thresh, 统计dict)
    """
    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith((".png", ".jpg")))
    if sample:
        files = files[:sample]
    scores = []
    for f in files:
        try:
            bgr = load_bgr(os.path.join(src_dir, f))
            scores.append(compute_blur_score(bgr))
        except Exception:
            continue
    scores = np.array(scores)
    blur_thresh = float(np.quantile(scores, quantile)) if len(scores) else 0.0
    stats = {
        "n": int(len(scores)),
        "blur_min": float(scores.min()) if len(scores) else 0.0,
        "blur_p10": float(np.quantile(scores, 0.10)) if len(scores) else 0.0,
        "blur_median": float(np.median(scores)) if len(scores) else 0.0,
        "blur_max": float(scores.max()) if len(scores) else 0.0,
        "quantile": quantile,
        "blur_thresh": blur_thresh,
    }
    return blur_thresh, stats


def is_acceptable(bgr, blur_thresh, min_fish_ratio=0.30):
    """返回 (ok: bool, reason: str, info: dict)。"""
    blur = compute_blur_score(bgr)
    ratio = compute_fish_ratio(bgr)
    info = {"blur_score": round(blur, 2), "fish_ratio": round(ratio, 3)}
    if ratio < min_fish_ratio:
        return False, "low_fish_ratio", info
    if blur < blur_thresh:
        return False, "blur", info
    return True, "ok", info
