#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
side_geometry_label.py — 用几何朝向法（方法 A）判定鱼体左右，并为裁剪图标注 c1/c2。

核心假设（来自用户场景）：固定相机 + 鱼背朝上，单条鱼在视频帧里只能看到一侧面；
鱼掉头游动时 180° 会换面，因此"图像中鱼头朝左/右"与"看到哪一侧面"存在一一映射。

约定：
  - 头朝左  ->  c1（左面）
  - 头朝右  ->  c2（右面）

为什么需要本步：video_extractor 现在用【单类】YOLO 只检测鱼、不判左右，
  产出的文件名形如 NNNN_s1_ZZZZ.png（不含 c1/c2）。本脚本读取增强图，
  用几何朝向法判定每张图的左右，并重命名为 NNNN_c1s1_ZZZZ.png / NNNN_c2s1_ZZZZ.png。
  （若文件已带 c1/c2，本脚本仍会重新判定，发现判反则纠正——向后兼容。）

  ★ 两种「置信度」务必分清（这是最容易混淆的点）：
    ① video_extractor 的 conf_side = YOLO「框里是不是鱼」的把握。低于它的帧在【抽帧阶段】
       就已被丢弃，根本不会成为文件，也就不会进到本脚本。
    ② 本脚本 detect_head_direction 返回的 confidence = 几何法「头朝左还是右」的把握，
       与 ① 完全是两回事。它决定我们【敢不敢给这张图打 c1/c2】：把握低 → 不打，
       该帧保持 s-only（不参与 ReID 交叉视角）。

若你的相机/鱼缸方向相反，请加 --flip-map。

用法：
  # 1. 先预览看统计（不加 --apply 即为预览模式，只打印统计、不改名、不写报告）
  python tools/side_geometry_label.py
  python tools/side_geometry_label.py --min-head-conf 0.25   # 预览时也可调严格度

  # 2. 确认后再实际改名（会自动备份原文件到 data/processed/_side_geom_backup/）
  python tools/side_geometry_label.py --apply

  # 3. 如果你的缸方向相反，翻转映射
  python tools/side_geometry_label.py --apply --flip-map
"""
import os
import re
import argparse
import shutil
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

_FRAMEWORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENHANCED_DIR = os.path.join(_FRAMEWORK, "data", "processed", "enhanced")
RAW_DIR = os.path.join(_FRAMEWORK, "data", "raw", "video_crops")
BACKUP_DIR = os.path.join(_FRAMEWORK, "data", "processed", "_side_geom_backup")
REPORT_PATH = os.path.join(_FRAMEWORK, "data", "processed", "_side_geom_report.jsonl")
VIDEO_REPORT_PATH = os.path.join(_FRAMEWORK, "data", "processed", "video_crop_report.jsonl")
QUALITY_REPORT_PATH = os.path.join(_FRAMEWORK, "data", "processed", "quality_report.jsonl")

# 文件名正则（决定主循环走哪条分支）：
#  C_ONLY_RE：文件【已经】带 c1/c2（如上一轮 side_geometry_label 跑过、或人工标过）。
#             干净流程里 video_extractor 产出的是 s-only，正常不会出现；
#             此分支仅用于【重跑/纠正】：重新判头向，若与原有 c 相反则改名纠正。
#  S_ONLY_RE：尚无左右标记（video_extractor 单类抽帧的标准产物 NNNN_s1_ZZZZ.png）。
#             主循环绝大多数情况走这条：本步首次给它分配 c1/c2。
#  ★ s1 只是「全身段」代号，和左右(c1/c2)无关；c 是 side_geometry_label 才首次写进文件名。
C_ONLY_RE = re.compile(r'^(\d{4})_c([12])s(\d)_(\d{4})\.png$')
S_ONLY_RE = re.compile(r'^(\d{4})_s(\d)_(\d{4})\.png$')


def get_fish_mask(gray: np.ndarray) -> np.ndarray:
    """
    用 Otsu + 形态学闭运算 + 最大连通域提取鱼体前景 mask。
    同时尝试"鱼体偏暗"和"鱼体偏亮"两种假设，取最靠近画面中心、面积适中的连通域。
    """
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_mask = None
    best_score = 0.0

    for inv in (False, True):
        bw = cv2.bitwise_not(otsu) if inv else otsu
        # 形态学闭运算把鱼体（可能因条纹被分成多块）连成一个整体
        k = max(15, min(h, w) // 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        n, labels = cv2.connectedComponents(closed, connectivity=8)
        if n <= 1:
            continue
        for i in range(1, n):
            mask_i = (labels == i)
            area = int(mask_i.sum())
            total = h * w
            if area < total * 0.05 or area > total * 0.95:
                continue
            ys, xs = np.where(mask_i)
            cx = xs.mean() / w
            cy = ys.mean() / h
            centered = 1.0 - ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
            score = area * centered
            if score > best_score:
                best_score = score
                best_mask = mask_i

    if best_mask is not None:
        return best_mask
    # fallback：如果连通域失败，直接返回一个全为前景的 mask（几何法会标为不可靠）
    return np.ones((h, w), dtype=bool)


def detect_head_direction(gray: np.ndarray, mask: np.ndarray, min_head_conf: float = 0.25):
    """
    对鱼体 mask 做主成分分析，找到长轴；
    用"眼睛暗斑位置"判断头端：头部有大而黑的瞳孔，是图中最稳定的判别信号。
    若眼睛检测失败，回退到"沿轴质量分布偏度"法。
    返回：('left'|'right'|'unknown', confidence, angle_deg, reason)
    """
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return 'unknown', 0.0, 0.0, 'mask_too_small'

    coords = np.column_stack((xs, ys)).astype(np.float32)
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    if cov.size == 1 or np.isnan(cov).any():
        return 'unknown', 0.0, 0.0, 'cov_error'

    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    principal = eigvecs[:, idx[0]]  # 主方向 (dx, dy)

    # 投影到主轴
    proj = centered @ principal
    p_min, p_max = proj.min(), proj.max()
    if p_max - p_min < 1e-3:
        return 'unknown', 0.0, 0.0, 'degenerate_axis'

    u = (proj - p_min) / (p_max - p_min)  # 0 ~ 1, 0 在 end0, 1 在 end1
    end0 = mean + principal * p_min
    end1 = mean + principal * p_max

    # 长轴与水平轴夹角
    angle = np.arctan2(abs(principal[1]), abs(principal[0])) * 180.0 / np.pi
    if angle > 60.0:
        return 'unknown', 0.0, float(angle), 'too_vertical'

    # ---------- 主要方法：眼睛暗斑定位（保留瞳孔定位法，本版不改判断逻辑）----------
    # 瞳孔通常是图中最暗的紧凑区域。先找全局最暗像素，再取其所在连通域的质心作为眼睛中心。
    masked_gray = np.where(mask, gray, 255)
    min_val = int(masked_gray[mask].min())
    # 以最暗值为中心，取一个 tolerance 形成暗区
    dark = (masked_gray <= min_val + 25).astype(np.uint8) * 255
    min_pos = np.unravel_index(np.argmin(masked_gray), masked_gray.shape)
    head_dir = None
    confidence = 0.0
    reason = 'centroid_fallback'
    if dark.sum() > 0:
        n, labels = cv2.connectedComponents(dark, connectivity=8)
        eye_label = labels[min_pos]
        if eye_label > 0:
            ys_i, xs_i = np.where(labels == eye_label)
            if len(xs_i) >= 10:
                eye_x = float(xs_i.mean())
                eye_y = float(ys_i.mean())
                proj_eye = (np.array([eye_x, eye_y]) - mean) @ principal
                u_eye = (proj_eye - p_min) / (p_max - p_min)
                head_at_end0 = u_eye < 0.5
                head_end = end0 if head_at_end0 else end1
                tail_end = end1 if head_at_end0 else end0
                head_left = head_end[0] < tail_end[0]
                head_dir = 'left' if head_left else 'right'
                # confidence：眼睛越靠近某一端越可靠（0.5 表示正好在中间，无法判断头尾）
                confidence = min(1.0, abs(u_eye - 0.5) * 2.0)
                reason = 'eye_blob'

    # ---------- 回退：沿轴质量分布偏度（仅当眼睛暗斑法未能定位时）----------
    if head_dir is None:
        u_mean = float(u.mean())
        head_at_end0 = u_mean < 0.5
        head_end = end0 if head_at_end0 else end1
        tail_end = end1 if head_at_end0 else end0
        head_left = head_end[0] < tail_end[0]
        head_dir = 'left' if head_left else 'right'
        confidence = min(1.0, abs(u_mean - 0.5) * 2.0)
        reason = 'centroid_fallback'

    # ---------- 置信度门槛：几何法对「头朝哪边」没把握时，不打 c1/c2 ----------
    # 注意：此 confidence 是【几何法判头向】的把握，与 video_extractor 里
    # YOLO 的「是否是鱼」置信度（conf_side）是两个完全不同的量！
    # 低于阈值 → 返回 unknown → 主循环里该帧保持 s-only（不参与 ReID）。
    if confidence < min_head_conf:
        return 'unknown', float(confidence), float(angle), 'low_head_conf'

    return head_dir, float(confidence), float(angle), reason


def determine_new_cam(head_dir: str, flip_map: bool) -> str:
    """
    默认：头朝左 -> c1（左面），头朝右 -> c2（右面）。
    --flip-map 时反过来：头朝左 -> c2，头朝右 -> c1。
    """
    if head_dir == 'unknown':
        return None
    if head_dir == 'left':
        return 'c2' if flip_map else 'c1'
    return 'c1' if flip_map else 'c2'


def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def save_jsonl(path: str, rows: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="几何朝向法重新判定鱼体左右并改名")
    ap.add_argument("--enhanced-dir", default=ENHANCED_DIR, help="增强图目录")
    ap.add_argument("--raw-dir", default=RAW_DIR, help="原图裁剪目录")
    ap.add_argument("--backup-dir", default=BACKUP_DIR, help="备份目录")
    ap.add_argument("--report", default=REPORT_PATH, help="几何法报告路径")
    ap.add_argument("--flip-map", action="store_true", help="翻转头向->c1/c2的映射")
    ap.add_argument("--min-head-conf", type=float, default=0.25,
                    help="几何法判头向的最低置信度；低于此值的帧不打 c1/c2（保持 s-only，不参与 ReID）。"
                         "与 video_extractor 的 conf_side 是两回事")
    ap.add_argument("--apply", action="store_true", help="实际执行重命名（默认 dry-run）")
    ap.add_argument("--keep-report", action="store_true", help="保留已有报告，追加本次记录")
    args = ap.parse_args()

    enhanced_dir = Path(args.enhanced_dir)
    raw_dir = Path(args.raw_dir)
    backup_dir = Path(args.backup_dir)

    if not enhanced_dir.exists():
        print(f"[错误] 增强图目录不存在: {enhanced_dir}")
        return

    files = sorted([f for f in enhanced_dir.iterdir() if f.is_file() and f.name.lower().endswith(".png")])
    if not files:
        print(f"[警告] 目录为空: {enhanced_dir}")
        return

    # 备份
    if args.apply:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_sub = backup_dir / ts
        backup_sub.mkdir(parents=True, exist_ok=True)
        print(f"[备份] 原文件将备份到: {backup_sub}")
    else:
        backup_sub = None
        print("[DRY-RUN] 不会修改任何文件，只输出统计\n")

    # 加载已有报告用于同步更新
    video_rows = load_jsonl(VIDEO_REPORT_PATH)
    quality_rows = load_jsonl(QUALITY_REPORT_PATH)
    video_index = {r.get("file"): i for i, r in enumerate(video_rows) if r.get("file")}
    quality_index = {r.get("file"): i for i, r in enumerate(quality_rows) if r.get("file")}

    geom_report = []
    if args.keep_report and os.path.exists(args.report):
        geom_report = load_jsonl(args.report)

    stats = {'total': 0, 'head_left': 0, 'head_right': 0, 'unknown': 0,
             'assigned': 0, 'changed': 0, 'unchanged': 0}

    for fpath in files:
        fname = fpath.name
        m = C_ONLY_RE.match(fname)
        if m:
            fish_id, old_cam_num, seg, frame = m.groups()
            old_cam = f"c{old_cam_num}"
            had_cam = True
        elif S_ONLY_RE.match(fname):
            m = S_ONLY_RE.match(fname)
            fish_id, seg, frame = m.groups()
            old_cam = None
            had_cam = False
        else:
            print(f"[跳过] 文件名格式不匹配: {fname}")
            continue

        # 用增强图做几何判断（对比度高，mask 更稳）
        img = np.array(Image.open(fpath).convert('L'))
        mask = get_fish_mask(img)
        head_dir, conf, angle, reason = detect_head_direction(img, mask, args.min_head_conf)

        new_cam = determine_new_cam(head_dir, args.flip_map)
        if new_cam is None:
            # 无法判定左右
            stats['unknown'] += 1
            if had_cam:
                new_cam = old_cam           # 已带 c1/c2，保持原样
                changed = False
            else:
                # s-only 且无法判定：留作 s-only（不参与 ReID 交叉视角），仅记录
                entry = {
                    "old_file": fname, "new_file": fname,
                    "fish_id": fish_id, "old_cam": None, "new_cam": None,
                    "head_dir": head_dir, "confidence": round(conf, 4),
                    "angle_deg": round(angle, 2), "reason": reason,
                    "changed": False,
                }
                geom_report.append(entry)
                stats['total'] += 1
                continue
        else:
            if head_dir == 'left':
                stats['head_left'] += 1
            else:
                stats['head_right'] += 1
            changed = (had_cam and new_cam != old_cam)

        new_name = f"{fish_id}_{new_cam}s{seg}_{frame}.png"
        if had_cam and changed:
            stats['changed'] += 1
        elif not had_cam:
            stats['assigned'] += 1         # 从 s-only 赋予 c1/c2
        else:
            stats['unchanged'] += 1

        entry = {
            "old_file": fname,
            "new_file": new_name,
            "fish_id": fish_id,
            "old_cam": old_cam,
            "new_cam": new_cam,
            "head_dir": head_dir,
            "confidence": round(conf, 4),
            "angle_deg": round(angle, 2),
            "reason": reason,
            "changed": changed,
        }
        geom_report.append(entry)
        stats['total'] += 1

        # 实际改名
        if args.apply:
            # 先同步更新报告的 cam 字段（无论是否改名）
            if fname in video_index:
                video_rows[video_index[fname]]["cam"] = new_cam
            if new_name != fname:
                # 备份原文件
                shutil.copy2(fpath, backup_sub / fname)
                fpath.rename(fpath.parent / new_name)

                raw_path = raw_dir / fname
                if raw_path.exists():
                    shutil.copy2(raw_path, backup_sub / raw_path.name)
                    raw_path.rename(raw_dir / new_name)

                # 同步更新报告里的文件名
                if fname in video_index:
                    video_rows[video_index[fname]]["file"] = new_name
                if fname in quality_index:
                    quality_rows[quality_index[fname]]["file"] = new_name

    if args.apply:
        # 保存更新后的报告
        save_jsonl(VIDEO_REPORT_PATH, video_rows)
        if os.path.exists(QUALITY_REPORT_PATH) or quality_rows:
            save_jsonl(QUALITY_REPORT_PATH, quality_rows)

    save_jsonl(args.report, geom_report)

    print("\n" + "=" * 60)
    print("几何朝向法左右判别统计")
    print("=" * 60)
    print(f"  处理图片数: {stats['total']}")
    print(f"  头朝左 -> {'c2' if args.flip_map else 'c1'}: {stats['head_left']} 张")
    print(f"  头朝右 -> {'c1' if args.flip_map else 'c2'}: {stats['head_right']} 张")
    print(f"  新赋予 c1/c2 (原 s-only): {stats['assigned']} 张")
    print(f"  无法判定(含低置信/留 s-only/保原样): {stats['unknown']} 张")
    print(f"  纠正改名(原 c 判反): {stats['changed']} 张")
    print(f"  保持原样: {stats['unchanged']} 张")
    if not args.apply:
        print("\n[DRY-RUN] 未修改文件。要实际改名请重跑并加 --apply")
    else:
        print(f"\n[完成] 已备份原文件到: {backup_sub}")
        print(f"[报告] 详细对照表: {args.report}")


if __name__ == "__main__":
    main()
