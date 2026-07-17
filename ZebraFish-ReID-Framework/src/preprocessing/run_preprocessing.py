"""
run_preprocessing.py — 预处理主流程（方案2）

对 bounding_box_train 中的鱼体裁剪图：
  1. 质量筛选（模糊 / 鱼体完整性 / 侧面优先）—— 丢弃低质量帧，提升后续识别准确率
  2. CLAHE 对比度增强 —— 突出条纹等身份特征
  3. 输出增强后 PNG（保持原始分辨率，不做尺寸缩放/不做归一化）
     —— 尺寸缩放与归一化由 reid 模块在推理时严格复刻 TransReID 变换完成

输出：
  data/processed/enhanced/*.png        增强图（与原文件名一致）
  data/processed/quality_report.jsonl  逐张质检流水
  configs/quality_thresholds.json      自适应模糊阈值与统计

用法：
  python src/preprocessing/run_preprocessing.py --limit 50 --dry-run
  python src/preprocessing/run_preprocessing.py
"""
import os
import argparse
import json
import cv2
import numpy as np
from PIL import Image

from quality import load_bgr, analyze_blur_threshold, is_acceptable, side_view_weight
from enhance import enhance_clahe

_FRAMEWORK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DEFAULT = r"D:/YUANRUXI0124/2026论文/（新）单条鱼数据集/market1501_video/market1501_video/bounding_box_train"
OUT_DIR = os.path.join(_FRAMEWORK, "data", "processed", "enhanced")
THRESH_PATH = os.path.join(_FRAMEWORK, "configs", "quality_thresholds.json")
REPORT_PATH = os.path.join(_FRAMEWORK, "data", "processed", "quality_report.jsonl")


def ensure_threshold(src_dir, force=False):
    if (not force) and os.path.exists(THRESH_PATH):
        with open(THRESH_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d["blur_thresh"], d
    blur_thresh, stats = analyze_blur_threshold(src_dir, quantile=0.10)
    with open(THRESH_PATH, "w", encoding="utf-8") as f:
        json.dump({"blur_thresh": blur_thresh, **stats}, f, indent=2, ensure_ascii=False)
    print(f"[阈值] 自适应模糊阈值 = {blur_thresh:.2f}（基于 {stats['n']} 张，p10）")
    return blur_thresh, stats


def process_one(fname, src_dir, out_dir, blur_thresh, clip_limit, dry_run, no_enhance):
    src = os.path.join(src_dir, fname)
    try:
        bgr = load_bgr(src)
    except Exception as e:
        return {"file": fname, "status": "skip_read", "reason": str(e)[:60]}

    ok, reason, info = is_acceptable(bgr, blur_thresh)
    meta = {"fish_id": fname.split("_")[0], "cam": fname.split("_")[1]}
    info["side_weight"] = side_view_weight(meta)
    info["reason"] = reason

    if not ok:
        return {"file": fname, "status": "rejected", **info}

    if not dry_run:
        if no_enhance:
            enhanced = bgr
        else:
            enhanced = enhance_clahe(bgr, clip_limit=clip_limit)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(os.path.join(out_dir, fname))
    return {"file": fname, "status": "accepted", **info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    ap.add_argument("--no-enhance", action="store_true", help="跳过 CLAHE 增强")
    ap.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clipLimit")
    ap.add_argument("--force-threshold", action="store_true", help="重新计算模糊阈值")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    blur_thresh, _ = ensure_threshold(args.src, force=args.force_threshold)

    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith((".png", ".jpg")))
    if args.limit:
        files = files[:args.limit]

    accepted = rejected = skipped = 0
    with open(REPORT_PATH, "w", encoding="utf-8") as rf:
        for fname in files:
            r = process_one(fname, args.src, args.out, blur_thresh,
                            args.clip_limit, args.dry_run, args.no_enhance)
            rf.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["status"] == "accepted":
                accepted += 1
            elif r["status"] == "rejected":
                rejected += 1
            else:
                skipped += 1

    total = accepted + rejected + skipped
    print(f"\n[完成] 共 {total} 张 | 接受 {accepted} | 拒绝 {rejected} | 跳过 {skipped}")
    if not args.dry_run:
        print(f"  增强图目录: {args.out}")
    print(f"  质检报告: {REPORT_PATH}")


if __name__ == "__main__":
    main()
