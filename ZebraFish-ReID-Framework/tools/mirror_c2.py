#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mirror_c2.py — 将斑马鱼数据集中 c2（右面）图片水平镜像。

背景：在 Re-ID 中，把右面（c2）鱼图水平镜像后，外观上接近左面（c1）视角，
可作为同一条鱼的左面增广数据，帮助模型学习"左右同鱼"。
固定 ReID 权重 transformer_20.pth 训练/验证时用的 c2 就是"已镜像、鱼头朝左"的，
因此原始视频流抽取的 c2（原始右面，鱼头朝向任意）在喂给 ReID 前必须先镜像。

两种用法：
  1) 旧数据集（默认，安全，绝不改源）：
       python tools/mirror_c2.py
     → 读 bounding_box_train，把 c2 镜像后写入独立的 bounding_box_train_c2_mirrored/
       安全护栏：src 不得等于 out，源目录永不改写。

  2) 视频流抽取结果（in-place，ReID 前处理）★本次小样用这个：
       python tools/mirror_c2.py --src data/processed/enhanced --inplace
     → 只在 enhanced/ 内把 c2 镜像为"鱼头朝左"（覆盖原 c2 文件）。
     ★ 重要：本命令【不会】动 raw/video_crops/ 里的 c2。
       raw 里的 c2 是你人工核对"右面检测是否正确"的原始依据，保持原样不动。
     ★ in-place 会覆盖 enhanced 中的源 c2 文件，仅用于"生成的裁剪产物"，不要用于珍贵原始数据。

  推荐完整流程（视频流）：
     video_extractor.py  (单类检测，输出 NNNN_s1_ZZZZ.png，不判左右)
        →  side_geometry_label.py --apply   (几何法判左右 → NNNN_cXs1_ZZZZ.png)
        →  mirror_c2.py --src data/processed/enhanced --inplace   (仅 enhanced 的 c2 变 head-left)
        →  pipeline.py --build --eval   (只读 enhanced/，不读 raw/)

注意：本批 PNG 用 OpenCV 在本机无法解码，统一用 PIL 读写。
"""
import os
import re
import argparse
from PIL import Image

NAME_RE = re.compile(r'^(\d{4})_c(\d)s(\d)_(\d{4})\.png$', re.IGNORECASE)

DEFAULT_SRC = r"D:\YUANRUXI0124\2026论文\（新）单条鱼数据集\market1501_video\market1501_video\bounding_box_train"
# 默认输出到源目录的同级文件夹，文件名保持不变
DEFAULT_OUT = r"D:\YUANRUXI0124\2026论文\（新）单条鱼数据集\market1501_video\market1501_video\bounding_box_train_c2_mirrored"


def mirror_one(src_path, inplace):
    """读图 -> 水平镜像 -> 就地覆盖(src_path) 或调用方已决定落点；本函数只做镜像+保存。"""
    img = Image.open(src_path)
    img_m = img.transpose(Image.FLIP_LEFT_RIGHT)
    if inplace:
        img_m.save(src_path)          # 覆盖源 c2 文件
    return img_m


def process_dir(src, out_dir, inplace, force, dry_run):
    """
    处理 src 目录下所有 c2 文件。
    inplace=True：镜像结果覆盖回 src 内原文件（out_dir 被忽略）。
    inplace=False：镜像结果写入 out_dir（文件名不变），已存在且非 force 则跳过。
    返回 (处理数, 跳过数)。
    """
    count = 0
    skipped = 0
    for fn in sorted(os.listdir(src)):
        m = NAME_RE.match(fn)
        if not m:
            continue
        if m.group(2) != "2":          # 只处理 c2
            continue
        src_path = os.path.join(src, fn)
        if inplace:
            out_path = src_path
        else:
            out_path = os.path.join(out_dir, fn)
            if os.path.exists(out_path) and not force:
                skipped += 1
                continue
        if dry_run:
            print(f"[dry-run] {fn}  ->  {'INPLACE' if inplace else out_path}")
        else:
            mirror_one(src_path, inplace)
        count += 1
    return count, skipped


def main():
    ap = argparse.ArgumentParser(description="镜像 c2 右面鱼图（水平翻转，存独立文件夹或 in-place）")
    ap.add_argument("--src", default=DEFAULT_SRC, help="源目录（c2 所在目录）")
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="镜像结果输出目录（仅非 in-place 模式使用，独立文件夹，文件名不变）")
    ap.add_argument("--inplace", action="store_true",
                    help="直接在原目录内把 c2 镜像覆盖（用于视频流抽取结果喂 ReID 前处理）；"
                         "只作用于 enhanced/ 内的 c2 文件，不会动 raw/video_crops 的原始 c2，"
                         "方便你人工核对右面检测是否正确。")
    ap.add_argument("--force", action="store_true",
                    help="若输出文件已存在则覆盖；默认跳过已存在文件（幂等）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要处理的文件，不写盘")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(f"源目录不存在: {args.src}")

    if args.inplace:
        print("[警告] in-place 模式：将直接覆盖源目录中的 c2 文件"
              "（仅用于 video_extractor 生成的裁剪产物，勿用于珍贵原始数据）。")
        src_abs = os.path.abspath(args.src)
        out_abs = src_abs
    else:
        src_abs = os.path.abspath(args.src)
        out_abs = os.path.abspath(args.out_dir)
        if src_abs == out_abs:
            raise SystemExit("安全拦截：输出目录不能与源目录相同，避免覆盖原始数据。")
        os.makedirs(out_abs, exist_ok=True)

    c1, s1 = process_dir(src_abs, out_abs, args.inplace, args.force, args.dry_run)
    print(f"\n源 c2 图片: 已处理 {c1} 张"
          f"{'（跳过已存在 ' + str(s1) + ' 张）' if (not args.inplace and s1) else ''}"
          f"  ({'dry-run' if args.dry_run else ('INPLACE @ ' + src_abs if args.inplace else '已写入 ' + out_abs)})")
    if args.inplace:
        print("[提示] 仅 enhanced/ 内的 c2 被镜像；raw/video_crops/ 的原始 c2 保持不动，可人工核对。")


if __name__ == "__main__":
    main()
