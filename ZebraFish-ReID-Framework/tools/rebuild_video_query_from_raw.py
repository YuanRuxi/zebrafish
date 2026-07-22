#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild clean video query folders from the unmirrored raw crops.

The current raw directory is treated as the source of truth for geometry. Output
files are reset to s-only names so a side-labeling pass can start fresh without
being biased by previous c1/c2 labels.
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image


FRAMEWORK = Path(__file__).resolve().parents[1]
DEFAULT_SRC = FRAMEWORK / "database" / "video_queries" / "raw"
DEFAULT_OUT_RAW = FRAMEWORK / "database" / "video_queries" / "relabel_work" / "raw_sonly"
DEFAULT_OUT_ENH = FRAMEWORK / "database" / "video_queries" / "relabel_work" / "enhanced_sonly"
NAME_RE = re.compile(r"^(\d{4})_(?:c[12])?s(\d)_(\d{4})\.png$", re.I)


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)


def enhance_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def clear_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild s-only video query folders from raw crops.")
    parser.add_argument("--src-raw", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out-raw", type=Path, default=DEFAULT_OUT_RAW)
    parser.add_argument("--out-enhanced", type=Path, default=DEFAULT_OUT_ENH)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    files_by_fish: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(args.src_raw.glob("*.png"), key=natural_key):
        m = NAME_RE.match(path.name)
        if not m:
            continue
        files_by_fish[m.group(1)].append(path)

    if not files_by_fish:
        print(f"[WARN] no matching PNG files in {args.src_raw}")
        return 0

    if not args.keep:
        clear_dir(args.out_raw)
        clear_dir(args.out_enhanced)
    else:
        args.out_raw.mkdir(parents=True, exist_ok=True)
        args.out_enhanced.mkdir(parents=True, exist_ok=True)

    total = 0
    for fish_id in sorted(files_by_fish):
        for idx, src in enumerate(files_by_fish[fish_id], 1):
            name = f"{fish_id}_s1_{idx:04d}.png"
            shutil.copy2(src, args.out_raw / name)
            bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if bgr is None:
                shutil.copy2(src, args.out_enhanced / name)
            else:
                enh = enhance_clahe(bgr)
                Image.fromarray(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)).save(args.out_enhanced / name)
            total += 1
        print(f"  {fish_id}: {len(files_by_fish[fish_id])} frame(s)")

    print(f"[DONE] rebuilt {total} frame(s)")
    print(f"[RAW] {args.out_raw}")
    print(f"[ENH] {args.out_enhanced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
