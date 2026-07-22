#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import manually sorted zebrafish photos into database/photos.

Expected source layout:

    SOURCE/
      1/
        L/*.jpg|png|...
        R/*.jpg|png|...
      2/
        L/
        R/

The script also accepts nested side folders named L or R under each numeric fish
folder. Output files are converted to PNG and renamed to the project convention:

    NNNN_cXs1_ZZZZ.png

where c1 = left side (L), c2 = right side (R), and ZZZZ is a per-fish running
index across both sides.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
except ImportError:
    register_heif_opener = None
else:
    register_heif_opener()


FRAMEWORK = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path(r"C:\Users\JiangYao\Desktop\26_Medical\7.14斑马鱼照片+视频")
DEFAULT_DST = FRAMEWORK / "database" / "photos"

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".heif",
    ".livp",
}

SIDE_TO_CAM = {
    "L": "c1",
    "R": "c2",
}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def natural_key(path: Path) -> tuple:
    """Sort names in a stable human-ish order without locale assumptions."""
    import re

    parts = re.split(r"(\d+)", path.name.lower())
    key = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)


def fish_id_from_folder(folder: Path) -> str | None:
    digits = "".join(ch for ch in folder.name if ch.isdigit())
    if not digits:
        return None
    value = int(digits)
    if value <= 0:
        return None
    return f"{value:04d}"


def iter_fish_folders(src: Path) -> Iterable[tuple[str, Path]]:
    for child in sorted(src.iterdir(), key=natural_key):
        if not child.is_dir():
            continue
        fish_id = fish_id_from_folder(child)
        if fish_id is not None:
            yield fish_id, child


def find_side_dirs(fish_dir: Path) -> dict[str, list[Path]]:
    """Find L/R side directories directly or nested below one fish folder."""
    side_dirs: dict[str, list[Path]] = {"L": [], "R": []}
    for path in fish_dir.rglob("*"):
        if not path.is_dir():
            continue
        side = path.name.strip().upper()
        if side in side_dirs:
            side_dirs[side].append(path)
    for side in side_dirs:
        side_dirs[side] = sorted(set(side_dirs[side]), key=lambda p: p.as_posix().lower())
    return side_dirs


def collect_images(fish_dir: Path) -> dict[str, list[Path]]:
    side_dirs = find_side_dirs(fish_dir)
    by_side: dict[str, list[Path]] = {"L": [], "R": []}
    for side, dirs in side_dirs.items():
        seen: set[Path] = set()
        for side_dir in dirs:
            for img in sorted(side_dir.rglob("*"), key=natural_key):
                if is_image(img) and img not in seen:
                    by_side[side].append(img)
                    seen.add(img)
    return by_side


HEIC_EXTS = {".heic", ".heif"}
LIVP_EXTS = {".livp"}
LIVP_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def output_suffix(src: Path, heic_mode: str) -> str:
    if src.suffix.lower() in HEIC_EXTS and heic_mode == "copy":
        return src.suffix.lower()
    return ".png"


def _open_livp_still(src: Path) -> Image.Image:
    """Open the still image inside an Apple LIVP archive."""
    if not zipfile.is_zipfile(src):
        raise ValueError("LIVP file is not a readable zip archive")
    with zipfile.ZipFile(src, "r") as zf:
        candidates = []
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or name.startswith("__MACOSX/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix in LIVP_IMAGE_EXTS:
                candidates.append(info)
        if not candidates:
            raise ValueError("LIVP archive contains no supported still image")
        # Prefer the largest still image; thumbnails and sidecars are usually smaller.
        info = max(candidates, key=lambda item: item.file_size)
        data = zf.read(info)
    im = Image.open(io.BytesIO(data))
    im.load()
    return im


def save_image(src: Path, dst: Path, overwrite: bool, heic_mode: str) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in HEIC_EXTS and heic_mode == "copy":
        shutil.copy2(src, dst)
        return
    if src.suffix.lower() in HEIC_EXTS and heic_mode == "skip":
        return
    if src.suffix.lower() in LIVP_EXTS:
        im = _open_livp_still(src)
    else:
        im = Image.open(src)
    with im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        im.save(dst, format="PNG")


def build_plan(
    src: Path,
    dst: Path,
    heic_mode: str,
    fish_start: int | None = None,
    fish_end: int | None = None,
) -> list[tuple[Path, Path, str, str]]:
    plan: list[tuple[Path, Path, str, str]] = []
    counters: defaultdict[str, int] = defaultdict(int)
    for fish_id, fish_dir in iter_fish_folders(src):
        fish_no = int(fish_id)
        if fish_start is not None and fish_no < fish_start:
            continue
        if fish_end is not None and fish_no > fish_end:
            continue
        by_side = collect_images(fish_dir)
        for side in ("L", "R"):
            cam = SIDE_TO_CAM[side]
            for img in by_side[side]:
                counters[fish_id] += 1
                frame = f"{counters[fish_id]:04d}"
                out_name = f"{fish_id}_{cam}s1_{frame}{output_suffix(img, heic_mode)}"
                plan.append((img, dst / out_name, fish_id, side))
    return plan


def diagnose_source(src: Path) -> None:
    fish_folders = list(iter_fish_folders(src))
    print(f"[DIAG] numeric fish folders found: {len(fish_folders)}")
    for fish_id, fish_dir in fish_folders[:20]:
        side_dirs = find_side_dirs(fish_dir)
        print(f"  fish {fish_id}: {fish_dir.name}")
        for side in ("L", "R"):
            dirs = side_dirs[side]
            print(f"    side {side} dirs: {len(dirs)}")
            for side_dir in dirs[:5]:
                files = [p for p in sorted(side_dir.rglob('*'), key=natural_key) if p.is_file()]
                images = [p for p in files if is_image(p)]
                print(f"      {side_dir}: {len(images)} recognized images, {len(files)} total files")
                for p in files[:8]:
                    mark = "image" if is_image(p) else f"ignored:{p.suffix or '<no ext>'}"
                    print(f"        - {p.name} [{mark}]")
                if len(files) > 8:
                    print(f"        ... {len(files) - 8} more")
            if len(dirs) > 5:
                print(f"      ... {len(dirs) - 5} more side dirs")
    if len(fish_folders) > 20:
        print(f"  ... {len(fish_folders) - 20} more fish folders")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import L/R zebrafish photos into database/photos with ReID naming."
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Source root folder.")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="Destination folder.")
    parser.add_argument("--apply", action="store_true", help="Actually copy/convert files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output PNGs.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N planned files.")
    parser.add_argument("--fish-start", type=int, default=None, help="Only import fish IDs >= this number.")
    parser.add_argument("--fish-end", type=int, default=None, help="Only import fish IDs <= this number.")
    parser.add_argument(
        "--heic-mode",
        choices=("copy", "convert", "skip"),
        default="convert",
        help=(
            "How to handle HEIC/HEIF files: convert writes PNG via Pillow/pillow-heif; "
            "copy keeps the original HEIC bytes with a normalized name; skip ignores them."
        ),
    )
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()

    if not src.exists() or not src.is_dir():
        print(f"[ERROR] Source folder does not exist or is not a directory: {src}")
        return 1

    plan = build_plan(
        src,
        dst,
        args.heic_mode,
        fish_start=args.fish_start,
        fish_end=args.fish_end,
    )
    if args.limit:
        plan = plan[: args.limit]

    if not plan:
        print("[WARN] No images found. Expected numeric fish folders containing L/R image folders.")
        print(f"       Source: {src}")
        diagnose_source(src)
        return 1

    by_fish_side: defaultdict[tuple[str, str], int] = defaultdict(int)
    conflicts = []
    for _in_path, out_path, fish_id, side in plan:
        by_fish_side[(fish_id, side)] += 1
        if out_path.exists() and not args.overwrite:
            conflicts.append(out_path)

    print(f"[SOURCE] {src}")
    print(f"[DEST]   {dst}")
    print(f"[PLAN]   {len(plan)} images")
    print(f"[MODE]   {'apply' if args.apply else 'dry-run'}")
    if args.fish_start is not None or args.fish_end is not None:
        print(f"[FILTER] fish_start={args.fish_start}, fish_end={args.fish_end}")
    print("")
    for (fish_id, side), count in sorted(by_fish_side.items()):
        print(f"  fish {fish_id} side {side}: {count}")

    print("\n[EXAMPLES]")
    for in_path, out_path, _fish_id, _side in plan[:10]:
        print(f"  {in_path} -> {out_path.name}")
    if len(plan) > 10:
        print(f"  ... {len(plan) - 10} more")

    if conflicts:
        print(f"\n[ERROR] {len(conflicts)} output files already exist. Use --overwrite to replace them.")
        for path in conflicts[:10]:
            print(f"  {path}")
        if len(conflicts) > 10:
            print(f"  ... {len(conflicts) - 10} more")
        return 1

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write files.")
        return 0

    written = 0
    for in_path, out_path, _fish_id, _side in plan:
        try:
            save_image(in_path, out_path, overwrite=args.overwrite, heic_mode=args.heic_mode)
        except Exception as exc:
            if in_path.suffix.lower() in HEIC_EXTS and args.heic_mode == "convert":
                print("[HINT] This Python environment cannot decode HEIC. Re-run with")
                print("       --heic-mode copy to keep HEIC files, or install a HEIC Pillow plugin.")
            print(f"[ERROR] Failed: {in_path} -> {out_path}: {exc}")
            return 1
        written += 1

    print(f"\n[DONE] Wrote {written} PNG files to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
