#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate video-extracted frames against the photo-built ReID gallery.

Gallery:
    database/photos_gallery.db

Queries:
    data/processed/enhanced/NNNN_cXsY_ZZZZ.png

The script reports retrieval metrics for:
  - image mode: query frame vs every photo feature in the gallery
  - identity mode: query frame vs one aggregated photo identity vector per fish

It also writes a CSV with per-query Top-K matches for inspection.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np


FRAMEWORK = Path(__file__).resolve().parents[1]
REID_DIR = FRAMEWORK / "src" / "reid"
if str(REID_DIR) not in sys.path:
    sys.path.insert(0, str(REID_DIR))

from feature_extractor import ZebraFishFeatureExtractor
from pipeline import GalleryDB


QUERY_RE = re.compile(r"^(\d{4})_c([12])s(\d)_(\d{4})\.(png|jpg|jpeg)$", re.I)


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)


def load_queries(query_dir: Path) -> list[dict]:
    rows = []
    skipped = 0
    for path in sorted(query_dir.iterdir(), key=natural_key):
        if not path.is_file():
            continue
        m = QUERY_RE.match(path.name)
        if not m:
            skipped += 1
            continue
        rows.append(
            {
                "path": path,
                "fname": path.name,
                "fish_id": m.group(1),
                "cam": int(m.group(2)) - 1,
                "frame": m.group(4),
            }
        )
    return rows, skipped


def ap_for_ranked(ranked_fish: list[str], true_fish: str, num_rel: int) -> float:
    if num_rel <= 0:
        return 0.0
    hits = 0
    ap = 0.0
    for rank, fish_id in enumerate(ranked_fish, 1):
        if fish_id == true_fish:
            hits += 1
            ap += hits / rank
    return ap / num_rel


def summarize(results: list[dict]) -> dict:
    if not results:
        return {"n_queries": 0, "rank1": 0.0, "rank5": 0.0, "mAP": 0.0}
    return {
        "n_queries": len(results),
        "rank1": float(np.mean([r["rank1"] for r in results])),
        "rank5": float(np.mean([r["rank5"] for r in results])),
        "mAP": float(np.mean([r["ap"] for r in results])),
    }


def evaluate_image_mode(query_records: list[dict], db: GalleryDB, extractor, topk: int, same_cam: bool = False) -> list[dict]:
    gallery = db.all_image_feats()
    results = []
    for q in query_records:
        rows = [row for row in gallery if (not same_cam or row[1] == q["cam"])]
        if not rows:
            continue
        g_fish = np.array([row[0] for row in rows])
        g_names = np.array([row[3] for row in rows])
        g_mat = np.stack([row[4] for row in rows], axis=0).astype(np.float32)
        feat, _cam = extractor.extract_from_path(str(q["path"]))
        sims = g_mat @ feat.astype(np.float32)
        order = np.argsort(-sims)
        ranked_fish = [str(g_fish[i]) for i in order]
        true_fish = q["fish_id"]
        num_rel = int(np.sum(g_fish == true_fish))
        top_unique = []
        seen = set()
        for idx in order:
            fish_id = str(g_fish[idx])
            if fish_id in seen:
                continue
            seen.add(fish_id)
            top_unique.append((fish_id, float(sims[idx]), str(g_names[idx])))
            if len(top_unique) >= topk:
                break
        results.append(
            {
                "query": q,
                "rank1": 1.0 if ranked_fish and ranked_fish[0] == true_fish else 0.0,
                "rank5": 1.0 if true_fish in ranked_fish[:5] else 0.0,
                "ap": ap_for_ranked(ranked_fish, true_fish, num_rel),
                "top": top_unique,
            }
        )
    return results


def evaluate_identity_mode(query_records: list[dict], db: GalleryDB, extractor, topk: int, same_cam: bool = False) -> list[dict]:
    all_idents = db.all_identities()
    results = []
    for q in query_records:
        if same_cam:
            suffix = f"#c{q['cam']}"
            idents = {k.split("#")[0]: v for k, v in all_idents.items() if k.endswith(suffix)}
        else:
            idents = {k: v for k, v in all_idents.items() if "#" not in k}
        if not idents:
            continue
        fish_ids = np.array(sorted(idents))
        mat = np.stack([idents[fid] for fid in fish_ids], axis=0).astype(np.float32)
        feat, _cam = extractor.extract_from_path(str(q["path"]))
        sims = mat @ feat.astype(np.float32)
        order = np.argsort(-sims)
        ranked_fish = [str(fish_ids[i]) for i in order]
        true_fish = q["fish_id"]
        top = [(str(fish_ids[i]), float(sims[i]), f"{fish_ids[i]}") for i in order[:topk]]
        results.append(
            {
                "query": q,
                "rank1": 1.0 if ranked_fish and ranked_fish[0] == true_fish else 0.0,
                "rank5": 1.0 if true_fish in ranked_fish[:5] else 0.0,
                "ap": ap_for_ranked(ranked_fish, true_fish, 1),
                "top": top,
            }
        )
    return results


def write_csv(path: Path, image_results: list[dict], identity_results: list[dict], topk: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ident_by_name = {r["query"]["fname"]: r for r in identity_results}
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["query", "true_fish", "cam", "image_rank1", "image_ap", "identity_rank1", "identity_ap"]
        for mode in ("image", "identity"):
            for i in range(1, topk + 1):
                header += [f"{mode}_top{i}_fish", f"{mode}_top{i}_score", f"{mode}_top{i}_source"]
        writer.writerow(header)
        for img_r in image_results:
            q = img_r["query"]
            ident_r = ident_by_name[q["fname"]]
            row = [
                q["fname"],
                q["fish_id"],
                f"c{q['cam'] + 1}",
                img_r["rank1"],
                f"{img_r['ap']:.6f}",
                ident_r["rank1"],
                f"{ident_r['ap']:.6f}",
            ]
            for result in (img_r, ident_r):
                top = result["top"]
                for i in range(topk):
                    if i < len(top):
                        row += [top[i][0], f"{top[i][1]:.6f}", top[i][2]]
                    else:
                        row += ["", "", ""]
            writer.writerow(row)


def print_summary(title: str, summary: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"queries : {summary['n_queries']}")
    print(f"Rank-1  : {summary['rank1'] * 100:.2f}%")
    print(f"Rank-5  : {summary['rank5'] * 100:.2f}%")
    print(f"mAP     : {summary['mAP'] * 100:.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate video frames against photo gallery.")
    parser.add_argument("--query-dir", type=Path, default=FRAMEWORK / "data" / "processed" / "enhanced")
    parser.add_argument("--db", type=Path, default=FRAMEWORK / "database" / "photos_gallery.db")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out", type=Path, default=FRAMEWORK / "database" / "video_vs_photo_gallery_results.csv")
    args = parser.parse_args()

    query_records, skipped = load_queries(args.query_dir)
    if not query_records:
        print(f"[ERROR] No valid query images found in {args.query_dir}")
        return 1
    db = GalleryDB(str(args.db))
    gallery_fish = {row[0] for row in db.all_image_feats()}
    query_records = [q for q in query_records if q["fish_id"] in gallery_fish]
    if not query_records:
        print("[ERROR] No query fish IDs overlap with the photo gallery.")
        return 1

    print(f"[QUERY] {args.query_dir}")
    print(f"[DB]    {args.db}")
    print(f"[INFO]  valid queries with gallery labels: {len(query_records)}")
    print(f"[INFO]  skipped unlabeled/nonmatching files: {skipped}")
    print(f"[INFO]  gallery fish IDs: {len(gallery_fish)}")

    extractor = ZebraFishFeatureExtractor()
    image_results = evaluate_image_mode(query_records, db, extractor, args.topk)
    identity_results = evaluate_identity_mode(query_records, db, extractor, args.topk)
    same_cam_image_results = evaluate_image_mode(query_records, db, extractor, args.topk, same_cam=True)
    same_cam_identity_results = evaluate_identity_mode(query_records, db, extractor, args.topk, same_cam=True)
    write_csv(args.out, image_results, identity_results, args.topk)

    print_summary("Photo Image Gallery (video frame -> individual photo features)", summarize(image_results))
    print_summary("Photo Identity Gallery (video frame -> aggregated fish identity)", summarize(identity_results))
    print_summary("Same-Side Photo Image Gallery (c1->c1, c2->c2)", summarize(same_cam_image_results))
    print_summary("Same-Side Photo Identity Gallery (c1->c1, c2->c2)", summarize(same_cam_identity_results))
    print(f"\n[CSV] {args.out}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
