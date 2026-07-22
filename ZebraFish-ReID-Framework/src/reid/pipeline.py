"""
pipeline.py — 斑马鱼 Re-ID 推理管线（方案2 的 reid 模块）

职责：
  1. 从增强图目录批量提取特征（复用 feature_extractor.ZebraFishFeatureExtractor）
  2. 质量加权聚合 → 每条鱼一个身份向量（identity embedding）
  3. Gallery 数据库（SQLite）构建与查询
  4. 交叉视角评估（Rank-1 / Rank-5 / mAP）

特征：3840 维 L2 归一化向量（JPM=True）。余弦相似度 = 向量点积（已归一化）。

用法：
  # 建库 + 交叉视角评估
  python src/reid/pipeline.py --build --eval

  # 仅查询某张图（需先建库）
  python src/reid/pipeline.py --query data/processed/enhanced/0001_c1s1_0001.png
"""
import os
import sys
import re
import json
import argparse
import sqlite3
import numpy as np

# 让本模块可直接 import 同目录的 feature_extractor
_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from feature_extractor import ZebraFishFeatureExtractor

_FRAMEWORK = os.path.dirname(os.path.dirname(_THIS))
ENHANCED_DIR = os.path.join(_FRAMEWORK, "data", "processed", "enhanced")
REPORT_PATH = os.path.join(_FRAMEWORK, "data", "processed", "quality_report.jsonl")
DB_PATH = os.path.join(_FRAMEWORK, "database", "gallery.db")
FEAT_DIM = 3840


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------
def _load_blur_map(report_path=REPORT_PATH):
    """
    从质检流水读取 文件名 -> blur_score 映射，用于质量加权聚合。
    同时合并 video_crop_report.jsonl（视频抽取模块产出），
    使视频来源的鱼图也能按清晰度加权。
    """
    blur = {}
    report_paths = [report_path]
    if os.path.dirname(report_path):
        report_paths.append(os.path.join(os.path.dirname(report_path),
                                         "video_crop_report.jsonl"))
    for rp in report_paths:
        if not os.path.exists(rp):
            continue
        with open(rp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("status") != "accepted":
                    continue
                blur[d["file"]] = float(d.get("blur_score", 0.0))
    return blur


def extract_dataset(enhanced_dir=ENHANCED_DIR, extractor=None, blur_map=None):
    """
    遍历增强图目录，逐张提取 3840 维特征。

    命名约定（务必分清）：
      - 已带左右标记的文件：NNNN_cXsY_ZZZZ.png（c1=左面, c2=右面）→ 这是 ReID 真正使用的图。
      - 无左右标记的文件：NNNN_sY_ZZZZ.png（s-only）→ 由 tools/side_geometry_label.py
        判定为 unknown（拿不准头朝左/右）时【有意保留】的遗留，不是"忘了跑 B"。
        这类图不参与 ReID，会被静默跳过。

    两者可以合法共存：side_geometry_label 把能确定的打 c1/c2，拿不准的留 s1。
    只有当目录里【一张 c1/c2 都没有】时才说明有问题（B 全判成 unknown，或压根没跑 B），
    此时给出警告。

    返回 list of dict:
      {fish_id, cam, frame, fname, feat(np.ndarray[3840]), blur}
    """
    # 已带 c1/c2 的文件（ReID 真正使用的）
    RECORD_RE = re.compile(r'^(\d{4})_c([12])s(\d)_(\d{4})\.(png|jpg|jpeg)$', re.IGNORECASE)
    # 无左右标记、有意遗留的 s-only 文件（不参与 ReID，非错误）
    S_ONLY_RE = re.compile(r'^(\d{4})_s(\d)_(\d{4})\.(png|jpg|jpeg)$', re.IGNORECASE)
    if extractor is None:
        extractor = ZebraFishFeatureExtractor()
    if blur_map is None:
        blur_map = _load_blur_map()
    files = sorted(f for f in os.listdir(enhanced_dir)
                   if f.lower().endswith((".png", ".jpg")))
    records = []
    s_only = 0
    other = 0
    for fname in files:
        m = RECORD_RE.match(fname)
        if m:
            fish_id = m.group(1)
            cam = int(m.group(2)) - 1            # 0=c1, 1=c2
            frame = m.group(4)
            try:
                feat, _ = extractor.extract_from_path(os.path.join(enhanced_dir, fname))
            except Exception as e:
                print(f"[警告] 提取失败 {fname}: {e}")
                continue
            records.append({
                "fish_id": fish_id,
                "cam": cam,
                "frame": frame,
                "fname": fname,
                "feat": feat.astype(np.float32),
                "blur": blur_map.get(fname, 0.0),
            })
            continue
        if S_ONLY_RE.match(fname):
            # 判 unknown 的有意遗留，不是错误，仅计数
            s_only += 1
            continue
        other += 1
    if s_only:
        print(f"[信息] 跳过 {s_only} 张无左右标记(s-only)的图"
              f"（side_geometry_label 判定为 unknown 的有意遗留，不参与 ReID）")
    if not records:
        if s_only or other:
            print(f"[警告] 未提取到任何带 c1/c2 的文件，ReID 无法进行。"
                  f"可能 side_geometry_label 把所有图都判成了 unknown（可调低 --min-head-conf），"
                  f"或未执行该步骤。")
        else:
            print(f"[警告] 目录中没有任何 NNNN_* 图片，请确认 video_extractor 已执行并产出增强图。")
    return records


# ---------------------------------------------------------------------------
# 质量加权聚合
# ---------------------------------------------------------------------------
def aggregate_identity(feats, weights=None):
    """
    将同一鱼的多张图特征聚合成一个身份向量。
    feats: np.ndarray [N, 3840]
    weights: np.ndarray [N]（如 blur_score，越大越清晰，权重越高）；None 则等权
    返回: np.ndarray [3840]，L2 归一化。
    """
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim == 1:
        return feats / (np.linalg.norm(feats) + 1e-12)
    if feats.shape[0] == 1:
        v = feats[0]
        return v / (np.linalg.norm(v) + 1e-12)
    if weights is None:
        w = np.ones(feats.shape[0], dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(w)
    w = w / w.sum()
    agg = (feats * w[:, None]).sum(axis=0)
    agg = agg / (np.linalg.norm(agg) + 1e-12)
    return agg.astype(np.float32)


# ---------------------------------------------------------------------------
# Gallery 数据库（SQLite）
# ---------------------------------------------------------------------------
class GalleryDB:
    def __init__(self, path=DB_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fish_id TEXT NOT NULL,
                cam INTEGER NOT NULL,
                frame TEXT,
                fname TEXT,
                feat BLOB
            )""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS identities (
                fish_id TEXT PRIMARY KEY,
                n_images INTEGER,
                feat BLOB
            )""")
        self.conn.commit()

    def clear(self):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM images")
        cur.execute("DELETE FROM identities")
        self.conn.commit()

    def add_image(self, fish_id, cam, frame, fname, feat):
        blob = np.asarray(feat, dtype=np.float32).tobytes()
        self.conn.execute(
            "INSERT INTO images(fish_id, cam, frame, fname, feat) VALUES (?,?,?,?,?)",
            (fish_id, int(cam), frame, fname, blob))

    def add_identity(self, fish_id, feat, n_images):
        blob = np.asarray(feat, dtype=np.float32).tobytes()
        self.conn.execute(
            "INSERT OR REPLACE INTO identities(fish_id, n_images, feat) VALUES (?,?,?)",
            (fish_id, int(n_images), blob))

    def commit(self):
        self.conn.commit()

    def all_image_feats(self):
        """返回 list of (fish_id, cam, frame, fname, feat[3840])。"""
        cur = self.conn.execute("SELECT fish_id, cam, frame, fname, feat FROM images")
        out = []
        for fish_id, cam, frame, fname, blob in cur.fetchall():
            feat = np.frombuffer(blob, dtype=np.float32)
            if feat.shape[0] == FEAT_DIM:
                out.append((fish_id, int(cam), frame, fname, feat))
        return out

    def all_identities(self):
        """返回 dict: fish_id -> feat[3840]。"""
        cur = self.conn.execute("SELECT fish_id, feat FROM identities")
        out = {}
        for fish_id, blob in cur.fetchall():
            feat = np.frombuffer(blob, dtype=np.float32)
            if feat.shape[0] == FEAT_DIM:
                out[fish_id] = feat
        return out

    def close(self):
        self.conn.close()


def build_gallery(records, db_path=DB_PATH, weight_by_blur=True):
    """
    由 records 构建 Gallery：
      - 逐张写入 images 表
      - 每条鱼按质量加权聚合成身份向量写入 identities 表
    返回 GalleryDB。
    """
    db = GalleryDB(db_path)
    db.clear()
    # 按鱼分组
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[r["fish_id"]].append(r)

    for fish_id, recs in groups.items():
        feats = np.stack([r["feat"] for r in recs], axis=0)   # [N, 3840]
        weights = np.array([r["blur"] for r in recs], dtype=np.float32) if weight_by_blur else None
        ident = aggregate_identity(feats, weights)
        db.add_identity(fish_id, ident, len(recs))            # 整鱼聚合向量
        # 斑马鱼两个侧面花纹存在差异 → 额外按侧面(cam)分别聚合，存各自代表向量
        by_cam = defaultdict(list)
        for r in recs:
            by_cam[r["cam"]].append(r)
        for cam, crecs in by_cam.items():
            cfeats = np.stack([r["feat"] for r in crecs], axis=0)
            cw = np.array([r["blur"] for r in crecs], dtype=np.float32) if weight_by_blur else None
            cident = aggregate_identity(cfeats, cw)
            db.add_identity(f"{fish_id}#c{cam}", cident, len(crecs))
        for r in recs:
            db.add_image(r["fish_id"], r["cam"], r["frame"], r["fname"], r["feat"])
    db.commit()
    print(f"[建库] 共 {len(groups)} 条鱼，{sum(len(v) for v in groups.values())} 张图 -> {db_path}")
    return db


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------
def query_feature(feat, db, topk=5, mode="image"):
    """
    给定查询特征（np.ndarray[3840]，建议已 L2 归一化），返回排名列表。
    mode:
      'image'     : 与 Gallery 中每张图比较，取相似度最高的鱼（标准 Re-ID 检索）
      'identity'  : 与每条鱼的聚合身份向量比较
    返回 list of (fish_id, score)，按 score 降序。
    """
    feat = np.asarray(feat, dtype=np.float32)
    if feat.ndim == 1:
        feat = feat[None, :]
    if mode == "identity":
        idents = db.all_identities()      # 可能含 "0001#c0" 等分侧面前缀
        fish_ids = list(idents.keys())
        mat = np.stack(list(idents.values()), axis=0)        # [F, 3840]
        sims = (mat @ feat.T).squeeze()                      # [F]
        base_ids = [fid.split("#")[0] for fid in fish_ids]   # 去掉 #cX 取基础鱼ID
        order = np.argsort(-sims)
        seen = set()
        result = []
        for i in order:
            b = base_ids[i]
            if b in seen:
                continue
            seen.add(b)
            result.append((b, float(sims[i])))
            if len(result) >= topk:
                break
        return result
    else:
        imgs = db.all_image_feats()
        fish_ids = [x[0] for x in imgs]
        mat = np.stack([x[4] for x in imgs], axis=0)          # [G, 3840]
        sims = (mat @ feat.T).squeeze()                      # [G]
        order = np.argsort(-sims)
        # 取 Top-K 不同鱼（同鱼多图可能占前列）
        seen = set()
        result = []
        for i in order:
            fid = fish_ids[i]
            if fid in seen:
                continue
            seen.add(fid)
            result.append((fid, float(sims[i])))
            if len(result) >= topk:
                break
        return result


def query_image(img_path, db, extractor=None, topk=5, mode="image"):
    if extractor is None:
        extractor = ZebraFishFeatureExtractor()
    feat, _ = extractor.extract_from_path(img_path)
    return query_feature(feat, db, topk=topk, mode=mode)


# ---------------------------------------------------------------------------
# 交叉视角评估（Rank-1 / Rank-5 / mAP）
# ---------------------------------------------------------------------------
def _ap_at_k(ranked_fish, true_fish, num_rel):
    """ranked_fish: list of fish_id（降序）；true_fish: 真实鱼ID；num_rel: 相关图总数。"""
    hits = 0
    ap = 0.0
    for rank, fid in enumerate(ranked_fish, start=1):
        if fid == true_fish:
            hits += 1
            ap += hits / rank
    return ap / num_rel if num_rel > 0 else 0.0


def evaluate_cross_view(db, topk=(1, 5)):
    """
    交叉视角评估：用一侧视角（c1）作 Gallery，另一侧（c2）作 Query，反向再做一次。
    这是 Re-ID 的真实场景——已知鱼从一个侧面拍摄，查询鱼从另一个侧面出现。
    返回 dict: {rank1, rank5, mAP, n_queries, n_fish}。
    """
    imgs = db.all_image_feats()  # (fish_id, cam, frame, fname, feat)
    from collections import defaultdict
    by_fish = defaultdict(lambda: {0: [], 1: []})
    for fish_id, cam, frame, fname, feat in imgs:
        by_fish[fish_id][cam].append(feat)

    # 仅保留同时拥有 c1 与 c2 的鱼
    valid = [fid for fid, d in by_fish.items() if d[0] and d[1]]
    if not valid:
        return {"rank1": 0.0, "rank5": 0.0, "mAP": 0.0,
                "n_queries": 0, "n_fish": 0, "note": "无同时含 c1/c2 的鱼"}

    correct1 = correct5 = 0
    aps = []
    n_queries = 0

    for g_cam, q_cam in ((0, 1), (1, 0)):   # 两个方向
        gallery = []   # (fish_id, feat)
        for fid in valid:
            for feat in by_fish[fid][g_cam]:
                gallery.append((fid, feat))
        queries = []
        for fid in valid:
            for feat in by_fish[fid][q_cam]:
                queries.append((fid, feat))
        if not gallery or not queries:
            continue
        g_fish = np.array([x[0] for x in gallery])
        g_mat = np.stack([x[1] for x in gallery], axis=0).astype(np.float32)  # [G,3840]
        q_fish = np.array([x[0] for x in queries])
        q_mat = np.stack([x[1] for x in queries], axis=0).astype(np.float32)  # [Q,3840]

        sims = g_mat @ q_mat.T                                     # [G, Q]
        for j in range(q_mat.shape[0]):
            true_fish = q_fish[j]
            order = np.argsort(-sims[:, j])
            ranked_fish = [g_fish[i] for i in order]
            pred1 = ranked_fish[0]
            if pred1 == true_fish:
                correct1 += 1
            if true_fish in ranked_fish[:max(topk)]:
                correct5 += 1
            num_rel = int(np.sum(g_fish == true_fish))
            aps.append(_ap_at_k(ranked_fish, true_fish, num_rel))
            n_queries += 1

    return {
        "rank1": correct1 / n_queries if n_queries else 0.0,
        "rank5": correct5 / n_queries if n_queries else 0.0,
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "n_queries": n_queries,
        "n_fish": len(valid),
    }


def evaluate_same_view(db, topk=(1, 5)):
    """
    同视角评估（对照基线）：用 c1 作 Gallery、c1 作 Query，c2 同理。
    这是“较易”场景——查询与库图像来自同一侧面，用于和交叉视角结果对比，
    直观显示侧面花纹差异带来的难度差距。
    """
    imgs = db.all_image_feats()
    from collections import defaultdict
    by_fish = defaultdict(lambda: {0: [], 1: []})
    for fish_id, cam, frame, fname, feat in imgs:
        by_fish[fish_id][cam].append(feat)

    valid = [fid for fid, d in by_fish.items() if d[0] and d[1]]
    if not valid:
        return {"rank1": 0.0, "rank5": 0.0, "mAP": 0.0,
                "n_queries": 0, "n_fish": 0, "note": "无同时含 c1/c2 的鱼"}

    correct1 = correct5 = 0
    aps = []
    n_queries = 0

    for cam in (0, 1):   # c1→c1, c2→c2
        gallery = [(fid, feat) for fid in valid for feat in by_fish[fid][cam]]
        queries = [(fid, feat) for fid in valid for feat in by_fish[fid][cam]]
        if not gallery or not queries:
            continue
        g_fish = np.array([x[0] for x in gallery])
        g_mat = np.stack([x[1] for x in gallery], axis=0).astype(np.float32)
        q_fish = np.array([x[0] for x in queries])
        q_mat = np.stack([x[1] for x in queries], axis=0).astype(np.float32)

        sims = g_mat @ q_mat.T
        for j in range(q_mat.shape[0]):
            true_fish = q_fish[j]
            order = np.argsort(-sims[:, j])
            ranked_fish = [g_fish[i] for i in order]
            if ranked_fish[0] == true_fish:
                correct1 += 1
            if true_fish in ranked_fish[:max(topk)]:
                correct5 += 1
            num_rel = int(np.sum(g_fish == true_fish))
            aps.append(_ap_at_k(ranked_fish, true_fish, num_rel))
            n_queries += 1

    return {
        "rank1": correct1 / n_queries if n_queries else 0.0,
        "rank5": correct5 / n_queries if n_queries else 0.0,
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "n_queries": n_queries,
        "n_fish": len(valid),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhanced", default=ENHANCED_DIR)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--build", action="store_true", help="从增强图构建 Gallery")
    ap.add_argument("--eval", action="store_true", help="交叉视角评估")
    ap.add_argument("--query", default=None, help="查询单张图，输出 Top-K 鱼ID")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--no-blur-weight", action="store_true", help="聚合时等权而非按清晰度加权")
    args = ap.parse_args()

    extractor = ZebraFishFeatureExtractor()

    if args.build or args.eval:
        print("[提取] 批量提取增强图特征 ...")
        records = extract_dataset(args.enhanced, extractor=extractor)
        print(f"  提取 {len(records)} 张（覆盖 {len(set(r['fish_id'] for r in records))} 条鱼）")
        if args.build:
            db = build_gallery(records, db_path=args.db,
                               weight_by_blur=not args.no_blur_weight)
        else:
            db = GalleryDB(args.db)
    else:
        db = GalleryDB(args.db)

    if args.eval:
        res = evaluate_cross_view(db)
        print("\n=== 交叉视角评估 (c1<->c2，最难场景) ===")
        print(f"  参与鱼数      : {res['n_fish']}")
        print(f"  查询数        : {res['n_queries']}")
        print(f"  Rank-1        : {res['rank1']*100:.2f}%")
        print(f"  Rank-5        : {res['rank5']*100:.2f}%")
        print(f"  mAP           : {res['mAP']*100:.2f}%")

        res2 = evaluate_same_view(db)
        print("\n=== 同视角评估 (c1->c1 / c2->c2，较易场景) ===")
        print(f"  参与鱼数      : {res2['n_fish']}")
        print(f"  查询数        : {res2['n_queries']}")
        print(f"  Rank-1        : {res2['rank1']*100:.2f}%")
        print(f"  Rank-5        : {res2['rank5']*100:.2f}%")
        print(f"  mAP           : {res2['mAP']*100:.2f}%")

    if args.query:
        res = query_image(args.query, db, extractor=extractor, topk=args.topk)
        print(f"\n[查询] {args.query}")
        for rank, (fid, score) in enumerate(res, 1):
            print(f"  #{rank} 鱼 {fid}  余弦相似度 {score:.4f}")

    db.close()


if __name__ == "__main__":
    main()
