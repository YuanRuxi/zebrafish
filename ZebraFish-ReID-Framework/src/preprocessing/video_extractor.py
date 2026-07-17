"""
video_extractor.py — 视频 → 斑马鱼裁剪图（preprocessing 板块模块）

【设计（用户提供完整视频，非侧面视频段）】
  用户直接喂给框架"每条鱼的完整视频"（可能含侧面、非侧面，左右面交替出现）。
  本模块用【单类 YOLO】逐帧检测斑马鱼（不再区分左右）：
    1) 用单类 YOLO 模型（zebrafish）检测；
    2) 仅当某帧被以"较高置信度"判为斑马鱼时，才裁出鱼体；
       非鱼/模糊/遮挡 → 置信度不足 → 直接舍去；
    3) 清晰度（Laplacian 方差）与去冗余（相邻帧 MAE）进一步筛选。

  ⚠️ 重要：本步骤【只检测鱼，不判断左右】。左右（c1/c2）由后处理
     tools/side_geometry_label.py（几何朝向法）判定并改文件名。
     因此原始裁剪图命名【不带 c1/c2】，仅含鱼ID + 全身段 s1 + 帧号。

  视频命名规范（只含鱼ID，左右由后处理判定，不写进文件名）：
        NNNN.ext           例：0001.mp4  →  鱼ID=0001
        NNNN_任意.ext       例：0001_tank3.mp4

输出：
  data/raw/video_crops/NNNN_s1_ZZZZ.png         原始裁剪（原分辨率，无 c1/c2）
  data/processed/enhanced/NNNN_s1_ZZZZ.png      增强后（CLAHE，与框架一致，无 c1/c2）
  data/processed/video_crop_report.jsonl        逐帧流水（cam 暂为 null，供后处理填）

命名：s1 固定（全身）；ZZZZ 为 该鱼ID 的全局顺序号。左右来源后由
      side_geometry_label.py 写入文件名（c1/c2）与报告 cam 字段。

帧选取（三段式）：
  1) 时间覆盖：按 sample_stride 均匀抽候选帧
  2) 检测+清晰度：单类 YOLO 置信 >= conf_side 且 Laplacian 方差 >= 模糊阈值
  3) 去冗余：与上一已选帧(在鱼裁剪上)的灰度 MAE > dedup_mae 才保留

YOLO：仅推理加载任意 .pt（单类 zebrafish 模型）；单鱼假设下取 conf 最大的框裁剪。
      ★ 训练/标注在框架之外，本模块只做推理加载。
      ★ 把训练得到的 best.pt 放进 models/（或改 configs/video_extraction.json 的 yolo_model）。

用法：
  python src/preprocessing/video_extractor.py --videos "原始视频/*.mp4" --dry-run
  python src/preprocessing/video_extractor.py --videos "原始视频/*.mp4"
  python src/preprocessing/video_extractor.py --videos 0001.mp4 --conf 0.95
"""
import os
import sys
import re
import json
import glob
import argparse
import numpy as np
import cv2
from PIL import Image

# 复用同目录的质检与增强（绕开 OpenCV 的 PNG 解码坑，沿用框架既有实现）
_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
from quality import compute_blur_score
from enhance import enhance_clahe

_FRAMEWORK = os.path.dirname(os.path.dirname(_THIS))

RAW_DIR = os.path.join(_FRAMEWORK, "data", "raw", "video_crops")
ENH_DIR = os.path.join(_FRAMEWORK, "data", "processed", "enhanced")
REPORT_PATH = os.path.join(_FRAMEWORK, "data", "processed", "video_crop_report.jsonl")
QUALITY_THRESH = os.path.join(_FRAMEWORK, "configs", "quality_thresholds.json")
CONFIG_PATH = os.path.join(_FRAMEWORK, "configs", "video_extraction.json")

# 视频命名：取文件名中第一个 4 位数字组作为鱼ID（NNNN）。
# 左右 c1/c2 由后处理几何法决定，本步不解析。
VIDEO_RE = re.compile(r"(\d{4})")
# 输出图片命名：NNNN_s1_ZZZZ.png（本步只检测鱼，不带 c1/c2；s1=全身）
NAME_RE = re.compile(r"^(\d{4})_s(\d)_(\d{4})\.png$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_yolo(model):
    """解析单类 YOLO 权重路径：支持绝对路径、相对框架根目录的路径、或默认 models/yolov8_zebrafish.pt。"""
    if model and os.path.isabs(model) and os.path.exists(model):
        return model
    abs_path = model if (model and os.path.isabs(model)) else os.path.join(
        _FRAMEWORK, model or os.path.join("models", "yolov8_zebrafish.pt"))
    if os.path.exists(abs_path):
        return abs_path
    return model  # 找不到则原样返回，交给 ultralytics 报清晰错误


# ---------------------------------------------------------------------------
# YOLO 检测（仅推理，单类）
# ---------------------------------------------------------------------------
def load_yolo(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)


def detect_fish(frame_bgr, model, conf_side, imgsz=1920):
    """
    在单帧上跑【单类】YOLO（zebrafish），返回 (box5, conf) 或 None：
      box5 = (x1,y1,x2,y2,conf)。
    仅当检测到斑马鱼且 conf>=conf_side 才返回；否则（非鱼/置信不足/未检测）
    → 返回 None（该帧舍去）。单鱼假设：在合格框中取 conf 最大者。
    imgsz: 推理尺寸；4K 整帧需放大(如1920)才能检出小鱼。

    ⚠️ 语义澄清：conf_side 只表示「框里是鱼」的把握，并不表示「鱼是否摆正 / 是否完整侧面」。
       姿态完整性（完整侧面、无遮挡、全身在框内）由下游 side_geometry_label（几何法判左右）
       + 可选「完整性过滤」负责，本函数只解决「是不是鱼」。
    """
    results = model(frame_bgr, verbose=False, imgsz=imgsz)[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = None
    best_score = -1.0
    for i in range(len(boxes)):
        c = float(boxes.conf[i].item())
        if c < conf_side:                  # 置信度不够高 → 视为非鱼，跳过
            continue
        if c > best_score:
            best_score = c
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            best = tuple(xyxy)
    if best is None:
        return None
    return (best[0], best[1], best[2], best[3], best_score), best_score


def crop_with_margin(frame, box, margin_ratio):
    """按 YOLO 框向外扩展 margin_ratio*max(w,h)，clamp 到图像边界，返回裁剪 BGR。"""
    x1, y1, x2, y2, _ = box
    h, w = frame.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    m = int(margin_ratio * max(bw, bh))
    x1 = max(0, x1 - m)
    y1 = max(0, y1 - m)
    x2 = min(w, x2 + m)
    y2 = min(h, y2 + m)
    if x2 <= x1 or y2 <= y1:
        return frame.copy()
    return frame[y1:y2, x1:x2].copy()


def frame_mae(a, b):
    """两帧（BGR）下采样灰度后的平均绝对误差，用于近重复帧判定。"""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ga = cv2.resize(ga, (64, 64))
    gb = cv2.resize(gb, (64, 64))
    return float(np.mean(np.abs(ga.astype(np.float32) - gb.astype(np.float32)) / 255.0))


# ---------------------------------------------------------------------------
# 三段式帧选取 + 裁剪
# ---------------------------------------------------------------------------
def iter_selected(video_path, model, cfg, blur_thresh):
    """
    逐帧遍历视频，按"时间覆盖 + 检测/清晰度 + 去冗余"产出自带裁剪的鱼图。
    模型=None（dry-run）时跳过检测，用整帧代替裁剪做去冗余判断。
    yield: (frame_idx, crop_bgr, blur_score, conf, cam)
      cam 恒为 None（左右由后处理几何法判定，本步不产出）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[警告] 无法打开视频: {video_path}")
        return
    stride = max(1, int(cfg["sample_stride"]))
    dedup = float(cfg["dedup_mae"])
    conf_side = float(cfg["conf_side"])
    imgsz = int(cfg.get("imgsz", 1920))
    last_crop = None
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % stride != 0:
            fi += 1
            continue
        if model is not None:
            det = detect_fish(frame, model, conf_side, imgsz)
            if det is None:              # 阶段2b：非鱼 / 置信不足，丢
                fi += 1
                continue
            (box5, conf) = det
            crop = crop_with_margin(frame, box5, cfg["margin_ratio"])
        else:
            crop = frame
            conf = 0.0
        cam = None
        if crop is None or crop.size == 0:
            fi += 1
            continue
        # 阶段2a：在“裁剪后的鱼体”上算清晰度（全帧对小鱼4K无效，会全被滤掉）
        blur = compute_blur_score(crop)
        if blur < blur_thresh:
            fi += 1
            continue
        if last_crop is not None and frame_mae(crop, last_crop) < dedup:  # 阶段3：太像，丢
            fi += 1
            continue
        last_crop = crop
        yield fi, crop, blur, conf, cam
        fi += 1
    cap.release()


def process_video(video_path, model, cfg, blur_thresh, raw_dir, enh_dir,
                  rf, counter, max_frames, dry_run, use_enhance,
                  fish_id=None):
    """处理单个视频：解析鱼ID → 帧选取 → 单类裁剪 → 命名(无c1/c2) → 存图 → 写报告。
    fish_id: 若提供（如 --auto-id 顺序编号），直接用作鱼ID，跳过文件名解析。
    命名：NNNN_s1_ZZZZ.png（本步不判左右，cam 记为 null）。"""
    base = os.path.basename(video_path)
    if fish_id is None:
        m = VIDEO_RE.search(base)
        if not m:
            print(f"[跳过] 视频名中未找到 4 位鱼ID: {base}（应为 NNNN.ext，如 0001.mp4）")
            return {"video": base, "accepted": 0, "badname": True}
        fish_id = m.group(1)

    accepted = 0
    for fi, crop, blur, conf, cam in iter_selected(video_path, model, cfg, blur_thresh):
        if counter.get(fish_id, 0) >= max_frames:   # 该鱼总量上限
            break
        counter[fish_id] = counter.get(fish_id, 0) + 1
        zzzz = counter[fish_id]
        fname = f"{fish_id}_s1_{zzzz:04d}.png"       # 本步不带 c1/c2
        if NAME_RE.match(fname) is None:
            raise RuntimeError(f"内部错误：生成的文件名 {fname} 不符合规范")

        if not dry_run:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(os.path.join(raw_dir, fname))
            if use_enhance:
                enh = enhance_clahe(crop, clip_limit=cfg["clip_limit"])
                rgbe = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgbe).save(os.path.join(enh_dir, fname))

        rec = {
            "file": fname,
            "status": "accepted",
            "blur_score": round(blur, 2),
            "fish_id": fish_id,
            "cam": None,                       # 左右由后处理几何法判定
            "frame": f"{zzzz:04d}",
            "video": base,
            "frame_idx": fi,
            "conf": round(float(conf), 3),    # YOLO 检测(单类)置信度
        }
        rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        accepted += 1

    return {"video": base, "accepted": accepted, "fish_id": fish_id}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="视频 → 斑马鱼裁剪图（新思路：完整视频 + 单类 YOLO 检测，只判'是不是鱼'，不判左右）")
    ap.add_argument("--videos", nargs="+", required=True,
                    help="视频文件/目录/通配符；按文件名中的 4 位鱼ID 解析")
    ap.add_argument("--out-raw", default=RAW_DIR)
    ap.add_argument("--out-enh", default=ENH_DIR)
    ap.add_argument("--report", default=REPORT_PATH)
    ap.add_argument("--yolo", default=None, help="单类 zebrafish 检测模型 .pt 路径（覆盖配置；左右由 side_geometry_label 判定）")
    ap.add_argument("--conf", type=float, default=None, help="YOLO 检测置信度阈值（是否为斑马鱼）；越高越严格，只保留高确定性鱼帧（左右判定在后续几何法）")
    ap.add_argument("--stride", type=int, default=None, help="候选帧步长（每隔 N 帧抽一帧）")
    ap.add_argument("--max-frames", type=int, default=None, help="每条鱼最多保留帧数（c1/c2 共用）")
    ap.add_argument("--dedup", type=float, default=None, help="去冗余 MAE 阈值（越小越严）")
    ap.add_argument("--margin", type=float, default=None, help="YOLO 框外扩比例")
    ap.add_argument("--no-enhance", action="store_true", help="跳过 CLAHE 增强")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个视频（调试）")
    ap.add_argument("--dry-run", action="store_true", help="只预览匹配与计数，不加载模型/不写文件")
    ap.add_argument("--auto-id", action="store_true",
                    help="按视频排序顺序自动编号鱼ID(0001,0002,...)，忽略文件名解析（用于时间戳命名视频）")
    ap.add_argument("--imgsz", type=int, default=None, help="YOLO 推理尺寸（4K 视频需 1920 才能检出小鱼）")
    args = ap.parse_args()

    cfg = load_config()
    if args.yolo:
        cfg["yolo_model"] = args.yolo
    if args.conf is not None:
        cfg["conf_side"] = args.conf
    if args.imgsz is not None:
        cfg["imgsz"] = args.imgsz
    if args.stride is not None:
        cfg["sample_stride"] = args.stride
    if args.max_frames is not None:
        cfg["max_frames_per_fish"] = args.max_frames
    if args.dedup is not None:
        cfg["dedup_mae"] = args.dedup
    if args.margin is not None:
        cfg["margin_ratio"] = args.margin
    if args.no_enhance:
        cfg["use_enhance"] = False

    # 模糊阈值：优先复用现有 quality_thresholds.json，否则用经验默认
    blur_thresh = cfg["blur_thresh"]
    if blur_thresh is None:
        if os.path.exists(QUALITY_THRESH):
            with open(QUALITY_THRESH, "r", encoding="utf-8") as f:
                blur_thresh = float(json.load(f)["blur_thresh"])
            print(f"[阈值] 复用 quality_thresholds.json 模糊阈值 = {blur_thresh:.2f}")
        else:
            blur_thresh = 5.91
            print(f"[阈值] 未找到 quality_thresholds.json，用默认 {blur_thresh:.2f}")

    # 收集视频（按文件名中的 4 位鱼ID 解析）
    vids = []
    for v in args.videos:
        if any(ch in v for ch in "*?[]"):
            vids.extend(sorted(glob.glob(v)))
        elif os.path.isdir(v):
            vids.extend(sorted(os.path.join(v, x) for x in os.listdir(v)
                               if x.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))))
        elif os.path.exists(v):
            vids.append(v)
    # --auto-id 模式下视频名可能完全不含 4 位鱼ID（如时间戳命名），
    # 鱼ID由顺序自动分配，故不按 VIDEO_RE 过滤；否则要求文件名含 4 位鱼ID。
    if not args.auto_id:
        vids = [v for v in vids if VIDEO_RE.search(os.path.basename(v))]
    if args.limit:
        vids = vids[:args.limit]

    # 自动编号鱼ID（--auto-id）：按排序后的视频顺序分配 0001,0002,...
    fish_id_of = {}
    if args.auto_id:
        vids_sorted = sorted(vids)
        for i, vp in enumerate(vids_sorted):
            fish_id_of[vp] = f"{i + 1:04d}"
        map_path = os.path.join(_FRAMEWORK, "data", "raw", "video_fish_id_map.json")
        with open(map_path, "w", encoding="utf-8") as mf:
            json.dump(
                {os.path.basename(vp): fish_id_of[vp] for vp in vids_sorted},
                mf, ensure_ascii=False, indent=2,
            )
        print(f"[auto-id] 为 {len(vids_sorted)} 个视频分配鱼ID 0001..{len(vids_sorted):04d}，"
              f"对照表: {map_path}")
    else:
        for vp in vids:
            fish_id_of[vp] = None  # 仍按文件名解析

    if not vids:
        if args.auto_id:
            print("[错误] 未找到任何视频文件。请检查 --videos 是否为存在的"
                  "视频文件/目录/通配符（如 \"D:/路径/xxx.mov\" 或 \"D:/路径/*.mov\"），"
                  "并注意含空格/中文的完整路径要用引号包住。")
        else:
            print("[错误] 未匹配到含 4 位鱼ID 的视频。需 NNNN.ext（如 0001.mp4），"
                  "或加 --auto-id 让脚本按顺序自动编号。")
        return

    model = None
    if not args.dry_run:
        yolo_path = resolve_yolo(cfg["yolo_model"])
        print(f"[YOLO] 加载 {yolo_path}（单类斑马鱼检测模型）")
        model = load_yolo(yolo_path)

    os.makedirs(args.out_raw, exist_ok=True)
    os.makedirs(args.out_enh, exist_ok=True)
    counter = {}
    # 防御性：从已有增强/原图里取每鱼已用的最大 ZZZZ，避免覆盖既有文件
    for _d in (args.out_raw, args.out_enh):
        if os.path.isdir(_d):
            for _fn in os.listdir(_d):
                _mm = NAME_RE.match(_fn)
                if _mm:
                    _fid = _mm.group(1)
                    _n = int(_mm.group(3))
                    if counter.get(_fid, 0) < _n:
                        counter[_fid] = _n
    total_acc = 0
    with open(args.report, "w", encoding="utf-8") as rf:
        for vp in vids:
            r = process_video(vp, model, cfg, blur_thresh, args.out_raw, args.out_enh,
                              rf, counter, cfg["max_frames_per_fish"],
                              args.dry_run, cfg["use_enhance"],
                              fish_id=fish_id_of.get(vp))
            total_acc += r.get("accepted", 0)
            tag = "dry-run" if args.dry_run else "ok"
            print(f"  [{tag}] {r['video']}: 接受 {r.get('accepted', 0)}")
    print(f"\n[完成] 共 {len(vids)} 个视频，生成 {total_acc} 张鱼图")
    if not args.dry_run:
        print(f"  原图: {args.out_raw}")
        print(f"  增强: {args.out_enh}")
    print(f"  报告: {args.report}")


if __name__ == "__main__":
    main()
