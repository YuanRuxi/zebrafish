#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_pipeline.py — 斑马鱼 Re-ID 全流程一键启动脚本
================================================================

一条命令跑完整条链路：

    视频 --> YOLO 检测+高置信筛帧+裁剪
         --> 几何朝向法判左右并改名 (c1/c2)
         --> c2 就地镜像成 head-left (对齐 TransReID 训练约定)
         --> TransReID 提特征 + 建 Gallery 数据库
         --> 交叉视角 / 同视角评估 (Rank-1 / Rank-5 / mAP)

设计要点
--------
1. **幂等**：每次运行默认先清空上一轮的中间产物
   (enhanced/ video_crops/ 报告 gallery.db)，避免 c2 被重复镜像
   (镜像两次会翻回原始右面) 或旧图混入。用 --keep 可保留。
2. **分阶段可跳过**：--skip-extract / --skip-side / --skip-mirror /
   --skip-reid，便于只重跑其中某一步。
3. **同解释器**：用当前 Python (sys.executable) 调各子脚本，
   保证 ultralytics / torch / cv2 环境一致。
4. **准确度回收**：捕获 pipeline 的评估输出，最后汇总打印。

典型用法
--------
    # 小样验证：几个新视频（时间戳命名用 --auto-id）
    python run_pipeline.py --videos "D:/.../视频集/*.mov" --auto-id \
        --imgsz 1920 --conf 0.7 --max-frames 40

    # 视频名自带 4 位鱼号 (0001.mp4) 则不用 --auto-id
    python run_pipeline.py --videos "data/videos/*.mp4" --conf 0.7

    # 只重跑 ReID（前面产物已就绪）
    python run_pipeline.py --skip-extract --skip-side --skip-mirror
"""
import os
import sys
import glob
import time
import shutil
import argparse
import subprocess

# --------------------------------------------------------------------------
# 路径常量（全部相对本文件所在的框架根目录）
# --------------------------------------------------------------------------
FRAMEWORK = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable  # 用启动本脚本的同一个解释器跑子脚本

VIDEO_EXTRACTOR = os.path.join(FRAMEWORK, "src", "preprocessing", "video_extractor.py")
SIDE_GEOMETRY   = os.path.join(FRAMEWORK, "tools", "side_geometry_label.py")
MIRROR_C2       = os.path.join(FRAMEWORK, "tools", "mirror_c2.py")
REID_PIPELINE   = os.path.join(FRAMEWORK, "src", "reid", "pipeline.py")

RAW_DIR        = os.path.join(FRAMEWORK, "data", "raw", "video_crops")
ENHANCED_DIR   = os.path.join(FRAMEWORK, "data", "processed", "enhanced")
CROP_REPORT    = os.path.join(FRAMEWORK, "data", "processed", "video_crop_report.jsonl")
GEOM_REPORT    = os.path.join(FRAMEWORK, "data", "processed", "_side_geom_report.jsonl")
DB_PATH        = os.path.join(FRAMEWORK, "database", "gallery.db")


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------
def banner(step_no, total, title):
    line = "=" * 64
    print(f"\n{line}\n【阶段 {step_no}/{total}】{title}\n{line}", flush=True)


def _clear_dir(path):
    """清空目录内容但保留目录本身。"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        return 0
    n = 0
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        try:
            if os.path.isfile(fp) or os.path.islink(fp):
                os.remove(fp)
                n += 1
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
                n += 1
        except OSError as e:
            print(f"  [警告] 无法删除 {fp}: {e}")
    return n


def clean_intermediates():
    """清空上一轮产物，保证幂等（不动 _backup_* 珍贵备份）。"""
    print("[清理] 清空上一轮中间产物 ...")
    a = _clear_dir(RAW_DIR)
    b = _clear_dir(ENHANCED_DIR)
    for f in (CROP_REPORT, GEOM_REPORT, DB_PATH):
        if os.path.isfile(f):
            os.remove(f)
            print(f"  已删 {os.path.relpath(f, FRAMEWORK)}")
    print(f"  video_crops 清 {a} 项，enhanced 清 {b} 项")


def run_stage(cmd, title, capture=False):
    """
    运行一个子进程阶段。
    capture=True 时同时实时打印并收集 stdout（用于回收准确度）。
    返回 (returncode, collected_stdout_or_None)。
    """
    print(f"[执行] {' '.join(_pretty(c) for c in cmd)}\n", flush=True)
    t0 = time.time()
    if not capture:
        ret = subprocess.run(cmd, cwd=FRAMEWORK)
        dt = time.time() - t0
        print(f"\n[{title}] 用时 {dt:.1f}s，退出码 {ret.returncode}", flush=True)
        return ret.returncode, None
    # 流式捕获
    lines = []
    proc = subprocess.Popen(cmd, cwd=FRAMEWORK, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    proc.wait()
    dt = time.time() - t0
    print(f"\n[{title}] 用时 {dt:.1f}s，退出码 {proc.returncode}", flush=True)
    return proc.returncode, "".join(lines)


def _pretty(c):
    """命令行片段带空格时加引号，方便复制。"""
    return f'"{c}"' if (" " in c or "*" in c) else c


def _extract_metrics(text):
    """从 pipeline 输出里抓 Rank-1 / Rank-5 / mAP 两组评估结果。"""
    if not text:
        return None
    blocks = {}
    cur = None
    for line in text.splitlines():
        s = line.strip()
        if "交叉视角评估" in s:
            cur = "交叉视角 (c1↔c2)"
            blocks[cur] = {}
        elif "同视角评估" in s:
            cur = "同视角 (c1→c1/c2→c2)"
            blocks[cur] = {}
        elif cur:
            for key in ("Rank-1", "Rank-5", "mAP"):
                if s.startswith(key):
                    # 形如 "Rank-1        : 67.27%"
                    val = s.split(":", 1)[-1].strip()
                    blocks[cur][key] = val
    return blocks or None


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="斑马鱼 Re-ID 全流程一键启动",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # 输入
    ap.add_argument("--videos", nargs="+", default=None,
                    help="视频路径/通配符（可多个）。除非全程跳过抽取，否则必填")
    ap.add_argument("--auto-id", action="store_true",
                    help="视频名不含 4 位鱼号时，按 sorted 顺序自动编 0001..")
    ap.add_argument("--imgsz", type=int, default=1920,
                    help="YOLO 推理尺寸（4K 视频需 1920，默认 1920）")
    ap.add_argument("--conf", type=float, default=0.7,
                    help="检测置信度阈值（是否为斑马鱼，默认 0.7；弱模型可调低诊断）")
    ap.add_argument("--max-frames", type=int, default=40,
                    help="每条鱼最多保留帧数（默认 40，小样测试用）")
    ap.add_argument("--stride", type=int, default=None, help="候选帧步长")
    # 左右判别
    ap.add_argument("--flip-map", action="store_true",
                    help="翻转 头向->c1/c2 的映射（若发现整体判反）")
    # 阶段开关
    ap.add_argument("--skip-extract", action="store_true", help="跳过 YOLO 抽帧裁剪")
    ap.add_argument("--skip-side", action="store_true", help="跳过几何法左右判别")
    ap.add_argument("--skip-mirror", action="store_true", help="跳过 c2 镜像")
    ap.add_argument("--skip-reid", action="store_true", help="跳过 ReID 建库与评估")
    # 幂等
    ap.add_argument("--keep", action="store_true",
                    help="保留上一轮中间产物（默认清空以保证幂等）")
    args = ap.parse_args()

    # 阶段计数
    stages = []
    if not args.skip_extract: stages.append("extract")
    if not args.skip_side:    stages.append("side")
    if not args.skip_mirror:  stages.append("mirror")
    if not args.skip_reid:    stages.append("reid")
    total = len(stages)
    if total == 0:
        print("[错误] 所有阶段都被跳过，没有事情可做。")
        return 1

    # 输入校验
    if not args.skip_extract:
        if not args.videos:
            print("[错误] 需要 --videos 指定视频。若只想跑后续阶段，请加 --skip-extract。")
            return 1
        # 展开通配符，确认至少有一个视频
        found = []
        for pat in args.videos:
            found.extend(glob.glob(pat))
        if not found:
            print(f"[错误] --videos 没有匹配到任何文件：{args.videos}")
            print("       请检查路径（Windows 下含空格/中文目录记得整体加引号）。")
            return 1
        print(f"[输入] 匹配到 {len(found)} 个视频：")
        for v in found[:10]:
            print(f"       - {v}")
        if len(found) > 10:
            print(f"       ... 其余 {len(found) - 10} 个")

    print(f"\n框架根目录 : {FRAMEWORK}")
    print(f"Python 解释器: {PY}")
    print(f"将执行阶段  : {' -> '.join(stages)}")

    # 幂等清理（只在要重新抽取时清，避免误删只跑 reid 的场景）
    if not args.keep and not args.skip_extract:
        clean_intermediates()

    idx = 0
    metrics_text = None

    # ---------------- 阶段1：YOLO 抽帧裁剪 ----------------
    if not args.skip_extract:
        idx += 1
        banner(idx, total, "YOLO 检测 + 高置信筛帧 + 裁剪")
        cmd = [PY, VIDEO_EXTRACTOR, "--videos", *args.videos,
               "--imgsz", str(args.imgsz), "--conf", str(args.conf),
               "--max-frames", str(args.max_frames)]
        if args.auto_id:
            cmd.append("--auto-id")
        if args.stride is not None:
            cmd += ["--stride", str(args.stride)]
        rc, _ = run_stage(cmd, "抽帧裁剪")
        if rc != 0:
            print("[中止] YOLO 抽帧失败，请检查上面的报错。")
            return rc
        n = len(os.listdir(ENHANCED_DIR)) if os.path.isdir(ENHANCED_DIR) else 0
        if n == 0:
            print("[中止] 抽帧后 enhanced/ 为空——可能 conf 太高或视频无合格侧面帧。\n"
                  "       建议：降低 --conf（如 0.3）重试，或确认 --imgsz 是否匹配分辨率。")
            return 1
        print(f"[结果] enhanced/ 共 {n} 张裁剪图")

    # ---------------- 阶段2：几何法判左右 ----------------
    if not args.skip_side:
        idx += 1
        banner(idx, total, "几何朝向法判左右 + 重命名 (c1/c2)")
        cmd = [PY, SIDE_GEOMETRY, "--apply"]
        if args.flip_map:
            cmd.append("--flip-map")
        rc, _ = run_stage(cmd, "左右判别")
        if rc != 0:
            print("[中止] 左右判别失败。")
            return rc

    # ---------------- 阶段3：c2 镜像 ----------------
    if not args.skip_mirror:
        idx += 1
        banner(idx, total, "c2 就地镜像成 head-left (对齐 TransReID)")
        cmd = [PY, MIRROR_C2, "--src", ENHANCED_DIR,
               "--inplace", "--force"]
        rc, _ = run_stage(cmd, "c2镜像")
        if rc != 0:
            print("[中止] c2 镜像失败。")
            return rc

    # ---------------- 阶段4：ReID 建库 + 评估 ----------------
    if not args.skip_reid:
        idx += 1
        banner(idx, total, "TransReID 提特征 + 建库 + 交叉视角评估")
        cmd = [PY, REID_PIPELINE, "--build", "--eval"]
        rc, metrics_text = run_stage(cmd, "ReID评估", capture=True)
        if rc != 0:
            print("[中止] ReID 评估失败。")
            return rc

    # ---------------- 最终汇总 ----------------
    print("\n" + "#" * 64)
    print("# 全流程完成")
    print("#" * 64)
    print(f"裁剪图目录 : {os.path.relpath(ENHANCED_DIR, FRAMEWORK)}")
    print(f"Gallery 库 : {os.path.relpath(DB_PATH, FRAMEWORK)}")
    metrics = _extract_metrics(metrics_text)
    if metrics:
        print("\n=== 准确度汇总 ===")
        for block, kv in metrics.items():
            vals = "  ".join(f"{k}={v}" for k, v in kv.items())
            print(f"  [{block}] {vals}")
    elif not args.skip_reid:
        print("\n（未能从输出解析到准确度，请查看上面 ReID 阶段的评估打印）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
