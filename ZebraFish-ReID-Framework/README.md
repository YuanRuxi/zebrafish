# ZebraFish-ReID-Framework — 斑马鱼视频重识别框架

将 **YOLO 检测** 与 **TransReID 重识别** 串成一个统一、自包含的推理框架：输入侧面游动的斑马鱼视频帧，输出鱼的身份 ID。

> **不重新训练。** 直接使用已有黄金权重 `transformer_20.pth`（JPM=True 版本），仅做推理。

---

## 1. 设计哲学（方案2）

| 模块 | 职责 | 是否接触 TransReID 内部？ |
|------|------|--------------------------|
| `preprocessing/` | **视频→鱼图**（`video_extractor.py`）+ 质量筛选 + CLAHE 对比度增强，输出**原分辨率**增强 PNG | 否 |
| `reid/` | **严格复刻** TransReID 官方推理：尺寸缩放 `[256,288]` + ImageNet 归一化 + 特征提取 + 建库/匹配 | 是 |

**关键决策**：尺寸缩放与归一化**不放在预处理里**，而由 `reid` 模块在推理时按官方脚本（`Dynamic_Domain_Eval.py`）原样复刻。
好处：预处理输出的是“给人看、给任何模型都能吃”的通用增强图，TransReID 的域假设（输入尺寸、均值方差）始终与训练保持一致，避免自己实现的预处理与训练分布偏离导致特征失效。

---

## 2. 目录结构

```
ZebraFish-ReID-Framework/
├── run_pipeline.py                   # ★一键启动全流程（抽帧→判左右→镜像→重识别→评估）
├── configs/
│   ├── vit_transreid_zf_v2.yml      # 黄金权重匹配配置（JPM=True, 输入 [256,288]）
│   ├── quality_thresholds.json       # 自适应模糊阈值（blur_thresh≈5.91, p10）
│   └── video_extraction.json         # 视频抽取参数（YOLO 路径/步长/去冗余/裁剪边距…）
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── quality.py                # 模糊(Laplacian方差)/鱼体占比 评估
│   │   ├── enhance.py                # CLAHE 增强（LAB 空间 L 通道）
│   │   ├── run_preprocessing.py      # 主预处理流程（已有裁剪图：筛选→增强→写原分辨率PNG）
│   │   └── video_extractor.py        # 视频→鱼图（命名解析/三段式帧选取/YOLO裁剪）
│   └── reid/
│       ├── feature_extractor.py      # TransReID 特征提取器（严格复刻官方推理）
│       └── pipeline.py               # 批量提取 / 质量加权聚合 / 建库 / 查询 / 交叉视角评估
├── tools/
│   ├── side_geometry_label.py       # ★几何朝向法判鱼体左右侧并重命名 c1/c2（零训练）
│   ├── mirror_c2.py                 # ★c2 水平镜像成 head-left（对齐 TransReID 训练约定）
│   └── diagnose_load.py             # 权重加载诊断（可选调试）
├── third_party/
│   └── TransReID/                    # vendored 官方源码（config / model / loss / utils）
├── models/
│   ├── transformer_20.pth           # 403MB 黄金权重（JPM=True, 20类, SIE_CAMERA）
│   └── yolov8_zebrafish.pt          # （必需）YOLO 单类检测权重（只检测 zebrafish，不判左右），放此处即可被自动发现
├── data/
│   ├── raw/video_crops/              # 视频抽取的【原始】鱼体裁剪（原分辨率 PNG）
│   └── processed/
│       ├── enhanced/                 # 增强图输出（原分辨率 PNG，喂给 ReID）
│       ├── quality_report.jsonl      # 逐张质检流水（图像预处理来源）
│       ├── video_crop_report.jsonl   # 逐帧质检流水（视频抽取来源，同 schema）
│       ├── _side_geom_report.jsonl   # 几何法左右判别逐张记录
│       ├── _side_geom_backup/        # 几何法改名前的原文件备份
│       └── _backup_20260715/         # 原始 2484 张增强图备份（勿删）
├── database/
│   └── gallery.db                    # Gallery 数据库（SQLite，运行后自动生成）
└── docs/
    ├── dataset_statistics.md         # 数据集统计（仅按文件名，未读像素）
    ├── 鱼体左右侧判别_文献与方法.md   # 左右侧判别文献综述与方案
    └── references.md                 # 参考文献（含帧选取 + YOLO 相关文献）
```

---

## 3. 环境依赖

- Python 3.x + `torch`（CPU 即可，本框架在 `torch 2.12.1+cpu` 验证通过）
- `torchvision`、`Pillow`、`numpy`、`opencv-python`
- **`ultralytics`**（仅 `video_extractor.py` 需要，用于加载 YOLO 权重做检测裁剪；其余模块不需要）
- **OpenCV PNG 解码坑**：本机 OpenCV 无法正确解码这批 PNG，框架内一律改用 **PIL** 读写（见 `quality.load_bgr`）。视频读取走 `cv2.VideoCapture`（ffmpeg 后端），与此坑无关。

---

## 4. 使用流程

```bash
# ── 路径 A：原始素材是【已裁剪好的鱼图】 ──────────────────────────
# 0a. 预处理：质量筛选 + CLAHE 增强（输出到 data/processed/enhanced/）
python src/preprocessing/run_preprocessing.py --dry-run --limit 50   # 先看拒绝率
python src/preprocessing/run_preprocessing.py                        # 全量 2760 张

# ── 路径 B：原始素材是【完整斑马鱼视频（含非侧面）】 ─────────────────
# 0b. 视频 → 鱼图（命名解析 + 三段式帧选取 + 单类YOLO检测/裁剪 + 增强）
#     视频命名只需含 4 位鱼ID：NNNN.ext（如 0001.mp4 / 0001_tank3.mp4）
#     ★ 本步骤只检测鱼、不判左右；左右 c1/c2 由 0c 几何法判定并写回文件名
python src/preprocessing/video_extractor.py --videos "原始视频/*.mp4" --dry-run  # 先预览匹配
python src/preprocessing/video_extractor.py --videos "原始视频/*.mp4"            # 正式抽取
#     → 产出 data/raw/video_crops/(原图) 与 data/processed/enhanced/(增强图)，命名 NNNN_s1_ZZZZ.png

# 0c. 几何朝向法判左右并改文件名（c1/c2）— 必须在本步之后、构建 Gallery 之前
python tools/side_geometry_label.py            # 先预览统计（不加 --apply 即为预览，只打印、不改名）
python tools/side_geometry_label.py --apply     # 实际改名 → NNNN_c1s1_*.png / NNNN_c2s1_*.png

# 1. 构建 Gallery + 交叉视角评估（Rank-1 / Rank-5 / mAP）
python src/reid/pipeline.py --build --eval

# 2. 查询单张图，返回 Top-K 鱼 ID + 余弦相似度
python src/reid/pipeline.py --query data/processed/enhanced/0001_c1s1_0001.png --topk 5
```

> 特征已 L2 归一化，余弦相似度 = 向量点积，取值 `[-1, 1]`，越大越可能是同一条鱼。

---

## 4.1 ★一键全流程（推荐）：`run_pipeline.py`

一条命令跑完整条链路，无需手动串联四个脚本：

```
视频 ──▶ ① YOLO 检测 + 高置信筛帧 + 裁剪 (video_extractor.py)
     ──▶ ② 几何朝向法判左右并改名 c1/c2 (side_geometry_label.py)
     ──▶ ③ c2 就地镜像成 head-left (mirror_c2.py --inplace)
     ──▶ ④ TransReID 提特征 + 建库 + 评估 (pipeline.py --build --eval)
```

### 命令示例

```bash
cd "D:\YUANRUXI0124\2026论文\（新）单条鱼数据集\ZebraFish-ReID-Framework"

# 小样验证：几个新视频（时间戳/无鱼号命名 → 加 --auto-id 按顺序编 0001..）
D:/anaconda3/python.exe run_pipeline.py \
    --videos "D:/.../视频集/*.mov" --auto-id --imgsz 1920 --conf 0.7 --max-frames 40

# 视频名自带 4 位鱼号（如 0001.mp4）→ 去掉 --auto-id
D:/anaconda3/python.exe run_pipeline.py --videos "data/videos/*.mp4" --conf 0.7

# 只重跑 ReID（前面产物已就绪，不清空）
D:/anaconda3/python.exe run_pipeline.py --skip-extract --skip-side --skip-mirror --keep
```

### 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--videos` | 必填 | 视频路径/通配符（可多个）。含空格/中文目录记得整体加引号 |
| `--auto-id` | 关 | 视频名不含 4 位鱼号时，按 sorted 顺序自动编 `0001..` |
| `--imgsz` | 1920 | YOLO 推理尺寸；**4K 视频必须 1920**，否则鱼被缩没检不出 |
| `--conf` | 0.7 | YOLO 检测置信度阈值（是否为斑马鱼，默认 0.7；弱模型可调低诊断） |
| `--max-frames` | 40 | 每条鱼最多保留帧数（小样测试用） |
| `--flip-map` | 关 | 若发现左右整体判反，翻转 头向→c1/c2 映射 |
| `--skip-extract/side/mirror/reid` | — | 按需跳过某一阶段 |
| `--keep` | 关 | 保留上一轮中间产物（默认清空以保证幂等） |

### 幂等性（重要）

- 默认每次运行**先清空** `data/raw/video_crops/`、`data/processed/enhanced/`、
  `video_crop_report.jsonl`、`_side_geom_report.jsonl`、`database/gallery.db`，
  再重新抽取。这可避免 **c2 被重复镜像**（镜像两次会翻回原始右面）或旧图混入。
- `_backup_20260715/`（原始 2484 张）与 `_side_geom_backup/` **不会被清理**。
- 若只想重跑后续阶段（不重新抽帧），务必加 `--keep` 并配合 `--skip-*`。

### 为什么需要「③ c2 镜像」

`video_extractor.py` 抽出的 c2 是**原始右面**（鱼头朝向任意，命名由 `side_geometry_label.py` 几何法判定），而黄金权重 `transformer_20.pth`
训练/验证时的 c2 是**已水平镜像、鱼头朝左**的。若不镜像直接喂 ReID，
c1↔c2 交叉视角评估会系统性失真。`run_pipeline.py` 已自动在建库前完成这一步。

> ⚠️ **镜像只作用于 `enhanced/` 内的 c2 文件**：`mirror_c2.py` **不会**动 `raw/video_crops/` 里的原始裁剪（仍以 `s1` 命名，无 c1/c2）。这方便你人工核对"右面检测得对不对"——可在跑镜像前先看 `enhanced/` 里 `NNNN_c2s1_*.png` 是否真的是右面，或用 `raw/` 原始裁剪对照。

---

## 5. 模型关键事实（已验证）

| 项 | 值 | 验证方式 |
|----|----|----------|
| 主干 | ViT-Base (`vit_base_patch16_224_TransReID`) | 配置文件 |
| JPM | **True** → 特征 = 768(全局) + 4×768(局部) = **3840 维** | 端到端前向输出 `(3840,)` |
| SIE_CAMERA / SIE_VIEW | True / False（camera_num=2） | 权重含 SIE 相机嵌入 |
| 输入尺寸 | **`[256, 288]`** | 反推：`base.pos_embed=(1,289,768)` → 288 patch + 1 CLS |
| 推理变换 | `Resize([256,288]) → ToTensor → Normalize(ImageNet)` | 复刻官方脚本 |
| 权重载入覆盖率 | **100% (211/211)** | `feature_extractor._load_weights` 形状匹配 |
| 测试特征 | `NECK_FEAT=after`, `FEAT_NORM=yes`, L2 归一化 | 配置文件 |

> **尺寸反推说明**：黄金权重 `base.pos_embed` 形状为 `(1, 289, 768)`，289 = 288 patch + 1 CLS token，
> 由 `num_x = (W-16)//16+1`、`num_y=(H-16)//16+1` 反解得训练/推理尺寸为 `[256, 288]`。
> 选错尺寸会导致 `pos_embed` 无法载入（已在 `_load_weights` 中加断言强制报错）。

---

## 6. 数据集命名规则

`NNNN_cXsY_ZZZZ.png`
- `NNNN` = 鱼 ID（0001–0031，共 31 条不同鱼）
- `cX` = 侧视图（`c1`=左侧面，`c2`=右侧面）
- `sY` = 部位段（本数据集中 `s1`=**全身**，即每张图都是鱼的整张全身裁剪；不存在 `s2` 系列）
- `ZZZZ` = 帧序列号

数据集统计详见 `docs/dataset_statistics.md`（仅按文件名统计，未读取像素）。

---

## 6.1 视频命名规则与抽取策略（模块 `video_extractor.py`）

> **设计**：用户直接喂给框架**每条鱼的完整视频**（视频里既有侧面、也有非侧面，左右面会交替出现）。
> 框架用 **单类 YOLO 模型（zebrafish）**逐帧检测，**仅保留被判定为斑马鱼（conf ≥ `conf_side`）的帧**；
> 非鱼（模糊/遮挡/空背景）→ 置信度不足 → 直接舍去。
> **⚠️ 本步骤只检测鱼、不判左右**；左右（c1/c2）由后处理 `tools/side_geometry_label.py`（几何朝向法）判定并写回文件名。

### 视频文件命名（输入）
`NNNN.ext` 或 `NNNN_任意描述.ext`
- `NNNN` = 鱼 ID（4 位，如 `0001`）；框架取文件名中**第一个 4 位数字组**作为鱼 ID
- 例：`0001.mp4`、`0001_tank3.mp4` 都解析为鱼 0001
- **侧面 `c1`/`c2` 不写在视频名里**，而是由后处理几何法决定

### 输出图片命名（本步骤不带 c1/c2）
`NNNN_s1_ZZZZ.png`
- `s1` **固定为 `s1`**（全身裁剪）
- **本步骤不含 `cX`**（左右待几何法判定）；`ZZZZ` 为 该鱼 ID 的全局顺序号
- `cam` 字段在 `video_crop_report.jsonl` 中暂记 `null`，待几何法填 `c1`/`c2`

### 三段式帧选取
1. **时间覆盖**：每隔 `sample_stride` 帧抽一个候选，覆盖整段视频的姿态/位置变化
2. **检测 + 清晰度**：单类 YOLO 检测 **conf ≥ `conf_side`（默认 0.7）** 才保留；同时 `Laplacian 方差 ≥ blur_thresh`
3. **去冗余**：与上一已选帧在 **鱼裁剪**上的灰度 MAE `> dedup_mae` 才保留，抑制相邻近重复帧

### YOLO 检测裁剪（仅推理，训练在框架外）
- **模型为单类**：只检测 `zebrafish`（不区分左右）。左右判定交给后处理几何法。
- 加载 `configs/video_extraction.json` 的 `yolo_model`（可 `--yolo` 覆盖；默认 `models/yolov8_zebrafish.pt`）
- 单鱼假设下取**置信度最高**的合格框，按 `margin_ratio` 外扩后裁剪
- **训练/标注不进最终产品**：重训只需在框架外用 X-AnyLabeling 标 **zebrafish** 矩形框 → 转 YOLO txt（单类）→ 训练 `yolov8s` → 导出 `best.pt` 放入 `models/` 即可

---

## 7. Gallery 与查询说明

- **建库**：每条鱼的增强图经特征提取后，按清晰度（`blur_score`）**质量加权聚合**为一条身份向量，写入 `database/gallery.db`。
- **查询**：默认 `image` 模式——查询图特征与 Gallery 中每张图逐一比较，取相似度最高的鱼（标准 Re-ID 检索）。也可用 `--mode identity` 直接与聚合身份向量比较。
- **交叉视角评估**：用一侧视角（如 c1）作 Gallery、另一侧（c2）作 Query，反向再做一次，度量 `Rank-1 / Rank-5 / mAP`——这正是 Re-ID 的核心场景（同一条鱼从不同侧面出现）。
