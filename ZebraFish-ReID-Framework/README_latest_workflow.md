# ZebraFish-ReID-Framework 最新工作流程说明

本文档记录当前项目中“照片建库 + 视频抽帧查询 + 聚合身份向量评估”的最新可复现实验流程。原始 `README.md` 已保留不动，本文件只补充当前 24 条鱼照片库与游动视频评估流程。

---

## 1. 当前目标

使用每条鱼的标准侧面照片建立 ReID 照片库，再从每条鱼的游动视频中抽取清晰、完整、尽量水平的侧面帧作为 Query，最后评估视频帧与照片库的匹配效果。

核心原则：

- 照片库来自 `database/photos_processed`
- 视频 Query 来自 `database/video_queries/enhanced`
- 左侧面记为 `c1`，右侧面记为 `c2`
- 不确定侧面的帧保留为 `s-only`，评估时跳过
- `enhanced` 中的 `c2` 会被水平镜像后送入 ReID，`raw` 保留原方向用于人工核对
- 评估重点看聚合身份向量，尤其是同侧面聚合身份向量

---

## 2. 关键目录

```text
ZebraFish-ReID-Framework/
├── database/
│   ├── photos/                         # 原始导入的标准照片
│   ├── photos_processed/               # 已处理好的建库照片，当前照片库输入
│   ├── photos_gallery.db               # 由 photos_processed 构建的照片特征库
│   └── video_queries/
│       ├── raw/                         # 当前正式视频帧原始裁剪，保留原方向
│       ├── enhanced/                    # 当前正式视频帧增强图，送入 ReID 查询
│       ├── rejected/                    # 非鱼帧/低质量帧/不适合 ReID 的帧
│       ├── fish10_extra/                # 10 号鱼补抽帧的临时记录
│       ├── relabel_work/                # 重新判左右侧时的临时工作区
│       ├── side_geom_backup/            # 旧版几何侧面判定前的备份
│       ├── robust_side_backup/          # 新版鲁棒侧面判定前的备份
│       └── official_backup/             # 正式 raw/enhanced 被替换前的整体备份
├── models/
│   ├── transformer_20.pth               # TransReID 权重
│   └── yolov8_zebrafish.pt              # YOLO 斑马鱼检测权重
└── tools/
    ├── import_photos_to_database.py
    ├── process_database_photos.py
    ├── extract_videos_fast_seek.py
    ├── filter_video_query_frames.py
    ├── rebuild_video_query_from_raw.py
    ├── robust_side_label_video_queries.py
    ├── mirror_c2.py
    └── evaluate_video_against_photo_gallery.py
```

---

## 3. 命名规范

项目沿用 Market1501/ReID 风格命名：

```text
NNNN_cXsY_ZZZZ.png
```

- `NNNN`：鱼的编号，例如 `0001`、`0024`
- `cX`：侧面编号，`c1` 为左侧面，`c2` 为右侧面
- `sY`：身体区域，本项目当前固定为 `s1`，表示整鱼侧面图
- `ZZZZ`：该鱼对应侧面/帧序号

视频抽帧初始阶段会先生成：

```text
NNNN_s1_ZZZZ.png
```

这表示还没有可靠判定左右侧。经过 `robust_side_label_video_queries.py` 后，能判定的帧会改名为 `NNNN_c1s1_ZZZZ.png` 或 `NNNN_c2s1_ZZZZ.png`；不能可靠判定的帧继续保留 `s-only`，后续评估跳过。

---

## 4. 当前完整流程

### 4.1 导入标准照片

如果需要从外部照片目录导入到项目照片库，使用：

```powershell
D:\Anaconda3\envs\transreid_zf\python.exe tools\import_photos_to_database.py --apply
```

该脚本支持：

- 普通图片格式
- HEIC/HEIF
- LIVP
- 按鱼编号文件夹导入
- 按 `L` / `R` 两侧文件夹生成 `c1` / `c2` 命名

当前项目已经完成导入与处理，最新建库输入为：

```text
database/photos_processed
```

### 4.2 处理标准照片

如果重新导入了照片，需要将 `database/photos` 处理为适合建库的标准侧面图：

```powershell
D:\Anaconda3\envs\transreid_zf\python.exe tools\process_database_photos.py --apply
```

处理目标：

- 最大程度保留鱼体主体
- 鱼体尽量水平
- 鱼头朝左
- 轻度增强，便于后续特征提取

输出：

```text
database/photos_processed
database/photos_rejected
database/photos_processed_report.jsonl
```

### 4.3 构建照片特征库

照片库已由 `database/photos_processed` 构建，当前数据库文件为：

```text
database/photos_gallery.db
```

如果需要重新建库，使用项目 ReID pipeline 的建库能力。当前代码已支持读取 `.png` / `.jpg` / `.jpeg`：

```powershell
D:\Anaconda3\envs\transreid_zf\python.exe src\reid\pipeline.py --build
```

照片库中会保存：

- 单张照片特征
- 每条鱼每侧面的聚合特征
- 每条鱼身份级聚合特征

---

## 5. 视频查询帧生成流程

### 5.1 从视频快速抽帧

从外部视频目录中按鱼编号子文件夹抽帧：

```powershell
D:\Anaconda3\envs\transreid_zf\python.exe tools\extract_videos_fast_seek.py `
  --src "C:\Users\JiangYao\Desktop\26_Medical\7.14斑马鱼照片+视频" `
  --fish-start 1 `
  --fish-end 24 `
  --samples-per-video 160 `
  --max-frames 60 `
  --conf 0.3 `
  --imgsz 1920 `
  --min-aspect 1.2 `
  --max-aspect 12 `
  --dedup 0.08 `
  --margin 0.24
```

输出：

```text
database/video_queries/raw
database/video_queries/enhanced
database/video_queries/video_crop_report.jsonl
database/video_queries/video_extract_summary.json
```

说明：

- `raw` 是原始裁剪图
- `enhanced` 是 CLAHE 增强图
- 此阶段只负责检测鱼和抽取候选侧面帧，不负责判定左右侧

### 5.2 筛掉非鱼帧

YOLO 偶尔会把缸边、背景、人脸等误检为鱼。使用后处理脚本筛掉：

```powershell
D:\Anaconda3\envs\transreid_zf\python.exe tools\filter_video_query_frames.py --apply
```

输出：

```text
database/video_queries/rejected
database/video_queries/filter_report.jsonl
```

被筛掉的帧不会删除，而是移动到 `rejected`，便于追溯。

### 5.3 从 raw 重建待标注查询集

因为左右判定必须基于未镜像、未污染的原始方向图，所以重新标注前先从 `raw` 重建一套干净的 `s-only` 工作目录：

```powershell
D:\Anaconda3\envs\transreid_zf\python.exe tools\rebuild_video_query_from_raw.py
```

输出：

```text
database/video_queries/relabel_work/raw_sonly
database/video_queries/relabel_work/enhanced_sonly
```

### 5.4 鲁棒判定左右侧面

使用新版侧面判定脚本：

```powershell
D:\Anaconda3\envs\transreid_zf\python.exe tools\robust_side_label_video_queries.py `
  --raw-dir database\video_queries\relabel_work\raw_sonly `
  --enhanced-dir database\video_queries\relabel_work\enhanced_sonly `
  --decision-dir database\video_queries\relabel_work\raw_sonly `
  --report database\video_queries\robust_side_report.jsonl `
  --min-head-conf 0.35 `
  --apply
```

该脚本使用：

- 彩色鱼体分割
- PCA 主轴估计鱼体方向
- 多阈值眼睛候选检测
- 冲突候选保护

如果判定证据不足，帧保留为 `NNNN_s1_ZZZZ.png`，不会强行归为 `c1` 或 `c2`。

### 5.5 镜像 c2 enhanced 图

为了对齐当前 TransReID 权重的输入约定，`enhanced` 中的 `c2` 需要水平镜像：

```powershell
D:\Anaconda3\envs\transreid_zf\python.exe tools\mirror_c2.py `
  --src database\video_queries\relabel_work\enhanced_sonly `
  --inplace `
  --force
```

注意：

- 只镜像 `enhanced`
- 不镜像 `raw`
- `raw` 保留原方向，方便人工核对左右侧面是否正确

### 5.6 切换正式查询目录

确认新标注结果可用后，将旧正式目录备份，再将 `relabel_work` 切换为正式目录：

```powershell
$ts=Get-Date -Format 'yyyyMMdd_HHmmss'
$backup=Join-Path 'database\video_queries\official_backup' $ts
New-Item -ItemType Directory -Path $backup | Out-Null
Move-Item -LiteralPath 'database\video_queries\raw' -Destination (Join-Path $backup 'raw')
Move-Item -LiteralPath 'database\video_queries\enhanced' -Destination (Join-Path $backup 'enhanced')
Move-Item -LiteralPath 'database\video_queries\relabel_work\raw_sonly' -Destination 'database\video_queries\raw'
Move-Item -LiteralPath 'database\video_queries\relabel_work\enhanced_sonly' -Destination 'database\video_queries\enhanced'
```

---

## 6. 查询与评估

使用修正后的 `database/video_queries/enhanced` 查询照片库：

```powershell
D:\Anaconda3\envs\transreid_zf\python.exe tools\evaluate_video_against_photo_gallery.py `
  --query-dir database\video_queries\enhanced `
  --db database\photos_gallery.db `
  --topk 5 `
  --out database\video_queries\video_vs_photo_identity_results_robust_side.csv
```

评估输出四类指标：

1. `Photo Image Gallery`
   视频帧与照片库中每一张单图特征比较。

2. `Photo Identity Gallery`
   视频帧与每条鱼的聚合身份向量比较。

3. `Same-Side Photo Image Gallery`
   视频帧只与同侧面的单图照片比较。

4. `Same-Side Photo Identity Gallery`
   视频帧只与同侧面的聚合身份向量比较。

当前更推荐关注：

```text
Same-Side Photo Identity Gallery
```

原因是它同时满足：

- 使用聚合身份向量，降低单张照片偶然误差
- 限制同侧面比较，避免左/右侧面混淆

---

## 7. 当前最新结果

基于修正后的鲁棒侧面判定流程，当前查询集统计为：

```text
总视频查询帧: 533
有效 c1/c2 查询帧: 472
s-only 跳过帧: 61
24 条鱼均至少包含 c1 和 c2 查询帧
```

最新评估结果：

```text
Photo Image Gallery
Rank-1: 6.14%
Rank-5: 14.19%
mAP   : 12.37%

Photo Identity Gallery
Rank-1: 3.81%
Rank-5: 31.36%
mAP   : 19.47%

Same-Side Photo Image Gallery
Rank-1: 5.93%
Rank-5: 16.74%
mAP   : 16.43%

Same-Side Photo Identity Gallery
Rank-1: 5.30%
Rank-5: 36.23%
mAP   : 20.97%
```

结果文件：

```text
database/video_queries/video_vs_photo_identity_results_robust_side.csv
```

---

## 8. 新增脚本说明

### `tools/import_photos_to_database.py`

将外部按鱼编号分类的照片导入到 `database/photos`，并按项目命名规范保存。支持 HEIC、HEIF、LIVP 和普通图片格式。

### `tools/process_database_photos.py`

处理 `database/photos` 中的标准照片，尽量保留完整鱼体、旋转到水平、鱼头朝左，并输出到 `database/photos_processed`。

### `tools/extract_videos_fast_seek.py`

从每条鱼的视频中快速跳采样抽帧，使用 YOLO 检测鱼体，输出 `raw` 和 `enhanced` 查询候选帧。

### `tools/filter_video_query_frames.py`

筛掉明显不是鱼的误检帧，以及姿态不适合 ReID 的帧。筛掉的文件会移动到 `rejected`，不会直接删除。

### `tools/rebuild_video_query_from_raw.py`

从 `raw` 原始方向裁剪图重建一套干净的 `s-only` 查询集，用于重新判左右侧面。

### `tools/robust_side_label_video_queries.py`

当前推荐的左右侧面判定脚本。比旧版 `side_geometry_label.py` 更保守，能减少把右侧面误判为左侧面的情况。

### `tools/mirror_c2.py`

将 `enhanced` 中的 `c2` 图水平镜像，使送入 ReID 模型的鱼尽量统一为鱼头朝左。不会处理 `raw`。

### `tools/evaluate_video_against_photo_gallery.py`

使用视频 Query 帧查询照片库，并输出单图、聚合身份向量、同侧单图、同侧聚合身份向量四类评估指标。

---

## 9. 注意事项

- 不要直接删除 `raw`、`enhanced`、`official_backup`、`rejected`，它们分别对应当前结果、备份和可追溯的拒绝样本。
- 如果要重新跑完整流程，建议先备份当前 `database/video_queries`。
- 左右侧面判定宁可保守，不确定就保留 `s-only`，不要强行归类。
- 人工核对侧面时，请看 `raw`，不要看已镜像过的 `enhanced`。
- 后续评估建议优先看 `Same-Side Photo Identity Gallery`。
