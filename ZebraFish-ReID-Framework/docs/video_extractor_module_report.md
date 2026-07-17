# 视频帧提取 + 单类 YOLO 检测 + 几何法判左右 — 模块说明

> 整理日期：2026-07-17
> 范围：本文件只讲 `video_extractor.py` 与 `side_geometry_label.py` 两个预处理模块。
> 完整端到端流程、目录结构、模型事实见根目录 `README.md`；左右判别原理见 `docs/鱼体左右侧判别_文献与方法.md`。

---

## 一、核心设计（务必先读）

- **YOLO 在本框架只做单类检测（`zebrafish`）**：回答"框里是不是一条鱼、鱼在哪"，**不回答**"这是左面还是右面"。
- **左右（c1/c2）由后处理 `tools/side_geometry_label.py`（几何朝向法）判定**，与 YOLO 解耦。
  理由：标准检测型 YOLO 对水平翻转近似不变，学不会左右这种"手性（镜像对称）"问题（详见 `docs/鱼体左右侧判别_文献与方法.md`）。
- 因此**抽帧阶段产出的文件名不带 c1/c2**，固定为 `NNNN_s1_ZZZZ.png`；`c1/c2` 在几何法那一步才第一次写进文件名。

---

## 二、端到端数据流

```
视频 (NNNN.mp4，文件名第一个 4 位数字 = 鱼ID)
  │
  ▼  video_extractor.py
  逐帧读 → 单类 YOLO 筛鱼(conf≥conf_side) → 清晰度 → 去重 → 裁剪
  命名 NNNN_s1_ZZZZ.png（无 c1/c2）
  产物：raw/video_crops/(原图 s1) + processed/enhanced/(增强图 s1)
        + processed/video_crop_report.jsonl（cam 字段暂记 null）
  │
  ▼  side_geometry_label.py --apply
  几何法判头向 → NNNN_c1s1_* / NNNN_c2s1_*（拿不准的保持 s1，即 unknown）
  同步更新 video_crop_report.jsonl 的 cam 字段
  │
  ▼  mirror_c2.py --src enhanced --inplace
  仅把 enhanced/ 内的 c2 镜像成 head-left（raw/ 里的 s1 原始裁剪不动，供人工核对）
  │
  ▼  pipeline.py --build --eval
  只读 enhanced/，只收带 c1/c2 的文件；s1 是 unknown 的有意遗留，不参与 ReID
```

---

## 三、video_extractor.py 参数与三段式帧选取

配置见 `configs/video_extraction.json`：

| 参数 | 默认 | 说明 |
|------|------|------|
| `conf_side` | 0.7 | 单类 YOLO 的"是不是鱼"置信度阈值。**调高只挡误检/模糊检测，不保证姿态完整** |
| `imgsz` | 1920 | 4K 视频必须 1920；640 会把鱼缩没导致 0 检出 |
| `blur_thresh` | 3.0 | 在**鱼裁剪图**上算的 Laplacian 方差下限（整帧算会全被杀，因为 4K 整帧对小鱼永远"清晰"） |
| `dedup_mae` | 0.06 | 与上一已选帧的灰度 MAE 上限，超过才保留（抑制近重复帧） |
| `sample_stride` | 8 | 时间覆盖步长（每 8 帧取一个候选） |
| `max_frames_per_fish` | 120 | 每条鱼封顶帧数（c1/c2 共用） |
| `margin_ratio` | 0.12 | 裁剪外扩比例 |

`iter_selected` 三段式逻辑：

```python
for each frame (每隔 sample_stride 帧取一个候选):
    det = detect_fish(frame, model, conf_side=0.7, imgsz=1920)
        # 单类 YOLO：只接受 zebrafish 检测，conf>=conf_side 取置信最高框
        # 返回 (box, conf)；否则返回 None（该帧舍去，左右由几何法判定）
    if det is None:                 # 非鱼 / 置信不足 → 丢
        continue
    crop = crop_with_margin(frame, box, margin_ratio=0.12)
    blur = compute_blur_score(crop)  # 在【裁剪鱼体】上算清晰度
    if blur < blur_thresh:           # 太糊 → 丢
        continue
    if 与上一帧 crop 的灰度 MAE < dedup_mae:  # 太像 → 丢
        continue
    yield (crop, conf, blur)         # 注意：这里没有 cam，左右待定
```

命名：`NNNN_s1_ZZZZ.png`
- `s1` = 全身段固定代号（每张图都是整条鱼的全身裁剪，本数据集中不存在 `s2` 系列）
- 文件名**不含 `cX`**（左右待几何法判定）；`ZZZZ` 为该鱼 ID 的全局顺序号
- `cam` 字段在 `video_crop_report.jsonl` 中暂记 `null`，待几何法填 `c1`/`c2`

---

## 四、side_geometry_label.py（几何法判左右）

- **输入**：`enhanced/` 里的 `NNNN_s1_ZZZZ.png`（也可能有上一轮留下的 `c1/c2` 文件，用于纠正）。
- **步骤**：抠鱼体 mask → PCA 主轴 → 瞳孔暗斑定位头端（回退：质量分布偏度）→ 头朝左=`c1`、头朝右=`c2`（可用 `--flip-map` 整体翻转映射）。
- **两种置信度务必分清**（这是最容易混淆的点）：
  1. `conf_side`：抽帧阶段 YOLO 的"是不是鱼"置信度。**已在 `video_extractor` 过滤**——低于它的帧在上一阶段就被丢，根本到不了这里。
  2. 本脚本算出的 `confidence`：几何法"头朝左还是右有多确定"。**低于 `--min-head-conf`（默认 0.25）直接判 `unknown`，不打 `c1/c2`**——拿不准的帧保持 `s1`（unknown），留给人工或后续更强的分类器，绝不强行打标污染 ReID。
- 改名时同步更新 `video_crop_report.jsonl` 的 `cam` 字段。
- **纠正模式**：若文件已带 `c1/c2` 且新判头向与原 `c` 相反，则改名纠正；判为 unknown 时保持原样。

> 因为 unknown 帧会以 `s1` 形式留下来，`pipeline.py` 见到 `s1` 属于**正常有意遗留**，只会静默跳过，不会报"忘了跑 B"。只有当目录里**一张 c1/c2 都没有**时，才说明几何法把全部判成了 unknown（可调低 `--min-head-conf`）。

---

## 五、已知问题 / 诊断（实测）

### 问题1：裁出"不是鱼"的图（误检 / false positive）
- 根因：训练数据少 + 阈值。
- 缓解：把 `conf_side` 调到 0.7~0.9（现已默认 0.7）；增加标注、覆盖更多鱼；`imgsz` 必须 1920。
- 注意：`conf_side` 调高**只能挡误检**，不会自动只保留"完整侧面、无遮挡"的帧（见问题3）。

### 问题2：左右判错 → 现已不是 YOLO 的问题
- 左右**不**由 YOLO 判定，"YOLO 左右判错"这个旧问题已不存在（旧版双类设计已废弃）。
- 现在是**几何法判头向可能不准**：斑马鱼体侧有深黑纵纹，最暗像素常落在条纹而非眼睛，导致 `reason='eye_blob'` 的头向可能错。
- 缓解：直接 `python tools/side_geometry_label.py` 看统计（该脚本不加 `--apply` 即为预览，没有 `--dry-run` 参数），重点看 `reason='eye_blob'` 且最终归 `unknown` 的比例；若太高，调低 `--min-head-conf` 放一点，或后续换成关键点 / 轻量左右分类器（见 `docs/鱼体左右侧判别_文献与方法.md` 方案②）。

### 问题3：如何只保留"完整侧面"
- `conf_side` 只过滤"是不是鱼"，**不保证**姿态完整（无遮挡、全身在框内）。
- 若需要只留完整侧面，应另加三道闸：① 长宽比下限（太竖→丢）；② YOLO 框不贴图像边缘（鱼被裁出框→丢）；③ 鱼体占框面积比例下限。
- **当前版本未实现这三道闸**（按需求暂不添加），后续如需再加。

---

## 六、与旧版的区别（防混淆）

| 项 | 旧版（已废弃） | 现版 |
|----|---------------|------|
| YOLO 类别 | 双类 `zebraFish_l` / `zebraFish_r` | 单类 `zebrafish` |
| c1/c2 来源 | YOLO 类别直接定 | 后处理几何法定 |
| 抽帧命名 | `NNNN_cXs1_ZZZZ.png` | `NNNN_s1_ZZZZ.png`（无 c） |
| c2 镜像 | 同步镜像 `raw` + `enhanced` | 只镜像 `enhanced`，`raw` 保留 `s1` 供人工核对 |
| 低置信左右 | （旧版无此概念） | 几何法低置信 → `unknown`，保持 `s1` 不打标 |

---

## 七、训练侧（在框架外，不进最终产品）

- 用 X-AnyLabeling 给视频帧画**单个类 `zebrafish`** 的矩形框（不区分左右）。
- `json_to_yolo.py` 已改为单类（`zebrafish` → class 0），按鱼 ID 分层划分 train/val。
- 训 `yolov8s`（已禁用左右翻转增强 `fliplr`，防左右标签被破坏）→ 导出 `best.pt` 放入 `models/yolov8_zebrafish.pt`。
- `video_extractor.py` 只加载 `.pt` 做推理，不训练。
