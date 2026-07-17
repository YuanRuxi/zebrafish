# 参考文献与引用说明

本框架（斑马鱼视频 Re-ID）在 **模型架构、前处理策略、评估协议** 三个层面参考了以下文献。
每篇文献均列出 **原文链接** 与 **本框架具体参考了它的哪些方面**。

> 说明：本框架**不重新训练** TransReID，而是直接复用其预训练权重 `transformer_20.pth`，
> 因此参考重点在于"如何严格复刻官方推理方式"与"前处理/评估的设计依据"。

---

## 参考文献列表（含链接）

| 编号 | 文献 | 类型 | 链接 |
|------|------|------|------|
| [1] | He et al., **TransReID: Transformer-Based Object Re-Identification**, ICCV 2021 | 模型架构 / 方法 | [CVF 论文页](https://openaccess.thecvf.com/content/ICCV2021/html/He_TransReID_Transformer-Based_Object_Re-Identification_ICCV_2021_paper.html) · [arXiv:2102.04378](https://arxiv.org/abs/2102.04378) · [DOI:10.1109/ICCV48922.2021.01474](https://doi.org/10.1109/iccv48922.2021.01474) · [官方代码](https://github.com/heshuting555/TransReID) |
| [2] | Cao et al., **Longitudinal Identification of Zebrafish Individuals by Deep Learning** (ESC-IDNet), 2025 | 应用领域 / 多视角思路 | [DOI:10.64898/2025.12.14.694189](https://dx.doi.org/10.64898/2025.12.14.694189) · [bioRxiv 全文](https://www.biorxiv.org/content/10.64898/2025.12.14.694189v1.full) |
| [3] | **Zebrafish identification with deep CNN and ViT architectures using a rolling training window**, Scientific Reports 15, 8580 (2025) | 鱼类识别实证 / 特征重要性 | [Nature 原文](https://www.nature.com/articles/s41598-025-86351-x) |
| [4] | CLAHE 技术及其在鱼类/水下图像中的应用 | 图像增强方法 | [CLAHE (Wikipedia)](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization) · [SiamFCA 水产跟踪](http://www.sciencedirect.com/science/article/pii/S0168169923009304) · [juvenile ayu 检测](https://www.jstage.jst.go.jp/article/jscejj/82/16/82_25-16107/_article/-char/en) |
| [5] | Zheng et al., **Scalable Person Re-identification: A Benchmark** (Market-1501), ICCV 2015 | 评估协议来源 | [CVF 论文页](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html) · [PDF](https://research.microsoft.com/en-us/um/people/jingdw/pubs/iccv15-reiddataset.pdf) · [arXiv:1505.02198](https://arxiv.org/abs/1505.02198) |
| [6] | **视频帧选取策略**（分段均匀采样 + 质量优选 + 去冗余）| 视频→图 的帧选取依据 | [视频行人 ReID 代表帧 (arXiv:1702.06294)](https://arxiv.org/abs/1702.06294) · [IUST PersonReId 质量优选 (arXiv:2412.18874)](https://arxiv.org/abs/2412.18874) |
| [7] | **野生动/畜关键帧提取（检测+模糊检测+聚类/去重）** | 无监督关键帧 / 去冗余依据 | [Kākā 鹦鹉 ReID 关键帧 (arXiv:2510.08775)](https://arxiv.org/abs/2510.08775) · [畜脸识别 SSIM+拉普拉斯去重 (CN114581948A)](http://patents.google.com/patent/CN114581948A/zh) |
| [8] | **斑马鱼 YOLO 单类检测（zebrafish）+ 后处理几何法判左右** | `video_extractor.py` 的 YOLO 检测裁剪依据（左右由 `side_geometry_label.py` 判定） | [ESC-YOLOv8-seg (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.04.28.649888v1.full) · [ZebraYOLO 群游检测 (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12504650/) · [ZebraTrack YOLOv8n 单鱼 (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.10.17.683088v1) |

---

## 各文献参考了哪些方面（详细说明）

### [1] TransReID (He et al., ICCV 2021) —— 模型架构与推理方式的核心依据

本框架的 **特征提取器（`src/reid/feature_extractor.py`）与配置（`configs/vit_transreid_zf_v2.yml`）** 几乎逐条复刻该文：

- **纯 Transformer 骨干**：用 ViT-Base 替代 CNN 作 Re-ID 骨干，本框架直接载入其预训练权重。
- **JPM（Jigsaw Patch Module，拼图补丁模块）**：通过 shift + patch shuffle 重排 patch 嵌入，输出"全局特征 + 4 段局部特征"。
  → 直接决定本框架特征维度为 **3840 = 768（全局 CLS）+ 4×768（局部）**，以及测试时取 concat 后做 L2 归一化。
- **SIE（Side Information Embeddings，侧信息嵌入）**：把相机/视角 ID 作为可学习嵌入注入模型，缓解相机/视角带来的特征偏差。
  → 决定配置中 `SIE_CAMERA=True, SIE_VIEW=False, camera_num=2`，以及 `c1→0 / c2→1` 的相机 ID 映射。
- **强基线改进 + ImageNet 预训练 + bnneck 颈层**：测试时 `NECK_FEAT=after, FEAT_NORM=yes`。
  → 决定特征后处理流程（`feature_extractor.py` 中 `F.normalize(p=2)`）。
- **输入尺寸反推**：该文使用 patch stride=[16,16] 的 ViT；我们从权重 `base.pos_embed` 形状 `(1,289,768)` 反推训练/推理尺寸为 **[256,288]**（289 = 16×18 patch + 1 CLS），与该文 ViT 配置一致，从而实现 **100% 权重载入**。

### [2] Cao et al., ESC-IDNet (2025) —— 应用领域合理性 + 多视角分别建模思路

- **课题合理性**：该文证明斑马鱼（有花纹物种）可依靠视觉特征做个体识别，验证了我们"斑马鱼 Re-ID"方向的成立。
- **多区域/多视角思路（关键）**：该文用 **lateral body（侧面身体）+ dorsal head（背侧头部）** 两个区域分别检测、对齐、识别，
  说明"斑马鱼不同视角/部位外观差异大、应分别建模"是学界共识。
  → **直接支持"为每条鱼的两个侧面（c1/c2）分别建库/存代表向量"的设计**。本框架因此为每条鱼按 `cam` 额外存储 `鱼ID#c0` / `鱼ID#c1` 两个侧面聚合向量（见 `build_gallery` 与 `query_feature` 的 identity 模式）。
- **纵向识别挑战**：随时间（成熟）外观变化导致识别退化，提示需要更新策略——本框架当前未做时序更新，列为后续工作。

### [3] Rolling-window CNN/ViT (Scientific Reports 15, 8580, 2025) —— 前处理"增强条纹、保持颜色"的实证支撑

- **特征重要性实证**：该文通过图像篡改实验证明，斑马鱼识别主要依赖 **条纹图案（stripe pattern）** 与 **颜色（color）** 两类特征；彩色图像时颜色主导特征空间，灰度时条纹主导。
  → **直接支撑本框架前处理哲学**：用 CLAHE 增强判别性强的"条纹对比度"，同时在 LAB 的 L 通道操作以"保持颜色不变"，不破坏同样重要的颜色信息。
- **ViT + ImageNet 预训练在斑马鱼上有效**：支撑我们直接复用预训练 TransReID（ViT-Base，ImageNet 初始化）而不从头训练。
- ⚠️ **重要澄清**：该文本身使用"滚动窗口训练"做分类，**并未使用 CLAHE**。CLAHE 是本框架借鉴通用图像增强方法、结合该文"条纹/颜色都重要"的结论后自行采用的前处理，不能算作该文的贡献。

### [4] CLAHE 技术及其鱼类/水下应用 —— 前处理 `enhance.py` 的方法来源

- **CLAHE 本身**（限制对比度自适应直方图均衡化，Zuiderveld 1994 经典方法，见 Wikipedia）：用于提升局部对比度、抑制噪声放大。
  → 本框架 `enhance.py` 中 `enhance_clahe()` 的理论来源。
- **LAB 空间 L 通道 CLAHE**：标准做法是在亮度通道增强、保留 a/b 色度通道，实现"增强条纹、保持颜色"。这是医学/水下影像的通用惯例，并非单一论文独创。
- **在鱼类/水下图像中的有效性**：
  - SiamFCA（水产鱼类跟踪）用 CLAHE 抑制水下噪声、提升对比度；
  - juvenile ayu（香鱼）检测论文证实 CLAHE 在浑浊/低照度下显著提升目标可见性、减少漏检。
  → 支撑我们将 CLAHE 用于鱼体图像增强的合理性。

### [5] Market-1501 benchmark (Zheng et al., ICCV 2015) —— 评估协议来源

- **Rank-1 / Rank-5（CMC）与 mAP 的定义**，以及 **"查询集与库集来自不同相机（cross-camera）"** 的评测设定，是 Re-ID 领域标准协议。TransReID 原论文即在 Market-1501 上按此协议报告结果。
  → 本框架 `evaluate_cross_view()` 的交叉视角评估**直接沿用该标准协议**：用一侧（c1）作 Gallery、另一侧（c2）作 Query，反向再做一次，报告 Rank-1/Rank-5/mAP。
  - 作为对照，本框架另提供 `evaluate_same_view()`（同视角 c1→c1 / c2→c2），用于显示"侧面花纹差异"带来的难度差距。

### [6] 视频帧选取策略（分段均匀采样 + 质量优选 + 去冗余）—— 新增模块 `video_extractor.py` 的方法依据

- **时间覆盖 / 分段均匀采样**：视频行人 Re-ID 研究（arXiv:1702.06294）证明"把视频切成 K 段、从每段抽代表帧"优于全局随机采样，且能避免时序坍缩；实验显示 4 帧左右最佳，过多冗余帧不提升甚至拖累。
  → 本框架 `video_extractor.py` 用 `sample_stride`（每隔 N 帧抽候选）实现"覆盖整段视频"的均匀覆盖。
- **质量优选**：IUST PersonReId（arXiv:2412.18874）在均匀采样后，用 BRISQUE 质量评估在候选帧中优选最清晰子集，并限制每人最多 50 帧以平衡多样性与冗余。
  → 本框架复用 `quality.compute_blur_score`（Laplacian 方差）做清晰度过滤，并用 `max_frames_per_fish` 限制每条鱼的总帧数（c1/c2 共用，因单视频会产出两种侧面）。
- **去冗余**：上述两文共同指向"视频帧高度冗余、需精选"。本框架第三阶段用"相邻已选帧的灰度 MAE > 阈值"剔除近重复帧（详见 [7]）。

### [7] 野生动/畜关键帧提取（检测 + 模糊检测 + 聚类 / 去重）—— 去冗余与质量保障依据

- **Kākā 鹦鹉 Re-ID（arXiv:2510.08775）**：提出无监督流程，组合 **YOLO/GroundingDINO 目标检测 + 光流模糊检测 + DINOv2 编码 + 聚类**，挑出代表性关键帧；明确指出"应选清晰、跨多样场景的帧，且 ML 能捕捉人眼不可辨的细微差异"。
  → 直接支撑本框架"先 YOLO 检测裁剪 → 再清晰度+去冗余精选"的两段式关键帧思路；其"光流模糊检测"对应我们的 Laplacian 模糊过滤。
- **畜脸识别（CN114581948A）**：用**光流运动量最小**取关键帧，再用 **SSIM 结构相似性去重 + 拉普拉斯方差择优保留**。
  → 支撑本框架"SSIM/MAE 近重复帧剔除 + Laplacian 清晰度择优"的工程实现（见 `video_extractor.frame_mae` 与清晰度过滤）。

### [8] 斑马鱼 YOLO 检测（单类：zebrafish）—— `video_extractor.py` 的 YOLO 检测裁剪依据

- **ESC-YOLOv8-seg（2025）**：基于 YOLOv8 的斑马鱼体表异常检测/分割框架，在水产复杂背景下对小目标斑马鱼达 ~98% 精度、106 FPS，证明 **YOLOv8 对斑马鱼检测高度有效**。
- **ZebraYOLO（2025）**：针对斑马鱼群游行为，在 YOLOv8s 上增配 P2 检测头 + 全局注意力，提升单/多目标定位精度；说明 YOLOv8 体系适用于斑马鱼定位。
- **ZebraTrack（2025）**：用 **YOLOv8n 检测单条斑马鱼**，从视频帧中提取目标并裁剪——正是本框架"单条鱼视频 → YOLO 检测 → 裁剪鱼体"的同类做法。
- **单类检测 + 后处理几何法判左右（本框架当前设计）**：标准检测型 YOLO **无法自行判定左/右朝向**（对水平翻转近似不变）。若强行让 YOLO 分左右两类，左右是"手性（镜像对称）"问题，分类梯度会被 box/object 损失主导而学得很弱。
  因此本框架改为：**YOLO 只做单类检测（zebrafish）**，负责"是不是鱼、框在哪"；**左右（c1/c2）交给后处理 `tools/side_geometry_label.py`（几何朝向法）判定**——
  固定机位 + 鱼背朝上时，"鱼头朝画面左/右"与"看到哪一侧面"一一对应，故用鱼体 PCA 主轴 + 眼睛暗斑定位头端即可判左右（文献法 [6][7] 支撑，零额外训练）。
  **这一设计的取舍**：侧面筛选不再由"高置信=左右类"保证，而是依赖"用户喂的以侧面为主的视频 + 清晰度阈值 + 几何法对非侧面的角度拒绝（鱼太竖直则判 unknown）"；换来的是 YOLO 训练简单、标签无左右歧义。
  ⚠️ **标注时只需标一个类 `zebrafish`**：框出鱼体即可，无需区分左右（左右由几何法判定）。建议覆盖尽量多的鱼与姿态。
- **对 YOLO 模型选择的支撑（回答"是否需要重训"）**：现有模型是**多目标 + 含非侧面鱼**训练的，对"单条鱼"虽可用（取最高置信度框），但框可能偏松、置信偏低；
  **重训一个单类（zebrafish）YOLOv8s** 会得到更紧更准的裁剪，提升下游 Re-ID 特征质量。
  本框架的 `video_extractor.py` **只做推理加载（任意 `.pt`）**，训练/标注按你要求放在框架之外（用 X-AnyLabeling 标 `zebrafish` 矩形框 → 转单类 YOLO txt → `yolov8s` 单类训练 → 导出 `best.pt` 放入 `models/` 即可）。

---

## 引用时的注意事项（给论文写作）

1. **[3] 与 CLAHE 的关系**要如实写：CLAHE 是通用增强方法（归 [4]），[3] 只提供了"条纹+颜色都重要"的实证依据，不要写成"[3] 使用了 CLAHE"。
2. **本框架未微调/重训** TransReID，因此 [1] 是"方法复刻"而非"在我们的数据上训练"；若论文强调"直接使用预训练权重做推理"，应明确此点。
3. **评估协议**建议同时报告 `evaluate_cross_view`（难，c1↔c2）与 `evaluate_same_view`（易，同视角），以全面反映模型在"侧面差异"下的真实能力——这正是 [2] 多视角思路的体现。
4. **视频抽取模块（新增）** 的帧选取依据 [6][7]、YOLO 单类检测裁剪依据 [8]；若论文描述"视频→鱼图"的预处理流程，应引用这几篇说明"分段均匀采样 + 清晰度过滤 + 近重复帧去冗余 + 单类 YOLO 检测裁剪（左右由后处理几何法判定）"的设计合理性。
