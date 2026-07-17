"""
feature_extractor.py
------------------------------------------------------------------------------
斑马鱼 Re-ID 特征提取器（推理专用）。

严格复刻 Dynamic_Domain_Eval.py 的官方加载与推理流程：
  1. 加载 vit_transreid_zf_v2.yml 配置（JPM=True, SIE_CAMERA=True）
  2. make_model(cfg, num_class=20, camera_num=2, view_num=1)
  3. 形状匹配过滤加载 transformer_20.pth（忽略分类层），覆盖率 ~99.9%
  4. 推理变换: Resize([256,288]) -> ToTensor -> Normalize(ImageNet)
  5. 前向: model(img, cam_label, view_label) -> L2 归一化 -> 特征向量

特征维度 = 768(global) + 4×768(JPM local) = 3840 维（与官方评估一致）。
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from PIL import Image
from torchvision import transforms

# 将 vendored TransReID 加入路径
_THIS = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK = os.path.dirname(os.path.dirname(_THIS))
_TP = os.path.join(_FRAMEWORK, "third_party", "TransReID")
if _TP not in sys.path:
    sys.path.insert(0, _TP)

from config import cfg
from model import make_model


def _default_weight_path():
    p = os.path.join(_FRAMEWORK, "models", "transformer_20.pth")
    if os.path.exists(p):
        return p
    return r"D:/YUANRUXI0124/2026论文/（新）单条鱼数据集/transformer_20.pth"


class ZebraFishFeatureExtractor:
    def __init__(self, weight_path=None, config_path=None, device=None):
        self.weight_path = weight_path or _default_weight_path()
        self.config_path = config_path or os.path.join(_FRAMEWORK, "configs", "vit_transreid_zf_v2.yml")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1) 配置（JPM=True 架构）
        # 注意：Windows 下 yacs.merge_from_file 默认用 GBK 打开 yml，
        # 若含中文注释会触发 UnicodeDecodeError；这里显式 UTF-8 读取后合并，跨平台安全。
        cfg.defrost()
        with open(self.config_path, "r", encoding="utf-8") as _f:
            _cfg_dict = yaml.safe_load(_f)
        cfg.merge_from_other_cfg(type(cfg)(_cfg_dict))
        cfg.freeze()

        # 2) 构建模型（camera_num=2 与权重 SIE 对齐）
        self.model = make_model(cfg, num_class=20, camera_num=2, view_num=1)
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

        # 3) 严格复刻的推理变换
        self.transform = transforms.Compose([
            transforms.Resize(cfg.INPUT.SIZE_TEST),          # [256, 288]
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN,
                                 std=cfg.INPUT.PIXEL_STD),   # ImageNet
        ])
        self.feat_dim = None  # 首次前向时确定

    def _load_weights(self):
        state_dict = torch.load(self.weight_path, map_location=self.device)
        model_dict = self.model.state_dict()
        new_sd = {k: v for k, v in state_dict.items()
                  if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(new_sd)
        self.model.load_state_dict(model_dict)
        self._loaded = len(new_sd)
        self._total = len(model_dict)
        # 关键自检：位置编码(pos_embed)必须完整载入，否则输入尺寸与训练不一致，
        # 特征会整体失效。
        pe_keys = [k for k in model_dict if "pos_embed" in k]
        if pe_keys and pe_keys[0] not in new_sd:
            raise RuntimeError(
                "致命错误：pos_embed 未载入！说明 INPUT.SIZE_TRAIN 与训练权重不一致。"
                "请核对 configs/vit_transreid_zf_v2.yml 的 SIZE_TRAIN（当前应为 [256,288]）。")

    @staticmethod
    def cam_id_from_name(fname):
        """文件名 NNNN_cXsY_ZZZZ.png -> 相机id: c1->0, c2->1"""
        parts = fname.split("_")
        return int(parts[1][1]) - 1

    @torch.no_grad()
    def extract(self, img_pil, cam_id=0):
        """
        img_pil: PIL RGB 图像（增强后的鱼体图）
        cam_id: 0(c1左) / 1(c2右)
        返回: L2 归一化的 numpy 向量 (feat_dim,)
        """
        img_t = self.transform(img_pil).unsqueeze(0).to(self.device)
        val_cam = torch.tensor([cam_id], dtype=torch.long, device=self.device)
        val_view = torch.tensor([0], dtype=torch.long, device=self.device)
        feat = self.model(img_t, cam_label=val_cam, view_label=val_view)
        feat = F.normalize(feat, p=2, dim=1).cpu().numpy().squeeze()
        if self.feat_dim is None:
            self.feat_dim = feat.shape[0]
        return feat

    def extract_from_path(self, img_path):
        fname = os.path.basename(img_path)
        cam_id = self.cam_id_from_name(fname)
        img = Image.open(img_path).convert("RGB")
        return self.extract(img, cam_id), cam_id

    def load_report(self):
        return {"loaded": self._loaded, "total": self._total,
                "coverage": self._loaded / self._total * 100,
                "feat_dim": self.feat_dim, "device": self.device}


if __name__ == "__main__":
    ext = ZebraFishFeatureExtractor()
    rep = ext.load_report()
    print(f"[模型载入] 覆盖率 {rep['coverage']:.2f}% ({rep['loaded']}/{rep['total']})")

    # 用一张增强图验证端到端
    enh_dir = os.path.join(_FRAMEWORK, "data", "processed", "enhanced")
    if os.path.isdir(enh_dir) and os.listdir(enh_dir):
        sample = sorted(os.listdir(enh_dir))[0]
        feat, cam = ext.extract_from_path(os.path.join(enh_dir, sample))
        print(f"[端到端] 样本 {sample} cam={cam} -> 特征维度 {feat.shape}, L2范数 {np.linalg.norm(feat):.4f}")
    else:
        print("[提示] enhanced 目录为空，请先运行预处理生成增强图。")
