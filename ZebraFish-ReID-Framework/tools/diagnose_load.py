"""
诊断脚本：验证重建的 vit_transreid_zf_v2.yml 配置能否完整加载黄金权重 transformer_20.pth。

方法：用形状匹配过滤加载（与 Dynamic_Domain_Eval.py 一致），统计
  - 总参数量 / 成功载入参数量 / 覆盖率
  - 关键层（patch_embed 骨干 / SIE 相机嵌入 / bnneck  Neck / 分类头）是否载入

若覆盖率偏低，说明 cfg 架构与训练时不一致，需调整 STRIDE_SIZE 等字段。
"""
import os
import sys
import yaml
import torch

# ---- 路径 ----
FRAMEWORK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TP = os.path.join(FRAMEWORK, "third_party", "TransReID")
WEIGHT = os.path.join(FRAMEWORK, "models", "transformer_20.pth")
if not os.path.exists(WEIGHT):
    # 退回到用户给出的原始位置
    WEIGHT = r"D:/YUANRUXI0124/2026论文/（新）单条鱼数据集/transformer_20.pth"

sys.path.insert(0, TP)
from config import cfg
from model import make_model


def load_coverage(stride, device):
    cfg.defrost()
    # Windows 下 yacs.merge_from_file 默认 GBK 打开 yml，含中文注释会报错；
    # 改为显式 UTF-8 读取后合并。
    _yml = os.path.join(FRAMEWORK, "configs", "vit_transreid_zf_v2.yml")
    with open(_yml, "r", encoding="utf-8") as _f:
        _cfg_dict = yaml.safe_load(_f)
    cfg.merge_from_other_cfg(type(cfg)(_cfg_dict))
    cfg.MODEL.STRIDE_SIZE = stride
    cfg.freeze()

    model = make_model(cfg, num_class=20, camera_num=2, view_num=1)
    state_dict = torch.load(WEIGHT, map_location=device)
    model_dict = model.state_dict()
    new_sd = {k: v for k, v in state_dict.items()
              if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(new_sd)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    total = sum(v.numel() for v in model.state_dict().values())
    loaded = sum(v.numel() for k, v in model.state_dict().items()
                 if k in new_sd and new_sd[k].shape == v.shape)
    ratio = loaded / total * 100

    # 关键层检查
    keys = set(new_sd.keys())
    checks = {
        "backbone.patch_embed": any("patch_embed" in k for k in keys),
        "SIE camera embed": any("camera" in k for k in keys),
        "neck(bnneck)": any("bnneck" in k or "neck" in k.lower() for k in keys),
        "global classifier": any(k == "classifier.weight" for k in keys),
        "JPM local classifier_1": any(k == "classifier_1.weight" for k in keys),
    }
    return ratio, total, loaded, checks


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, weight={WEIGHT}")
    print("=" * 60)
    for stride in ([16, 16], [12, 12]):
        ratio, total, loaded, checks = load_coverage(stride, device)
        print(f"\nSTRIDE_SIZE = {stride}")
        print(f"  覆盖率: {ratio:.2f}%  ({loaded:,}/{total:,} params)")
        for name, ok in checks.items():
            print(f"    [{'OK' if ok else 'XX'}] {name}")


if __name__ == "__main__":
    main()
