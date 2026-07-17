"""
enhance.py — 对比度增强（方案2核心：让条纹/斑纹等身份特征更突出）

方法：在 LAB 色彩空间的 L（亮度）通道做 CLAHE（限制对比度自适应直方图均衡），
再合并回 RGB。

文献依据：
  - Nature 2025 斑马鱼论文：保持颜色信息（灰度->彩色会误分类），仅增强对比度
  - 用户需求：“清晰处理、使得特征更加明显”

注意：这是“轻度”增强（clip_limit 默认 2.0），目的是统一不同帧间的光照/水质
差异、突出条纹，而非改变语义内容。可通过参数关闭或调弱。
"""
import numpy as np
import cv2


def enhance_clahe(bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """
    bgr: BGR numpy 数组
    返回: 增强后的 BGR numpy 数组
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
