# pbr_modules/__init__.py

from .predictor import PBRMaterialPredictor
from .brdf_renderer import render_pbr_image
from .style_loss import VGGStyleLoss

# 声明向外暴露的核心组件
__all__ = [
    "PBRMaterialPredictor",
    "render_pbr_image",
    "VGGStyleLoss"
]