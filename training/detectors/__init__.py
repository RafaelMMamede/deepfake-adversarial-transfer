"""Detector registry for the six evaluated backbones."""

import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

from metrics.registry import DETECTOR

from .xception_detector import XceptionDetector
from .efficientnetb4_detector import EfficientDetector
from .resnet34_detector import ResnetDetector
from .deit_detector_custom import DeiTDetector
from .swin_detector_custom import SwinDetector
from .vit_detector_custom import ViTDetector

__all__ = [
    "DETECTOR",
    "XceptionDetector",
    "EfficientDetector",
    "ResnetDetector",
    "DeiTDetector",
    "SwinDetector",
    "ViTDetector",
]