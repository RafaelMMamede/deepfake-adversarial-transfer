"""Loss registry for the evaluated detectors."""

import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

from metrics.registry import LOSSFUNC
from .cross_entropy_loss import CrossEntropyLoss

__all__ = ["LOSSFUNC", "CrossEntropyLoss"]