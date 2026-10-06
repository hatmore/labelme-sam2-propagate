"""
LabelMe SAM2 Propagate - Video annotation acceleration using SAM 2.1
"""

__version__ = "0.1.0"
__author__ = "hatmore"
__description__ = "SAM2 video propagation tool for LabelMe annotations"

from .core import run_propagation
from .utils import mask_to_polygon, polygon_to_mask, visualize_annotation

__all__ = [
    "run_propagation",
    "mask_to_polygon",
    "polygon_to_mask",
    "visualize_annotation",
]
