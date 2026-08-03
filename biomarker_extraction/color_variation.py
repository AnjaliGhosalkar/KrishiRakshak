"""
Color variation biomarker.

Measures dispersion of saturation and value channels in HSV space.
"""

from __future__ import annotations

import numpy as np

from biomarker_extraction.preprocessing import to_hsv

# Tunable normalization constants documented for future calibration.
SATURATION_STD_SCALE = 70.0
VALUE_STD_SCALE = 80.0


def compute_color_variation(image_bgr: np.ndarray) -> float:
    """
    Return color variation score in [0, 1].

    Method:
        1. Convert image to HSV
        2. Compute standard deviation of S and V channels
        3. Normalize each channel std by documented scale constants
        4. Final score = average of normalized S and V components

    Hue is excluded because it is circular and unstable under lighting shifts.
    """
    hsv = to_hsv(image_bgr)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    sat_std = float(np.std(saturation))
    val_std = float(np.std(value))

    sat_component = min(sat_std / SATURATION_STD_SCALE, 1.0)
    val_component = min(val_std / VALUE_STD_SCALE, 1.0)

    score = 0.5 * sat_component + 0.5 * val_component
    return float(max(0.0, min(1.0, score)))
