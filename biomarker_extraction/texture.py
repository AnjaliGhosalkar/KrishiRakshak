"""
Texture roughness biomarker.

Combines uniform Local Binary Pattern (LBP) dispersion with mean Sobel
gradient magnitude to quantify local texture irregularity.
"""

from __future__ import annotations

import cv2
import numpy as np

from biomarker_extraction.preprocessing import to_grayscale

# Tunable normalization constants documented for future calibration.
LBP_ENTROPY_SCALE = 2.5
GRADIENT_SCALE = 40.0


def _uniform_lbp(gray: np.ndarray) -> np.ndarray:
    """
    Compute 8-neighbor uniform LBP codes (P=8, R=1).

    Border pixels are excluded from downstream statistics.
    """
    gray = gray.astype(np.uint8)
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
    center = padded[1:-1, 1:-1]

    neighbors = [
        padded[0:-2, 0:-2],
        padded[0:-2, 1:-1],
        padded[0:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, 0:-2],
        padded[1:-1, 0:-2],
    ]

    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    return lbp


def _normalized_entropy(values: np.ndarray, bins: int = 256) -> float:
    """Shannon entropy normalized to [0, 1] by dividing by log2(bins)."""
    hist, _ = np.histogram(values.ravel(), bins=bins, range=(0, bins), density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    entropy = -np.sum(hist * np.log2(hist))
    return float(min(1.0, entropy / np.log2(bins)))


def compute_texture_roughness(image_bgr: np.ndarray) -> float:
    """
    Return texture roughness in [0, 1].

    Method:
        1. Uniform LBP entropy on grayscale image
        2. Mean Sobel gradient magnitude on grayscale image
        3. Final score = 0.6 * lbp_component + 0.4 * gradient_component

    Normalization:
        lbp_component = min(lbp_entropy / LBP_ENTROPY_SCALE, 1.0)
        gradient_component = 1 - exp(-mean_gradient / GRADIENT_SCALE)
    """
    gray = to_grayscale(image_bgr)
    lbp = _uniform_lbp(gray)
    lbp_entropy = _normalized_entropy(lbp, bins=256)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    mean_gradient = float(np.mean(gradient_magnitude))

    lbp_component = min(lbp_entropy / LBP_ENTROPY_SCALE, 1.0)
    gradient_component = float(1.0 - np.exp(-mean_gradient / GRADIENT_SCALE))

    score = 0.6 * lbp_component + 0.4 * gradient_component
    return float(max(0.0, min(1.0, score)))
