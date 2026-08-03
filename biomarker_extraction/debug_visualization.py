"""
Debug visualization for biomarker extraction experiments.

Saves side-by-side images showing original content, lesion mask, and
contour overlays without modifying source images.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from biomarker_extraction.lesion_detection import LesionDetectionResult


def _ensure_bgr(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    return image_bgr.copy()


def create_debug_panel(
    image_bgr: np.ndarray,
    detection: LesionDetectionResult,
) -> np.ndarray:
    """
    Build a horizontal panel: Original | Mask overlay | Contours.
    """
    original = _ensure_bgr(image_bgr)

    mask_overlay = original.copy()
    mask_indices = detection.mask > 0
    mask_overlay[mask_indices] = (
        0.4 * mask_overlay[mask_indices] + 0.6 * np.array([0, 0, 255], dtype=np.float32)
    ).astype(np.uint8)

    contour_view = original.copy()
    if detection.contours:
        cv2.drawContours(contour_view, detection.contours, -1, (0, 255, 0), 2)

    panel = np.hstack([original, mask_overlay, contour_view])
    return panel


def save_debug_visualization(
    image_bgr: np.ndarray,
    detection: LesionDetectionResult,
    output_path: str,
) -> str:
    """
    Save debug panel to output_path.

    Returns the written file path.
    """
    panel = create_debug_panel(image_bgr, detection)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, panel)
    return output_path


def default_debug_path(image_path: str, output_dir: str = "biomarker_debug") -> str:
    """Generate a default debug output path for an input image."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_dir, f"{base_name}_debug.jpg")
