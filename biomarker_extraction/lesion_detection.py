"""
Modular candidate lesion-region detection.

The strategy is intentionally simple and swappable so segmentation can be
improved later without changing biomarker modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from biomarker_extraction.preprocessing import to_grayscale, to_hsv, to_lab


@dataclass
class LesionDetectionResult:
    """Output of candidate lesion detection."""

    mask: np.ndarray
    contours: List[np.ndarray] = field(default_factory=list)
    analyzed_pixel_count: int = 0
    lesion_pixel_count: int = 0


# Tunable parameters documented for future refinement.
MIN_COMPONENT_AREA = 40
MORPH_KERNEL_SIZE = 5
DEVIATION_BLUR_KSIZE = 31
DEVIATION_THRESHOLD = 18.0
SATURATION_THRESHOLD = 45
VALUE_LOW_THRESHOLD = 35
VALUE_HIGH_THRESHOLD = 245


def _remove_border_connected(mask: np.ndarray) -> np.ndarray:
    """Remove foreground components touching image borders (common artifact source)."""
    cleaned = mask.copy()
    height, width = cleaned.shape
    flood = cleaned.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 0)
    return flood


def _build_deviation_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detect pixels that deviate from a locally smoothed color baseline.

    Uses LAB distance from a large-kernel median blur as a lighting-robust
    deviation signal, then Otsu-thresholds the deviation map.
    """
    lab = to_lab(image_bgr).astype(np.float32)
    baseline = cv2.medianBlur(lab.astype(np.uint8), DEVIATION_BLUR_KSIZE).astype(np.float32)
    deviation = np.linalg.norm(lab - baseline, axis=2)

    deviation_u8 = cv2.normalize(deviation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(deviation_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Require deviation above a minimum absolute level to reduce noise.
    strong = (deviation >= DEVIATION_THRESHOLD).astype(np.uint8) * 255
    return cv2.bitwise_and(mask, strong)


def _build_color_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detect unusually saturated or extreme-value regions in HSV space.

    This catches reddish/inflamed or very dark/bright lesion-like areas.
    """
    hsv = to_hsv(image_bgr)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    sat_mask = (saturation >= SATURATION_THRESHOLD).astype(np.uint8) * 255
    value_mask = (
        (value <= VALUE_LOW_THRESHOLD) | (value >= VALUE_HIGH_THRESHOLD)
    ).astype(np.uint8) * 255
    return cv2.bitwise_or(sat_mask, value_mask)


def _build_texture_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Highlight high local-contrast regions using Laplacian magnitude.

    Helps capture raised/textured lesion areas missed by color-only rules.
    """
    gray = to_grayscale(image_bgr)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    magnitude = np.abs(laplacian)
    magnitude_u8 = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(magnitude_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def detect_lesion_candidates(image_bgr: np.ndarray) -> LesionDetectionResult:
    """
    Detect candidate lesion-like regions using a modular multi-signal mask.

    Signals are combined with logical OR, then cleaned morphologically.
    Connected components smaller than MIN_COMPONENT_AREA are discarded.
    """
    deviation_mask = _build_deviation_mask(image_bgr)
    color_mask = _build_color_mask(image_bgr)
    texture_mask = _build_texture_mask(image_bgr)

    combined = cv2.bitwise_or(deviation_mask, color_mask)
    combined = cv2.bitwise_or(combined, texture_mask)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = _remove_border_connected(combined)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    filtered = np.zeros_like(combined)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= MIN_COMPONENT_AREA:
            filtered[labels == label_idx] = 255

    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    analyzed_pixel_count = int(filtered.size)
    lesion_pixel_count = int(np.count_nonzero(filtered))

    return LesionDetectionResult(
        mask=filtered,
        contours=contours,
        analyzed_pixel_count=analyzed_pixel_count,
        lesion_pixel_count=lesion_pixel_count,
    )


def get_component_centroids(
    mask: np.ndarray,
    min_area: int = MIN_COMPONENT_AREA,
) -> List[Tuple[float, float]]:
    """Return (x, y) centroids for connected components above min_area."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    points: List[Tuple[float, float]] = []
    for label_idx in range(1, num_labels):
        if stats[label_idx, cv2.CC_STAT_AREA] >= min_area:
            points.append((float(centroids[label_idx, 0]), float(centroids[label_idx, 1])))
    return points
