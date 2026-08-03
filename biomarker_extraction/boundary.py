"""
Boundary irregularity biomarker.

Quantifies contour irregularity using circularity derived from contour area
and perimeter.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from biomarker_extraction.lesion_detection import LesionDetectionResult, MIN_COMPONENT_AREA


def _contour_circularity(contour: np.ndarray) -> Optional[float]:
    """
    Compute circularity = 4 * pi * A / P^2.

    Returns None when area or perimeter is invalid.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if area <= 0 or perimeter <= 0:
        return None
    return float((4.0 * np.pi * area) / (perimeter ** 2))


def compute_boundary_irregularity(detection: LesionDetectionResult) -> Optional[float]:
    """
    Return boundary irregularity in [0, 1], or None if no valid contours.

    Method:
        1. For each contour above MIN_COMPONENT_AREA:
           circularity = 4 * pi * A / P^2
           irregularity = 1 - min(circularity, 1.0)
        2. Return area-weighted mean irregularity across contours

    A perfect circle has circularity 1.0 and irregularity 0.0.
    More irregular boundaries yield higher irregularity scores.
    """
    weighted_sum = 0.0
    total_area = 0.0

    for contour in detection.contours:
        area = cv2.contourArea(contour)
        if area < MIN_COMPONENT_AREA:
            continue

        circularity = _contour_circularity(contour)
        if circularity is None:
            continue

        circularity = min(circularity, 1.0)
        irregularity = 1.0 - circularity
        weighted_sum += irregularity * area
        total_area += area

    if total_area <= 0:
        return None

    score = weighted_sum / total_area
    return float(max(0.0, min(1.0, score)))
