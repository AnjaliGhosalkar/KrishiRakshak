"""
Lesion clustering biomarker.

Estimates spatial grouping of candidate lesion regions using connected-
component centroids.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from biomarker_extraction.lesion_detection import LesionDetectionResult, get_component_centroids


def _mean_pairwise_distance(points: List[Tuple[float, float]]) -> float:
    """Compute mean Euclidean distance over all unique point pairs."""
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            distances.append(float(np.hypot(dx, dy)))
    return float(np.mean(distances))


def compute_lesion_clustering(
    detection: LesionDetectionResult,
    image_shape: Tuple[int, int],
) -> Optional[float]:
    """
    Return clustering score in [0, 1], or None if insufficient regions.

    Method:
        1. Extract centroids of connected components from lesion mask
        2. If fewer than 2 centroids, return None (unavailable)
        3. Compute mean pairwise centroid distance
        4. Normalize by half the image diagonal
        5. clustering = 1 - min(mean_distance / (0.5 * diagonal), 1.0)

    Higher values indicate centroids are closer together (more clustered).
    """
    centroids = get_component_centroids(detection.mask)
    if len(centroids) < 2:
        return None

    height, width = image_shape[:2]
    diagonal = float(np.hypot(width, height))
    if diagonal <= 0:
        return None

    mean_distance = _mean_pairwise_distance(centroids)
    normalized_distance = mean_distance / (0.5 * diagonal)
    score = 1.0 - min(normalized_distance, 1.0)
    return float(max(0.0, min(1.0, score)))
