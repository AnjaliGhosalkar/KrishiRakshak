"""
Lesion density biomarker.

Measures the proportion of analyzed image area occupied by candidate
lesion-like regions.
"""

from __future__ import annotations

from biomarker_extraction.lesion_detection import LesionDetectionResult


def compute_lesion_density(detection: LesionDetectionResult) -> float:
    """
    Return lesion pixel ratio in [0, 1].

    Formula:
        lesion_density = lesion_pixel_count / analyzed_pixel_count
    """
    if detection.analyzed_pixel_count <= 0:
        return 0.0

    ratio = detection.lesion_pixel_count / float(detection.analyzed_pixel_count)
    return float(max(0.0, min(1.0, ratio)))
