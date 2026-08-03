"""
Main interface for experimental visual biomarker extraction.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import numpy as np

from biomarker_extraction.boundary import compute_boundary_irregularity
from biomarker_extraction.clustering import compute_lesion_clustering
from biomarker_extraction.color_variation import compute_color_variation
from biomarker_extraction.debug_visualization import default_debug_path, save_debug_visualization
from biomarker_extraction.lesion_density import compute_lesion_density
from biomarker_extraction.lesion_detection import LesionDetectionResult, detect_lesion_candidates
from biomarker_extraction.preprocessing import (
    load_image_bgr,
    load_image_bgr_from_array,
    resize_for_analysis,
)
from biomarker_extraction.texture import compute_texture_roughness

ImageInput = Union[str, np.ndarray]


def _load_image(image: ImageInput) -> np.ndarray:
    if isinstance(image, str):
        if not os.path.isfile(image):
            raise FileNotFoundError(f"Image not found: {image}")
        return load_image_bgr(image)
    return load_image_bgr_from_array(image)


def _round_value(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def extract_biomarkers(
    image: ImageInput,
    debug_output_dir: Optional[str] = None,
    save_debug: bool = False,
    max_dimension: int = 768,
) -> Dict[str, Any]:
    """
    Extract visual biomarkers from a cow/buffalo skin image.

    Parameters
    ----------
    image:
        File path or RGB/BGR uint8 numpy array (HxWx3).
    debug_output_dir:
        Directory where debug visualization will be saved when save_debug=True.
    save_debug:
        Whether to write a debug visualization panel to disk.
    max_dimension:
        Maximum image dimension used during biomarker analysis preprocessing.

    Returns
    -------
    dict with biomarker values and optional metadata/debug path.
    """
    image_bgr = _load_image(image)
    image_bgr, _ = resize_for_analysis(image_bgr, max_dimension=max_dimension)

    detection: LesionDetectionResult = detect_lesion_candidates(image_bgr)

    lesion_density = compute_lesion_density(detection)
    texture_roughness = compute_texture_roughness(image_bgr)
    color_variation = compute_color_variation(image_bgr)
    lesion_clustering = compute_lesion_clustering(detection, image_bgr.shape)
    boundary_irregularity = compute_boundary_irregularity(detection)

    result: Dict[str, Any] = {
        "lesion_density": _round_value(lesion_density),
        "texture_roughness": _round_value(texture_roughness),
        "color_variation": _round_value(color_variation),
        "lesion_clustering": _round_value(lesion_clustering),
        "boundary_irregularity": _round_value(boundary_irregularity),
    }

    result["metadata"] = {
        "analyzed_pixel_count": detection.analyzed_pixel_count,
        "lesion_pixel_count": detection.lesion_pixel_count,
        "contour_count": len(detection.contours),
        "max_dimension": max_dimension,
    }

    if save_debug:
        debug_dir = debug_output_dir or "biomarker_debug"
        if isinstance(image, str):
            debug_path = default_debug_path(image, output_dir=debug_dir)
        else:
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "array_input_debug.jpg")

        result["debug_image_path"] = save_debug_visualization(
            image_bgr,
            detection,
            debug_path,
        )

    return result
