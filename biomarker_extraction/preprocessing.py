"""
Biomarker-specific image preprocessing.

Separate from the MobileNet disease-classification preprocessing in app.py.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def load_image_bgr(image_path: str) -> np.ndarray:
    """Load an image from disk as a BGR uint8 array."""
    with Image.open(image_path) as img:
        rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_image_bgr_from_array(image: np.ndarray) -> np.ndarray:
    """Ensure an in-memory image is a BGR uint8 array."""
    if image.dtype != np.uint8:
        raise ValueError("Image array must have dtype uint8")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Image array must be HxWx3 or HxWx4")

    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Accept RGB input; callers passing BGR should use load_image_bgr instead.
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def resize_for_analysis(
    image_bgr: np.ndarray,
    max_dimension: int = 768,
) -> Tuple[np.ndarray, float]:
    """
    Resize large images while preserving aspect ratio.

    Returns the resized image and the scale factor applied relative to the
    original (resized = original * scale).
    """
    height, width = image_bgr.shape[:2]
    longest = max(height, width)

    if longest <= max_dimension:
        return image_bgr.copy(), 1.0

    scale = max_dimension / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to single-channel grayscale."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def to_lab(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to LAB."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
