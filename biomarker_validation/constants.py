"""
Dataset label mappings for biomarker validation.

The repository stores Foot and Mouth disease images in two separate folders:
    dataset/foot/
    dataset/mouth/

For grouped disease analysis, both map to the intended label "Foot and Mouth".
Original folder names are always preserved in dataset_folder.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Raw dataset folder names under dataset/
DATASET_FOLDERS = ["healthy", "lumpy", "mastitis", "foot", "mouth"]

STAGE_NAMES = ["early", "moderate", "severe"]

# Map raw folder -> intended disease label for grouped analysis
FOLDER_TO_DISEASE: Dict[str, str] = {
    "healthy": "Healthy",
    "lumpy": "Lumpy",
    "mastitis": "Mastitis",
    "foot": "Foot and Mouth",
    "mouth": "Foot and Mouth",
}

DISEASE_LABELS = ["Healthy", "Lumpy", "Mastitis", "Foot and Mouth"]

BIOMARKER_COLUMNS = [
    "lesion_density",
    "texture_roughness",
    "color_variation",
    "lesion_clustering",
    "boundary_irregularity",
]

DEFAULT_RANDOM_SEED = 42


def parse_dataset_path(image_path: str, dataset_root: str) -> Tuple[str, str, str]:
    """
    Parse dataset_folder, disease label, and stage from an image path.

    Returns
    -------
    dataset_folder, disease, stage

    stage is "none" for healthy (flat folder) images.
    """
    rel = os.path.relpath(image_path, dataset_root)
    parts = rel.replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Unexpected dataset path structure: {image_path}")

    dataset_folder = parts[0].lower()
    disease = FOLDER_TO_DISEASE.get(dataset_folder)
    if disease is None:
        raise ValueError(f"Unknown dataset folder: {dataset_folder}")

    if dataset_folder == "healthy":
        stage = "none"
    elif len(parts) >= 3 and parts[1].lower() in STAGE_NAMES:
        stage = parts[1].lower()
    else:
        stage = "unknown"

    return dataset_folder, disease, stage
