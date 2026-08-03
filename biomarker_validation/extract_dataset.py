"""
Extract biomarkers from dataset images using the existing extractor.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Dict, List, Optional

from biomarker_extraction.extractor import extract_biomarkers
from biomarker_validation.constants import (
    BIOMARKER_COLUMNS,
    DEFAULT_RANDOM_SEED,
    DISEASE_LABELS,
    parse_dataset_path,
)
from biomarker_validation.dataset_discovery import collect_dataset_images, is_image_file


def sample_images_by_disease(
    image_paths: List[str],
    dataset_root: str,
    max_per_disease: Optional[int],
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> List[str]:
    """
    Reproducibly sample up to max_per_disease images per mapped disease label.

    Foot and Mouth combines images from dataset/foot/ and dataset/mouth/.
    """
    if max_per_disease is None:
        return image_paths

    grouped: Dict[str, List[str]] = {label: [] for label in DISEASE_LABELS}
    for path in image_paths:
        _, disease, _ = parse_dataset_path(path, dataset_root)
        grouped[disease].append(path)

    rng = random.Random(random_seed)
    sampled: List[str] = []
    for disease in DISEASE_LABELS:
        paths = grouped.get(disease, [])
        if len(paths) <= max_per_disease:
            sampled.extend(paths)
        else:
            chosen = rng.sample(paths, max_per_disease)
            sampled.extend(chosen)
    sampled.sort()
    return sampled


def extract_dataset_biomarkers(
    dataset_root: str,
    output_csv: str,
    max_per_disease: Optional[int] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_dimension: int = 768,
) -> Dict[str, object]:
    """
    Process dataset images and write biomarker CSV.

    Returns metadata about the extraction run.
    """
    all_images = collect_dataset_images(dataset_root)
    selected_images = sample_images_by_disease(
        all_images,
        dataset_root,
        max_per_disease=max_per_disease,
        random_seed=random_seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    fieldnames = [
        "image_name",
        "image_path",
        "dataset_folder",
        "disease",
        "stage",
        *BIOMARKER_COLUMNS,
        "contour_count",
        "extraction_status",
        "error_message",
    ]

    successful = 0
    failed = 0
    rows: List[Dict[str, object]] = []

    for image_path in selected_images:
        dataset_folder, disease, stage = parse_dataset_path(image_path, dataset_root)
        row: Dict[str, object] = {
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
            "dataset_folder": dataset_folder,
            "disease": disease,
            "stage": stage,
            "contour_count": None,
            "extraction_status": "success",
            "error_message": "",
        }

        try:
            result = extract_biomarkers(image_path, max_dimension=max_dimension)
            metadata = result.get("metadata", {})
            row["contour_count"] = metadata.get("contour_count")
            for col in BIOMARKER_COLUMNS:
                row[col] = result.get(col)
            successful += 1
        except Exception as exc:
            row["extraction_status"] = "failed"
            row["error_message"] = str(exc)
            for col in BIOMARKER_COLUMNS:
                row[col] = None
            failed += 1

        rows.append(row)

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sampling_mode = "full dataset" if max_per_disease is None else "sampled dataset"
    return {
        "dataset_root": os.path.abspath(dataset_root),
        "total_images_in_dataset": len(all_images),
        "images_selected": len(selected_images),
        "successful_extractions": successful,
        "failed_extractions": failed,
        "sampling_mode": sampling_mode,
        "max_per_disease": max_per_disease,
        "random_seed": random_seed if max_per_disease is not None else None,
        "output_csv": output_csv,
    }
