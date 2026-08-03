"""
Dataset discovery utilities for biomarker validation.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from biomarker_validation.constants import (
    DATASET_FOLDERS,
    FOLDER_TO_DISEASE,
    IMAGE_EXTENSIONS,
    STAGE_NAMES,
    parse_dataset_path,
)


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def collect_dataset_images(dataset_root: str) -> List[str]:
    """Collect all image paths under the dataset root."""
    images: List[str] = []
    for folder in DATASET_FOLDERS:
        folder_path = os.path.join(dataset_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for root, _, files in os.walk(folder_path):
            for filename in files:
                full_path = os.path.join(root, filename)
                if is_image_file(full_path):
                    images.append(full_path)
    images.sort()
    return images


def discover_dataset(dataset_root: str, dimension_sample_size: int = 200) -> Dict[str, Any]:
    """
    Inspect dataset structure and return a discovery report dict.
    """
    report: Dict[str, Any] = {
        "dataset_root": os.path.abspath(dataset_root),
        "dataset_used_by_project": os.path.abspath(dataset_root),
        "categories": {},
        "formats": Counter(),
        "invalid_images": [],
        "dimensions_sampled": [],
        "exact_duplicate_groups": [],
        "exact_duplicate_file_count": 0,
    }

    images = collect_dataset_images(dataset_root)
    report["total_images"] = len(images)

    per_folder: Dict[str, int] = defaultdict(int)
    per_stage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_disease: Dict[str, int] = defaultdict(int)

    hash_to_paths: Dict[str, List[str]] = defaultdict(list)
    dim_counter: Counter = Counter()
    sampled_dims = 0

    for image_path in images:
        ext = os.path.splitext(image_path)[1].lower()
        report["formats"][ext] += 1

        try:
            dataset_folder, disease, stage = parse_dataset_path(image_path, dataset_root)
        except ValueError as exc:
            report["invalid_images"].append({"path": image_path, "error": str(exc)})
            continue

        per_folder[dataset_folder] += 1
        per_disease[disease] += 1
        per_stage[dataset_folder][stage] += 1

        if sampled_dims < dimension_sample_size:
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                dim_counter[f"{width}x{height}"] += 1
                report["dimensions_sampled"].append(
                    {"path": image_path, "width": width, "height": height}
                )
                sampled_dims += 1
            except Exception as exc:
                report["invalid_images"].append({"path": image_path, "error": str(exc)})

        try:
            with open(image_path, "rb") as handle:
                digest = hashlib.md5(handle.read()).hexdigest()
            hash_to_paths[digest].append(image_path)
        except Exception as exc:
            report["invalid_images"].append({"path": image_path, "error": str(exc)})

    for folder in DATASET_FOLDERS:
        folder_path = os.path.join(dataset_root, folder)
        if not os.path.isdir(folder_path):
            report["categories"][folder] = {"exists": False, "image_count": 0}
            continue

        subdirs = [
            name
            for name in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, name))
        ]
        report["categories"][folder] = {
            "exists": True,
            "dataset_folder": folder,
            "mapped_disease": FOLDER_TO_DISEASE[folder],
            "image_count": per_folder.get(folder, 0),
            "stage_subdirectories": sorted(subdirs),
            "images_by_stage": dict(per_stage.get(folder, {})),
        }

    duplicate_groups = [paths for paths in hash_to_paths.values() if len(paths) > 1]
    report["exact_duplicate_groups"] = [
        {"hash": hashlib.md5(open(paths[0], "rb").read()).hexdigest(), "paths": paths}
        for paths in duplicate_groups[:50]
    ]
    report["exact_duplicate_group_count"] = len(duplicate_groups)
    report["exact_duplicate_file_count"] = sum(len(g) for g in duplicate_groups)
    report["images_by_disease"] = dict(per_disease)
    report["images_by_folder"] = dict(per_folder)
    report["common_dimensions_sample"] = dim_counter.most_common(10)
    report["dimension_sample_size"] = sampled_dims
    report["rotated_duplicate_detection"] = (
        "Not performed reliably in this experiment. Only exact byte-level duplicates "
        "were detected via MD5 hashing."
    )

    return report


def discovery_to_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten discovery report for CSV export."""
    rows: List[Dict[str, Any]] = []
    for folder, info in report.get("categories", {}).items():
        if not info.get("exists"):
            continue
        rows.append(
            {
                "dataset_folder": folder,
                "mapped_disease": info.get("mapped_disease"),
                "image_count": info.get("image_count"),
                "stage_subdirectories": ",".join(info.get("stage_subdirectories", [])),
            }
        )
        for stage, count in info.get("images_by_stage", {}).items():
            rows.append(
                {
                    "dataset_folder": folder,
                    "mapped_disease": info.get("mapped_disease"),
                    "stage": stage,
                    "image_count": count,
                    "record_type": "stage_breakdown",
                }
            )
    return rows
