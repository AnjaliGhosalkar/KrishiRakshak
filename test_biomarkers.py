#!/usr/bin/env python
"""
Standalone test and batch runner for experimental visual biomarker extraction.

Usage:
    python test_biomarkers.py <image_path>
    python test_biomarkers.py <image_path> --debug biomarker_debug
    python test_biomarkers.py --dir dataset/lumpy/moderate --csv biomarker_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Iterable, List, Optional

from biomarker_extraction.extractor import extract_biomarkers

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def _collect_images_from_dir(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    image_paths: List[str] = []
    for root, _, files in os.walk(directory):
        for filename in files:
            full_path = os.path.join(root, filename)
            if _is_image_file(full_path):
                image_paths.append(full_path)
    image_paths.sort()
    return image_paths


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.4f}"


def print_biomarker_report(image_label: str, result: dict) -> None:
    """Print a readable biomarker report."""
    print(f"Image:\n{image_label}\n")
    print("Visual Biomarkers")
    print("-------------------------")
    print(f"Lesion Density:        {_format_value(result.get('lesion_density'))}")
    print(f"Texture Roughness:     {_format_value(result.get('texture_roughness'))}")
    print(f"Color Variation:       {_format_value(result.get('color_variation'))}")
    print(f"Lesion Clustering:     {_format_value(result.get('lesion_clustering'))}")
    print(f"Boundary Irregularity: {_format_value(result.get('boundary_irregularity'))}")

    metadata = result.get("metadata", {})
    if metadata:
        print("\nMetadata")
        print("-------------------------")
        print(f"Contours detected:     {metadata.get('contour_count')}")
        print(f"Lesion pixels:         {metadata.get('lesion_pixel_count')}")
        print(f"Analyzed pixels:       {metadata.get('analyzed_pixel_count')}")

    debug_path = result.get("debug_image_path")
    if debug_path:
        print(f"\nDebug visualization:   {debug_path}")


def write_csv(results: Iterable[dict], csv_path: str) -> None:
    """Write biomarker results to CSV for research experiments."""
    fieldnames = [
        "image_name",
        "image_path",
        "lesion_density",
        "texture_roughness",
        "color_variation",
        "lesion_clustering",
        "boundary_irregularity",
        "contour_count",
    ]

    output_dir = os.path.dirname(os.path.abspath(csv_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def process_image(
    image_path: str,
    save_debug: bool = False,
    debug_output_dir: Optional[str] = None,
    max_dimension: int = 768,
) -> dict:
    """Extract biomarkers for one image and return CSV-ready row."""
    result = extract_biomarkers(
        image_path,
        save_debug=save_debug,
        debug_output_dir=debug_output_dir,
        max_dimension=max_dimension,
    )
    metadata = result.get("metadata", {})
    return {
        "image_name": os.path.basename(image_path),
        "image_path": image_path,
        "lesion_density": result.get("lesion_density"),
        "texture_roughness": result.get("texture_roughness"),
        "color_variation": result.get("color_variation"),
        "lesion_clustering": result.get("lesion_clustering"),
        "boundary_irregularity": result.get("boundary_irregularity"),
        "contour_count": metadata.get("contour_count"),
        "_full_result": result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract experimental visual biomarkers from livestock skin images.",
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to a single image file.",
    )
    parser.add_argument(
        "--dir",
        dest="directory",
        help="Process all images in a directory recursively.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Write batch results to CSV.",
    )
    parser.add_argument(
        "--debug",
        dest="debug_output_dir",
        help="Directory for debug visualization images.",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=768,
        help="Maximum image dimension for biomarker preprocessing (default: 768).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    save_debug = bool(args.debug_output_dir)
    debug_output_dir = args.debug_output_dir or "biomarker_debug"

    if args.directory:
        image_paths = _collect_images_from_dir(args.directory)
        if not image_paths:
            print(f"No images found in directory: {args.directory}")
            return 1

        csv_rows = []
        last_result = None
        for image_path in image_paths:
            row = process_image(
                image_path,
                save_debug=save_debug,
                debug_output_dir=debug_output_dir,
                max_dimension=args.max_dimension,
            )
            last_result = row["_full_result"]
            csv_rows.append({key: value for key, value in row.items() if not key.startswith("_")})

        if args.csv_path:
            write_csv(csv_rows, args.csv_path)
            print(f"CSV written to: {args.csv_path}")

        if len(image_paths) == 1 and last_result is not None:
            print_biomarker_report(image_paths[0], last_result)
        else:
            print(f"Processed {len(image_paths)} images.")
        return 0

    if not args.image_path:
        parser.print_help()
        return 1

    if not os.path.isfile(args.image_path):
        print(f"Image not found: {args.image_path}")
        return 1

    result = extract_biomarkers(
        args.image_path,
        save_debug=save_debug,
        debug_output_dir=debug_output_dir,
        max_dimension=args.max_dimension,
    )
    print_biomarker_report(args.image_path, result)

    if args.csv_path:
        row = {
            "image_name": os.path.basename(args.image_path),
            "image_path": args.image_path,
            "lesion_density": result.get("lesion_density"),
            "texture_roughness": result.get("texture_roughness"),
            "color_variation": result.get("color_variation"),
            "lesion_clustering": result.get("lesion_clustering"),
            "boundary_irregularity": result.get("boundary_irregularity"),
            "contour_count": result.get("metadata", {}).get("contour_count"),
        }
        write_csv([row], args.csv_path)
        print(f"\nCSV written to: {args.csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
