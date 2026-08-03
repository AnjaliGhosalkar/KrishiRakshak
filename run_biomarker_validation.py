#!/usr/bin/env python
"""
Run the KrishiRakshak biomarker validation experiment (Step 2).

Examples:
    python run_biomarker_validation.py --max-per-class 50
    python run_biomarker_validation.py
    python run_biomarker_validation.py --analyze-only biomarker_analysis/biomarker_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from biomarker_validation.analyze import run_full_analysis
from biomarker_validation.constants import DEFAULT_RANDOM_SEED
from biomarker_validation.dataset_discovery import discover_dataset, discovery_to_rows
from biomarker_validation.extract_dataset import extract_dataset_biomarkers


def run_discovery(dataset_root: str, output_dir: str) -> dict:
    report = discover_dataset(dataset_root)
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "dataset_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        rows = discovery_to_rows(report)
        if rows:
            writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r.keys()}))
            writer.writeheader()
            writer.writerows(rows)

    with open(os.path.join(output_dir, "dataset_discovery.json"), "w", encoding="utf-8") as handle:
        json_report = {
            key: (dict(value) if hasattr(value, "items") and not isinstance(value, dict) else value)
            for key, value in report.items()
        }
        json_report["formats"] = dict(report["formats"])
        handle.write(json.dumps(json_report, indent=2, default=str))

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KrishiRakshak biomarker validation experiment")
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Path to dataset root (default: dataset)",
    )
    parser.add_argument(
        "--output-dir",
        default="biomarker_analysis",
        help="Output directory for analysis artifacts",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Sample up to N images per mapped disease label (reproducible). Omit for full dataset.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for sampling (default: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=768,
        help="Max image dimension passed to biomarker extractor",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip optional baseline classification experiment",
    )
    parser.add_argument(
        "--analyze-only",
        metavar="CSV",
        help="Skip extraction and analyze an existing biomarker results CSV",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_root = os.path.abspath(args.dataset_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Step 1: Discovering dataset structure...")
    discovery = run_discovery(dataset_root, output_dir)
    print(f"  Total images in dataset: {discovery['total_images']}")
    print("  Folder -> mapped disease:")
    for folder, info in discovery["categories"].items():
        if info.get("exists"):
            print(f"    {folder} -> {info['mapped_disease']}: {info['image_count']} images")

    if args.analyze_only:
        results_csv = os.path.abspath(args.analyze_only)
        extraction_meta = {
            "dataset_root": dataset_root,
            "total_images_in_dataset": discovery["total_images"],
            "images_selected": "unknown (analyze-only mode)",
            "successful_extractions": "unknown",
            "failed_extractions": "unknown",
            "sampling_mode": "precomputed CSV",
            "max_per_disease": None,
            "random_seed": None,
            "output_csv": results_csv,
        }
    else:
        results_csv = os.path.join(output_dir, "biomarker_results.csv")
        if args.max_per_class is not None:
            print(
                f"Step 2: Extracting biomarkers (SAMPLED: max {args.max_per_class} per disease, seed={args.random_seed})..."
            )
        else:
            print("Step 2: Extracting biomarkers (FULL DATASET)...")

        extraction_meta = extract_dataset_biomarkers(
            dataset_root=dataset_root,
            output_csv=results_csv,
            max_per_disease=args.max_per_class,
            random_seed=args.random_seed,
            max_dimension=args.max_dimension,
        )
        print(f"  Selected images: {extraction_meta['images_selected']}")
        print(f"  Successful: {extraction_meta['successful_extractions']}")
        print(f"  Failed: {extraction_meta['failed_extractions']}")
        print(f"  Results CSV: {results_csv}")

    print("Step 3-12: Running analysis...")
    analysis = run_full_analysis(
        results_csv=results_csv,
        output_dir=output_dir,
        discovery=discovery,
        extraction_meta=extraction_meta,
        run_baseline=not args.skip_baseline,
        random_seed=args.random_seed,
    )

    print(f"Analysis complete. Report: {os.path.join(output_dir, 'analysis_report.md')}")
    print("Sampling mode:", extraction_meta.get("sampling_mode"))
    if analysis["baseline_result"].get("performed"):
        print(
            "Baseline experiment: accuracy="
            f"{analysis['baseline_result']['accuracy']}, macro F1="
            f"{analysis['baseline_result']['macro_f1']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
