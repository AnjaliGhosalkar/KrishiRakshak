"""
EBDRE pipeline orchestrator for KrishiRakshak.

Runs biomarker extraction + evidence matching + CSS + ranking + summary
after CNN/RF prediction. Does not alter model code or EBDRE internals.
"""

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np

from biomarker_extraction import extract_biomarkers
from ebdre.reasoning.evidence_matching import match_evidence_profile
from ebdre.reasoning.evidence_ranking import rank_evidence
from ebdre.reporting.clinical_summary import generate_evidence_summary
from ebdre.scoring.clinical_support_score import calculate_css

Numeric = Union[int, float]


def _profile_stage(stage: str) -> str:
    """
    Map prediction stage to a Disease Evidence Profile stage key.

    Profiles define early / moderate / severe only. Healthy predictions use
    stage \"none\" in the API; map that (and unknown) to a structural
    placeholder without changing profile data.
    """
    stage_key = (stage or "").strip().lower()
    if stage_key in {"early", "moderate", "severe"}:
        return stage_key
    return "moderate"


def run_ebdre_pipeline(
    image_rgb: np.ndarray,
    disease: str,
    stage: str,
    cnn_confidence: Numeric,
) -> Dict[str, Any]:
    """
    Execute the full EBDRE reasoning layer on an RGB uint8 image.

    Parameters
    ----------
    image_rgb:
        HxWx3 RGB uint8 array (original upload, not MobileNet-preprocessed).
    disease:
        Predicted disease key (e.g. \"lumpy\", \"healthy\").
    stage:
        Predicted stage (e.g. \"moderate\", \"none\").
    cnn_confidence:
        Model confidence as a percentage number (0–100).

    Returns
    -------
    Dict suitable for merging into the Flask /predict JSON response.
    """
    disease_key = (disease or "").strip().lower()
    stage_display = (stage or "").strip()
    stage_key = _profile_stage(stage_display)

    # 1) Biomarker extraction (existing prototype — algorithms unchanged)
    biomarker_result = extract_biomarkers(image_rgb)
    observed_biomarkers = {
        "lesion_density": biomarker_result.get("lesion_density"),
        "texture_roughness": biomarker_result.get("texture_roughness"),
        "color_variation": biomarker_result.get("color_variation"),
        "lesion_clustering": biomarker_result.get("lesion_clustering"),
        "boundary_irregularity": biomarker_result.get("boundary_irregularity"),
    }

    # 2) Evidence matching against Disease Evidence Profile
    match_result = match_evidence_profile(
        disease=disease_key,
        stage=stage_key,
        biomarkers=observed_biomarkers,
    )

    # 3) Adaptive Clinical Support Score
    css_result = calculate_css(match_result)

    # 4) Evidence ranking
    ranking_result = rank_evidence(match_result)

    # 5) Clinical evidence summary
    clinical_summary = generate_evidence_summary(
        disease=disease_key.capitalize(),
        stage=stage_display.capitalize() if stage_display else stage_key.capitalize(),
        cnn_confidence=cnn_confidence,
        clinical_support_score=css_result["clinical_support_score"],
        ranked_evidence=ranking_result,
        matching_result=match_result,
        evidence_strength=css_result["evidence_strength"],
    )

    return {
        "cnn_confidence": round(float(cnn_confidence), 2),
        "clinical_support_score": css_result["clinical_support_score"],
        "evidence_strength": css_result["evidence_strength"],
        "available_evidence": css_result["available_evidence"],
        "confidence_level": css_result["confidence_level"],
        "ranked_evidence": ranking_result["ranked_evidence"],
        "clinical_summary": clinical_summary,
        "biomarkers": observed_biomarkers,
        "evidence_matching": {
            "disease": match_result["disease"],
            "stage": match_result["stage"],
            "evidence_results": match_result["evidence_results"],
            "available_evidence": match_result["available_evidence"],
            "missing_evidence": match_result["missing_evidence"],
        },
    }


def safe_run_ebdre_pipeline(
    image_rgb: np.ndarray,
    disease: str,
    stage: str,
    cnn_confidence: Numeric,
) -> Dict[str, Any]:
    """
    Run EBDRE without failing the primary CNN/RF prediction response.

    On error, returns null EBDRE fields plus an error message so the
    existing prediction payload remains usable.
    """
    try:
        return run_ebdre_pipeline(
            image_rgb=image_rgb,
            disease=disease,
            stage=stage,
            cnn_confidence=cnn_confidence,
        )
    except Exception as exc:  # noqa: BLE001 — keep prediction path alive
        return {
            "cnn_confidence": round(float(cnn_confidence), 2),
            "clinical_support_score": None,
            "evidence_strength": None,
            "available_evidence": 0,
            "confidence_level": "low",
            "ranked_evidence": [],
            "clinical_summary": None,
            "biomarkers": None,
            "evidence_matching": None,
            "ebdre_error": str(exc),
        }
