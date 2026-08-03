"""
Evidence Matching Engine for KrishiRakshak EBDRE.

Compares observed visual biomarkers from an image against the expected
qualitative levels in a Disease Evidence Profile.

This module performs matching only. It does NOT compute Clinical Support
Score, evidence ranking, or clinical summaries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict, Union

from ebdre.profiles.disease_profiles import (
    BIOMARKER_KEY_TO_EXTRACTOR,
    BIOMARKER_KEYS,
    BIOMARKER_LEVELS,
    get_disease_profile,
)

# Ordered ordinal scale for distance-based support (excludes "unavailable").
_ORDINAL_LEVELS: List[str] = [
    "very_low",
    "low",
    "medium",
    "high",
    "very_high",
]

_LEVEL_INDEX: Dict[str, int] = {level: i for i, level in enumerate(_ORDINAL_LEVELS)}

SUPPORT_CATEGORIES: List[str] = [
    "strong_support",
    "partial_support",
    "weak_support",
    "contradictory",
    "unavailable",
]

# (lower_inclusive, upper_exclusive, level) — last bin is upper-inclusive via clamp.
DEFAULT_BIOMARKER_THRESHOLDS: List[Tuple[float, float, str]] = [
    (0.0, 0.2, "very_low"),
    (0.2, 0.4, "low"),
    (0.4, 0.6, "medium"),
    (0.6, 0.8, "high"),
    (0.8, 1.0, "very_high"),
]

# Accept both short profile keys and biomarker_extraction full names.
_EXTRACTOR_TO_SHORT: Dict[str, str] = {
    full: short for short, full in BIOMARKER_KEY_TO_EXTRACTOR.items()
}

NumericOrNone = Union[float, int, None]


class BiomarkerEvidenceResult(TypedDict):
    observed: str
    expected: str
    support: str


class EvidenceMatchingResult(TypedDict):
    disease: str
    stage: str
    evidence_results: Dict[str, BiomarkerEvidenceResult]
    available_evidence: int
    missing_evidence: int


def normalize_biomarker_value(
    value: NumericOrNone,
    thresholds: Optional[Sequence[Tuple[float, float, str]]] = None,
) -> str:
    """
    Convert a numerical biomarker value in [0, 1] to a qualitative level.

    Parameters
    ----------
    value:
        Observed biomarker score, or None / null when extraction failed.
    thresholds:
        Optional sequence of (lower_inclusive, upper_exclusive, level).
        Defaults to DEFAULT_BIOMARKER_THRESHOLDS. The final bin also accepts
        values equal to its upper bound (typically 1.0).

    Returns
    -------
    One of: very_low, low, medium, high, very_high, unavailable.
    """
    if value is None:
        return "unavailable"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unavailable"

    if numeric != numeric:  # NaN
        return "unavailable"

    bins = list(thresholds) if thresholds is not None else DEFAULT_BIOMARKER_THRESHOLDS
    if not bins:
        raise ValueError("thresholds must contain at least one bin")

    # Clamp into the configured range for stable binning.
    lo_bound = bins[0][0]
    hi_bound = bins[-1][1]
    if numeric < lo_bound:
        numeric = lo_bound
    elif numeric > hi_bound:
        numeric = hi_bound

    for lower, upper, level in bins[:-1]:
        if lower <= numeric < upper:
            return level

    last_lower, last_upper, last_level = bins[-1]
    if last_lower <= numeric <= last_upper:
        return last_level

    # Fallback — should not occur after clamping.
    return "unavailable"


def compare_evidence(observed_level: str, expected_level: str) -> str:
    """
    Compare qualitative observed vs expected biomarker levels.

    Returns one of SUPPORT_CATEGORIES:
      strong_support  — exact match
      partial_support — adjacent levels (distance 1)
      weak_support    — distance 2
      contradictory   — distance >= 3
      unavailable     — either side missing / unavailable
    """
    observed = (observed_level or "unavailable").strip().lower()
    expected = (expected_level or "unavailable").strip().lower()

    if observed == "unavailable" or expected == "unavailable":
        return "unavailable"

    if observed not in _LEVEL_INDEX or expected not in _LEVEL_INDEX:
        return "unavailable"

    distance = abs(_LEVEL_INDEX[observed] - _LEVEL_INDEX[expected])
    if distance == 0:
        return "strong_support"
    if distance == 1:
        return "partial_support"
    if distance == 2:
        return "weak_support"
    return "contradictory"


def _normalize_observed_dict(
    biomarkers: Mapping[str, Any],
) -> Dict[str, Any]:
    """Map incoming biomarker keys to short profile keys."""
    normalized: Dict[str, Any] = {}
    for key, value in biomarkers.items():
        key_l = str(key).strip().lower()
        if key_l in BIOMARKER_KEYS:
            short = key_l
        elif key_l in _EXTRACTOR_TO_SHORT:
            short = _EXTRACTOR_TO_SHORT[key_l]
        else:
            continue
        # Prefer first-seen short key; do not overwrite if already set.
        if short not in normalized:
            normalized[short] = value
    return normalized


def match_evidence_profile(
    disease: str,
    stage: str,
    biomarkers: Mapping[str, Any],
    thresholds: Optional[Sequence[Tuple[float, float, str]]] = None,
) -> EvidenceMatchingResult:
    """
    Match observed biomarkers against the Disease Evidence Profile.

    Parameters
    ----------
    disease:
        Predicted disease (e.g. "lumpy").
    stage:
        Predicted severity stage (e.g. "moderate").
    biomarkers:
        Observed values keyed by short names (texture, density, ...) or
        extractor names (texture_roughness, lesion_density, ...).
        Missing / null values are treated as unavailable.
    thresholds:
        Optional qualitative bin edges for normalize_biomarker_value().

    Returns
    -------
    Structured evidence matching result with per-biomarker support labels
    and available / missing evidence counts.
    """
    profile = get_disease_profile(disease, stage)
    expected_map = profile["biomarkers"]
    observed_map = _normalize_observed_dict(biomarkers)

    evidence_results: Dict[str, BiomarkerEvidenceResult] = {}
    available = 0
    missing = 0

    for key in BIOMARKER_KEYS:
        raw_observed = observed_map.get(key, None)
        # Allow pre-normalized qualitative strings as well as numerics.
        if isinstance(raw_observed, str):
            observed_level = raw_observed.strip().lower()
            if observed_level not in BIOMARKER_LEVELS:
                observed_level = normalize_biomarker_value(None, thresholds=thresholds)
        else:
            observed_level = normalize_biomarker_value(raw_observed, thresholds=thresholds)

        expected_level = str(expected_map.get(key, "unavailable")).strip().lower()
        support = compare_evidence(observed_level, expected_level)

        evidence_results[key] = {
            "observed": observed_level,
            "expected": expected_level,
            "support": support,
        }

        if support == "unavailable":
            missing += 1
        else:
            available += 1

    return {
        "disease": profile["disease"],
        "stage": profile["stage"],
        "evidence_results": evidence_results,
        "available_evidence": available,
        "missing_evidence": missing,
    }
