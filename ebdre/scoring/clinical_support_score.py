"""
Adaptive Clinical Support Score (CSS) for KrishiRakshak EBDRE.

CSS answers:
  "How strongly does the extracted visual evidence support the predicted disease?"

CSS is independent of CNN confidence:
  - CNN confidence  → model certainty about the class label
  - CSS             → agreement between observed biomarkers and the
                      Disease Evidence Profile for that prediction

IMPORTANT — provisional weights
-------------------------------
Support weights and biomarker reliability weights below are configurable
placeholders. They are NOT medically validated.

Weights can later be optimized using:
  - pilot / full-dataset biomarker validation
  - agreement with veterinary review
  - calibration against held-out cases

This module computes CSS only. It does not rank evidence or generate summaries.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, TypedDict

from ebdre.profiles.disease_profiles import BIOMARKER_KEYS

# ---------------------------------------------------------------------------
# Configurable weights (placeholders — tune with validation data later)
# ---------------------------------------------------------------------------

# How much each support category contributes to agreement.
# Positive = supports prediction; negative = conflicts with prediction.
# "unavailable" is never scored (ignored in the weighted average).
DEFAULT_SUPPORT_WEIGHTS: Dict[str, float] = {
    "strong_support": 1.0,
    "partial_support": 0.55,
    "weak_support": 0.20,
    "contradictory": -0.60,
}

# Relative trust in each biomarker channel for CSS.
# Higher = more influence when available. Low-reliability channels still
# contribute when present, but less than high-reliability ones.
DEFAULT_RELIABILITY_WEIGHTS: Dict[str, float] = {
    "texture": 1.00,  # high reliability (placeholder)
    "density": 0.75,  # medium reliability (placeholder)
    "color": 0.75,  # medium reliability (placeholder)
    "clustering": 0.40,  # low reliability (placeholder; often missing)
    "boundary": 0.40,  # low reliability (placeholder; often missing)
}

# Soft floor for availability dampening. Missing evidence reduces certainty
# gently (does not collapse the score).
_AVAILABILITY_FLOOR = 0.70

# Qualitative bands for evidence_strength from final CSS (0–100).
_STRENGTH_BINS = (
    (75.0, "strong"),
    (50.0, "moderate"),
    (25.0, "weak"),
    (0.0, "contradictory"),
)


class ClinicalSupportResult(TypedDict):
    clinical_support_score: int
    evidence_strength: str
    available_evidence: int
    confidence_level: str


def calculate_evidence_score(
    evidence_results: Mapping[str, Mapping[str, Any]],
    support_weights: Optional[Mapping[str, float]] = None,
    reliability_weights: Optional[Mapping[str, float]] = None,
) -> Optional[float]:
    """
    Weighted agreement score from available biomarker support labels.

    Formula
    -------
    For each biomarker b with support s != "unavailable":

        numerator   += support_weights[s] * reliability_weights[b]
        denominator += reliability_weights[b]

    evidence_score = numerator / denominator

    Returns None when no usable (non-unavailable) evidence exists.
    Typical range is approximately [contradictory_weight, strong_support_weight]
    when reliability weights are positive.
    """
    supports = support_weights or DEFAULT_SUPPORT_WEIGHTS
    reliability = reliability_weights or DEFAULT_RELIABILITY_WEIGHTS

    numerator = 0.0
    denominator = 0.0

    for key in BIOMARKER_KEYS:
        entry = evidence_results.get(key)
        if not entry:
            continue
        support = str(entry.get("support", "unavailable")).strip().lower()
        if support == "unavailable" or support not in supports:
            continue

        rel = float(reliability.get(key, 0.0))
        if rel <= 0.0:
            continue

        numerator += float(supports[support]) * rel
        denominator += rel

    if denominator <= 0.0:
        return None
    return numerator / denominator


def calculate_availability_factor(
    available_evidence: int,
    total_biomarkers: Optional[int] = None,
    floor: float = _AVAILABILITY_FLOOR,
) -> float:
    """
    Soft certainty factor from how many biomarkers were usable.

    Design
    ------
    Missing evidence should reduce certainty, not heavily punish the score.

        availability_factor = floor + (1 - floor) * (available / total)

    With default floor=0.70:
      5/5 available → 1.00
      3/5 available → 0.88
      1/5 available → 0.76
      0/5 available → 0.70  (score path handled separately when no evidence)

    Returns a value in [floor, 1.0].
    """
    total = total_biomarkers if total_biomarkers is not None else len(BIOMARKER_KEYS)
    if total <= 0:
        return floor

    available = max(0, int(available_evidence))
    ratio = min(1.0, available / float(total))
    floor_clamped = min(max(float(floor), 0.0), 1.0)
    return floor_clamped + (1.0 - floor_clamped) * ratio


def _map_evidence_score_to_100(
    evidence_score: float,
    support_weights: Mapping[str, float],
) -> float:
    """
    Map weighted agreement onto [0, 100] using configured support extremes.

        css_base = 100 * (score - min_w) / (max_w - min_w)
    """
    values = list(support_weights.values())
    min_w = min(values)
    max_w = max(values)
    if max_w == min_w:
        return 50.0
    normalized = (evidence_score - min_w) / (max_w - min_w)
    return max(0.0, min(100.0, 100.0 * normalized))


def _evidence_strength_label(css: float) -> str:
    for threshold, label in _STRENGTH_BINS:
        if css >= threshold:
            return label
    return "contradictory"


def _confidence_level(available_evidence: int) -> str:
    """Certainty about CSS based on evidence coverage (not agreement)."""
    if available_evidence >= 4:
        return "high"
    if available_evidence >= 2:
        return "medium"
    return "low"


def calculate_css(
    matching_result: Mapping[str, Any],
    support_weights: Optional[Mapping[str, float]] = None,
    reliability_weights: Optional[Mapping[str, float]] = None,
    availability_floor: float = _AVAILABILITY_FLOOR,
) -> ClinicalSupportResult:
    """
    Compute Adaptive Clinical Support Score from match_evidence_profile() output.

    Adaptive pipeline
    -----------------
    1. evidence_score   = reliability-weighted mean of support weights
                          (unavailable biomarkers ignored)
    2. css_base         = map evidence_score linearly into [0, 100]
    3. availability_factor = soft dampener from available / total biomarkers
    4. CSS              = round(css_base * availability_factor)

    When no biomarkers are available, CSS defaults to a neutral 50 with
    evidence_strength="insufficient" and confidence_level="low".

    Parameters
    ----------
    matching_result:
        Dict from match_evidence_profile(), or any mapping containing
        ``evidence_results`` (and optionally ``available_evidence``).
    support_weights / reliability_weights:
        Optional overrides of the provisional defaults.
    availability_floor:
        Soft floor for availability dampening (default 0.70).
    """
    supports = dict(support_weights or DEFAULT_SUPPORT_WEIGHTS)
    reliability = dict(reliability_weights or DEFAULT_RELIABILITY_WEIGHTS)

    evidence_results = matching_result.get("evidence_results") or {}
    if not isinstance(evidence_results, Mapping):
        evidence_results = {}

    # Prefer caller-provided count; otherwise derive from support labels.
    if "available_evidence" in matching_result:
        available = int(matching_result["available_evidence"])
    else:
        available = sum(
            1
            for key in BIOMARKER_KEYS
            if str((evidence_results.get(key) or {}).get("support", "unavailable"))
            .strip()
            .lower()
            != "unavailable"
        )

    evidence_score = calculate_evidence_score(
        evidence_results,
        support_weights=supports,
        reliability_weights=reliability,
    )

    if evidence_score is None or available <= 0:
        return {
            "clinical_support_score": 50,
            "evidence_strength": "insufficient",
            "available_evidence": available,
            "confidence_level": "low",
        }

    css_base = _map_evidence_score_to_100(evidence_score, supports)
    availability_factor = calculate_availability_factor(
        available_evidence=available,
        total_biomarkers=len(BIOMARKER_KEYS),
        floor=availability_floor,
    )
    css = int(round(max(0.0, min(100.0, css_base * availability_factor))))

    return {
        "clinical_support_score": css,
        "evidence_strength": _evidence_strength_label(float(css)),
        "available_evidence": available,
        "confidence_level": _confidence_level(available),
    }
