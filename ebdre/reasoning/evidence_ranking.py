"""
Evidence Ranking for KrishiRakshak EBDRE.

Ranks available visual biomarkers by how much they contribute to supporting
(or contradicting) the predicted disease-stage profile.

Answers:
  "Which visual evidence contributed most to this prediction?"

IMPORTANT — provisional weights
-------------------------------
Ranking / reliability weights below are configurable placeholders.
They are NOT medically validated and can later be tuned with validation data.

This module ranks evidence only. It does not generate clinical summaries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, TypedDict

from ebdre.profiles.disease_profiles import BIOMARKER_KEYS

# ---------------------------------------------------------------------------
# Configurable ranking weights (placeholders — tune later)
# ---------------------------------------------------------------------------

# Support priority for ranking. Unavailable evidence is ignored entirely.
# Aligned in spirit with CSS support weights; kept local so ranking does not
# depend on or mutate the scoring module.
DEFAULT_RANKING_SUPPORT_WEIGHTS: Dict[str, float] = {
    "strong_support": 1.00,  # highest priority
    "partial_support": 0.55,  # medium
    "weak_support": 0.20,  # low
    "contradictory": -0.60,  # negative contribution
}

# Biomarker reliability / disease-evidence importance channel weights.
# Callers may pass CSS DEFAULT_RELIABILITY_WEIGHTS (or tuned values) instead.
DEFAULT_RANKING_RELIABILITY_WEIGHTS: Dict[str, float] = {
    "texture": 1.00,  # high
    "density": 0.75,  # medium
    "color": 0.75,  # medium
    "clustering": 0.40,  # low
    "boundary": 0.40,  # low
}

# Map combined importance_score → qualitative importance label.
# Contradictory (negative) scores are labeled separately.
_IMPORTANCE_BINS = (
    (0.70, "high"),
    (0.35, "medium"),
    (0.00, "low"),
)


class RankedEvidenceItem(TypedDict):
    biomarker: str
    importance: str
    support: str


class EvidenceRankingResult(TypedDict):
    ranked_evidence: List[RankedEvidenceItem]


def calculate_evidence_importance(
    biomarker: str,
    support: str,
    support_weights: Optional[Mapping[str, float]] = None,
    reliability_weights: Optional[Mapping[str, float]] = None,
) -> Optional[float]:
    """
    Compute a numeric importance score for one biomarker.

    Formula
    -------
        importance_score = support_weight[support] * reliability_weight[biomarker]

    Returns
    -------
    float score, or None when the biomarker should be ignored
    (unavailable support, unknown support, or non-positive reliability).
    """
    supports = support_weights or DEFAULT_RANKING_SUPPORT_WEIGHTS
    reliability = reliability_weights or DEFAULT_RANKING_RELIABILITY_WEIGHTS

    support_key = (support or "unavailable").strip().lower()
    if support_key == "unavailable" or support_key not in supports:
        return None

    biomarker_key = (biomarker or "").strip().lower()
    rel = float(reliability.get(biomarker_key, 0.0))
    if rel <= 0.0:
        return None

    return float(supports[support_key]) * rel


def _importance_label(importance_score: float) -> str:
    """Convert numeric importance score to a qualitative label."""
    if importance_score < 0.0:
        return "contradictory"
    for threshold, label in _IMPORTANCE_BINS:
        if importance_score >= threshold:
            return label
    return "low"


def _extract_evidence_results(
    matching_result: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    """Accept full match_evidence_profile() output or a bare evidence_results map."""
    if "evidence_results" in matching_result and isinstance(
        matching_result["evidence_results"], Mapping
    ):
        return matching_result["evidence_results"]
    # Bare map: { "texture": {"support": "..."}, ... }
    return matching_result  # type: ignore[return-value]


def rank_evidence(
    matching_result: Mapping[str, Any],
    support_weights: Optional[Mapping[str, float]] = None,
    reliability_weights: Optional[Mapping[str, float]] = None,
) -> EvidenceRankingResult:
    """
    Rank available biomarkers by contribution to the prediction.

    Parameters
    ----------
    matching_result:
        Output of ``match_evidence_profile()``, or a dict of
        ``{biomarker: {"support": ...}}``.
    support_weights:
        Optional override of DEFAULT_RANKING_SUPPORT_WEIGHTS.
    reliability_weights:
        Optional biomarker reliability weights. Pass CSS
        ``DEFAULT_RELIABILITY_WEIGHTS`` (or tuned weights) to keep CSS and
        ranking aligned.

    Returns
    -------
    ``{"ranked_evidence": [...]}`` sorted by importance_score descending.
    Unavailable biomarkers are omitted. Contradictory items sort last among
    scored evidence (negative scores).
    """
    supports = support_weights or DEFAULT_RANKING_SUPPORT_WEIGHTS
    reliability = reliability_weights or DEFAULT_RANKING_RELIABILITY_WEIGHTS
    evidence_results = _extract_evidence_results(matching_result)

    scored: List[tuple[float, RankedEvidenceItem]] = []

    for key in BIOMARKER_KEYS:
        entry = evidence_results.get(key)
        if not isinstance(entry, Mapping):
            continue

        support = str(entry.get("support", "unavailable")).strip().lower()
        score = calculate_evidence_importance(
            biomarker=key,
            support=support,
            support_weights=supports,
            reliability_weights=reliability,
        )
        if score is None:
            continue

        scored.append(
            (
                score,
                {
                    "biomarker": key,
                    "importance": _importance_label(score),
                    "support": support,
                },
            )
        )

    # Highest contribution first; contradictory (negative) sinks to the bottom.
    scored.sort(key=lambda item: item[0], reverse=True)

    return {
        "ranked_evidence": [item for _, item in scored],
    }
