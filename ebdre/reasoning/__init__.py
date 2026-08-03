"""EBDRE reasoning subpackage (evidence matching and ranking)."""

from ebdre.reasoning.evidence_matching import (
    DEFAULT_BIOMARKER_THRESHOLDS,
    SUPPORT_CATEGORIES,
    compare_evidence,
    match_evidence_profile,
    normalize_biomarker_value,
)
from ebdre.reasoning.evidence_ranking import (
    DEFAULT_RANKING_RELIABILITY_WEIGHTS,
    DEFAULT_RANKING_SUPPORT_WEIGHTS,
    calculate_evidence_importance,
    rank_evidence,
)

__all__ = [
    "DEFAULT_BIOMARKER_THRESHOLDS",
    "DEFAULT_RANKING_RELIABILITY_WEIGHTS",
    "DEFAULT_RANKING_SUPPORT_WEIGHTS",
    "SUPPORT_CATEGORIES",
    "calculate_evidence_importance",
    "compare_evidence",
    "match_evidence_profile",
    "normalize_biomarker_value",
    "rank_evidence",
]
