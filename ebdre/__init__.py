"""
Evidence-Based Disease Reasoning Engine (EBDRE) package.

Implemented so far:
  - Disease Evidence Profile knowledge base (profiles/)
  - Evidence Matching Engine (reasoning/)
  - Adaptive Clinical Support Score (scoring/)
  - Evidence Ranking (reasoning/)
  - Clinical Evidence Summary (reporting/)
"""

from ebdre.profiles.disease_profiles import (
    BIOMARKER_LEVELS,
    BIOMARKER_KEYS,
    DISEASES,
    STAGES,
    get_disease_profile,
    list_disease_profiles,
)
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
from ebdre.reporting.clinical_summary import (
    format_evidence_statement,
    generate_evidence_summary,
)
from ebdre.scoring.clinical_support_score import (
    DEFAULT_RELIABILITY_WEIGHTS,
    DEFAULT_SUPPORT_WEIGHTS,
    calculate_availability_factor,
    calculate_css,
    calculate_evidence_score,
)

__all__ = [
    "BIOMARKER_KEYS",
    "BIOMARKER_LEVELS",
    "DEFAULT_BIOMARKER_THRESHOLDS",
    "DEFAULT_RANKING_RELIABILITY_WEIGHTS",
    "DEFAULT_RANKING_SUPPORT_WEIGHTS",
    "DEFAULT_RELIABILITY_WEIGHTS",
    "DEFAULT_SUPPORT_WEIGHTS",
    "DISEASES",
    "STAGES",
    "SUPPORT_CATEGORIES",
    "calculate_availability_factor",
    "calculate_css",
    "calculate_evidence_importance",
    "calculate_evidence_score",
    "compare_evidence",
    "format_evidence_statement",
    "generate_evidence_summary",
    "get_disease_profile",
    "list_disease_profiles",
    "match_evidence_profile",
    "normalize_biomarker_value",
    "rank_evidence",
]
