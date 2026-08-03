"""
Clinical Evidence Summary Generator for KrishiRakshak EBDRE.

Converts EBDRE matching / ranking / CSS outputs into a concise,
veterinarian-facing evidence report.

IMPORTANT
---------
Statements are template-based and derived only from:
  - biomarker name
  - support level
  - importance ranking
  - Clinical Support Score (for the overall statement)

They do NOT add unsupported medical diagnoses, treatment advice, or
clinical claims beyond what the EBDRE pipeline already computed.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TypedDict, Union

from ebdre.profiles.disease_profiles import BIOMARKER_KEYS

# Display names for short biomarker keys (descriptive labels only).
BIOMARKER_DISPLAY_NAMES: Dict[str, str] = {
    "texture": "texture",
    "density": "lesion density",
    "color": "color variation",
    "clustering": "lesion clustering",
    "boundary": "boundary irregularity",
}

# Templates keyed by support category. Placeholders: {biomarker}
# Wording stays observational — no disease causation claims.
_SUPPORT_STATEMENT_TEMPLATES: Dict[str, str] = {
    "strong_support": "Strong {biomarker} pattern detected; aligns with expected profile",
    "partial_support": "{biomarker_title} pattern partially supports the prediction",
    "weak_support": "Weak {biomarker} signal relative to the expected profile",
    "contradictory": "{biomarker_title} pattern conflicts with the expected profile",
}

_UNAVAILABLE_TEMPLATE = "{biomarker_title} unavailable"

# Overall statement templates from CSS bands (not medical diagnosis).
_OVERALL_TEMPLATES: Dict[str, str] = {
    "strong": "Visual evidence strongly supports the predicted disease.",
    "moderate": "Visual evidence moderately supports the predicted disease.",
    "weak": "Visual evidence only weakly supports the predicted disease.",
    "contradictory": "Visual evidence conflicts with the predicted disease profile.",
    "insufficient": "Insufficient visual biomarker evidence to assess support for the prediction.",
}


class ClinicalEvidenceSummary(TypedDict):
    title: str
    prediction: str
    stage: str
    cnn_confidence: Union[int, float, str]
    clinical_support_score: Union[int, float]
    supporting_evidence: List[str]
    unavailable_evidence: List[str]
    conflicting_evidence: List[str]
    overall_statement: str


def _display_name(biomarker: str) -> str:
    key = (biomarker or "").strip().lower()
    return BIOMARKER_DISPLAY_NAMES.get(key, key.replace("_", " "))


def _title_case_biomarker(biomarker: str) -> str:
    name = _display_name(biomarker)
    return name[:1].upper() + name[1:] if name else name


def format_evidence_statement(
    biomarker: str,
    support: str,
    importance: Optional[str] = None,
) -> Optional[str]:
    """
    Format one template-based evidence statement.

    Uses only biomarker name, support level, and optional importance.
    Returns None for unavailable / unknown support (handled separately).
    """
    support_key = (support or "").strip().lower()
    if support_key == "unavailable":
        return _UNAVAILABLE_TEMPLATE.format(
            biomarker_title=_title_case_biomarker(biomarker),
        )

    template = _SUPPORT_STATEMENT_TEMPLATES.get(support_key)
    if template is None:
        return None

    statement = template.format(
        biomarker=_display_name(biomarker),
        biomarker_title=_title_case_biomarker(biomarker),
    )

    # Optional importance cue from ranking — non-diagnostic annotation only.
    importance_key = (importance or "").strip().lower()
    if importance_key == "high":
        statement = f"High-importance evidence: {statement}"
    elif importance_key == "medium":
        statement = f"Medium-importance evidence: {statement}"
    elif importance_key == "low":
        statement = f"Lower-importance evidence: {statement}"

    return statement


def _normalize_ranked_evidence(
    ranked_evidence: Union[Sequence[Mapping[str, Any]], Mapping[str, Any], None],
) -> List[Mapping[str, Any]]:
    """Accept a list, or ``{"ranked_evidence": [...]}`` from rank_evidence()."""
    if ranked_evidence is None:
        return []
    if isinstance(ranked_evidence, Mapping) and "ranked_evidence" in ranked_evidence:
        items = ranked_evidence.get("ranked_evidence") or []
        return list(items) if isinstance(items, Sequence) else []
    if isinstance(ranked_evidence, Sequence):
        return [item for item in ranked_evidence if isinstance(item, Mapping)]
    return []


def _collect_unavailable(
    matching_result: Optional[Mapping[str, Any]],
    unavailable_biomarkers: Optional[Iterable[str]],
) -> List[str]:
    """Build unavailable statements from match output and/or an explicit list."""
    keys: List[str] = []

    if unavailable_biomarkers is not None:
        for key in unavailable_biomarkers:
            key_l = str(key).strip().lower()
            if key_l and key_l not in keys:
                keys.append(key_l)

    if matching_result is not None:
        evidence_results = matching_result.get("evidence_results", matching_result)
        if isinstance(evidence_results, Mapping):
            for key in BIOMARKER_KEYS:
                entry = evidence_results.get(key)
                if not isinstance(entry, Mapping):
                    continue
                support = str(entry.get("support", "")).strip().lower()
                if support == "unavailable" and key not in keys:
                    keys.append(key)

    return [
        _UNAVAILABLE_TEMPLATE.format(biomarker_title=_title_case_biomarker(key))
        for key in keys
    ]


def _overall_statement_from_css(
    clinical_support_score: Union[int, float],
    evidence_strength: Optional[str] = None,
) -> str:
    """
    Select overall statement from CSS score / optional evidence_strength label.

    Bands align with CSS qualitative labels; wording is support-level only.
    """
    if evidence_strength:
        strength = evidence_strength.strip().lower()
        if strength in _OVERALL_TEMPLATES:
            return _OVERALL_TEMPLATES[strength]

    try:
        score = float(clinical_support_score)
    except (TypeError, ValueError):
        return _OVERALL_TEMPLATES["insufficient"]

    if score >= 75:
        return _OVERALL_TEMPLATES["strong"]
    if score >= 50:
        return _OVERALL_TEMPLATES["moderate"]
    if score >= 25:
        return _OVERALL_TEMPLATES["weak"]
    return _OVERALL_TEMPLATES["contradictory"]


def _normalize_confidence(cnn_confidence: Any) -> Union[int, float, str]:
    """Preserve numeric confidence; strip a trailing % if provided as string."""
    if isinstance(cnn_confidence, (int, float)):
        return cnn_confidence
    if isinstance(cnn_confidence, str):
        text = cnn_confidence.strip()
        if text.endswith("%"):
            text = text[:-1].strip()
            try:
                value = float(text)
                return int(value) if value.is_integer() else value
            except ValueError:
                return cnn_confidence.strip()
        try:
            value = float(text)
            return int(value) if value.is_integer() else value
        except ValueError:
            return cnn_confidence.strip()
    return cnn_confidence


def generate_evidence_summary(
    disease: str,
    stage: str,
    cnn_confidence: Any,
    clinical_support_score: Union[int, float],
    ranked_evidence: Union[Sequence[Mapping[str, Any]], Mapping[str, Any], None],
    matching_result: Optional[Mapping[str, Any]] = None,
    unavailable_biomarkers: Optional[Iterable[str]] = None,
    evidence_strength: Optional[str] = None,
) -> ClinicalEvidenceSummary:
    """
    Build a structured clinical evidence summary for veterinarian review.

    Parameters
    ----------
    disease / stage:
        Predicted disease and severity stage (display strings as provided).
    cnn_confidence:
        Model confidence (numeric or percent string). Shown separately from CSS.
    clinical_support_score:
        Adaptive CSS (0–100) from calculate_css().
    ranked_evidence:
        List from rank_evidence(), or the full ``{"ranked_evidence": [...]}`` dict.
    matching_result:
        Optional match_evidence_profile() output used to list unavailable biomarkers.
    unavailable_biomarkers:
        Optional explicit list of unavailable short biomarker keys.
    evidence_strength:
        Optional CSS ``evidence_strength`` label for the overall statement.
    """
    ranked_items = _normalize_ranked_evidence(ranked_evidence)

    supporting: List[str] = []
    conflicting: List[str] = []

    for item in ranked_items:
        biomarker = str(item.get("biomarker", "")).strip().lower()
        support = str(item.get("support", "")).strip().lower()
        importance = str(item.get("importance", "")).strip().lower() or None

        statement = format_evidence_statement(
            biomarker=biomarker,
            support=support,
            importance=importance,
        )
        if statement is None:
            continue

        if support == "contradictory":
            conflicting.append(statement)
        elif support in {"strong_support", "partial_support", "weak_support"}:
            supporting.append(statement)

    unavailable = _collect_unavailable(matching_result, unavailable_biomarkers)

    return {
        "title": "Clinical Evidence Summary",
        "prediction": str(disease).strip(),
        "stage": str(stage).strip(),
        "cnn_confidence": _normalize_confidence(cnn_confidence),
        "clinical_support_score": clinical_support_score,
        "supporting_evidence": supporting,
        "unavailable_evidence": unavailable,
        "conflicting_evidence": conflicting,
        "overall_statement": _overall_statement_from_css(
            clinical_support_score,
            evidence_strength=evidence_strength,
        ),
    }
