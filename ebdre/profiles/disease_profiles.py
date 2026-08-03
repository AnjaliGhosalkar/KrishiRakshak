"""
Disease Evidence Profiles for KrishiRakshak EBDRE.

Defines expected visual biomarker patterns for each disease and severity stage.
Future EBDRE matching will compare Observed Biomarkers against these profiles.

IMPORTANT — provisional knowledge base
-------------------------------------
These profiles are structural placeholders only. They are NOT validated
veterinary or clinical reference values.

Profiles will later be refined using:
  - pilot biomarker analysis (biomarker_analysis/)
  - full dataset statistics
  - veterinary / domain references

Do not treat current qualitative levels as medical truth.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Mapping, Optional, TypedDict


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

DISEASES: List[str] = [
    "foot",
    "mouth",
    "lumpy",
    "mastitis",
    "healthy",
]

STAGES: List[str] = [
    "early",
    "moderate",
    "severe",
]

# Qualitative levels used in expected biomarker fields.
BIOMARKER_LEVELS: List[str] = [
    "very_low",
    "low",
    "medium",
    "high",
    "very_high",
    "unavailable",
]

# Short keys stored in each profile (stable API for future matching modules).
BIOMARKER_KEYS: List[str] = [
    "texture",
    "density",
    "color",
    "clustering",
    "boundary",
]

# Mapping from short profile keys → names used by biomarker_extraction/.
BIOMARKER_KEY_TO_EXTRACTOR: Mapping[str, str] = {
    "texture": "texture_roughness",
    "density": "lesion_density",
    "color": "color_variation",
    "clustering": "lesion_clustering",
    "boundary": "boundary_irregularity",
}


class BiomarkerExpectations(TypedDict):
    texture: str
    density: str
    color: str
    clustering: str
    boundary: str


class DiseaseEvidenceProfile(TypedDict):
    disease: str
    stage: str
    biomarkers: BiomarkerExpectations
    # True until levels are refined from pilot / full-dataset / vet sources.
    provisional: bool


def _profile(
    disease: str,
    stage: str,
    texture: str,
    density: str,
    color: str,
    clustering: str,
    boundary: str,
) -> DiseaseEvidenceProfile:
    """Build one provisional disease-stage evidence profile."""
    return {
        "disease": disease,
        "stage": stage,
        "biomarkers": {
            "texture": texture,
            "density": density,
            "color": color,
            "clustering": clustering,
            "boundary": boundary,
        },
        "provisional": True,
    }


# ---------------------------------------------------------------------------
# Disease Evidence Profile registry (15 profiles: 5 diseases × 3 stages)
#
# All biomarker levels below are PLACEHOLDERS.
# They encode only a coarse expected progression skeleton so the knowledge
# base is complete and expandable. Replace after statistical / clinical review.
#
# Notes:
# - clustering / boundary often return unavailable in current extraction when
#   lesion candidates are sparse; profiles may therefore use "unavailable"
#   until thresholds are calibrated.
# - healthy stage labels (early/moderate/severe) exist for structural
#   completeness with the 5×3 grid; they are not clinical staging of disease.
# ---------------------------------------------------------------------------

DISEASE_EVIDENCE_PROFILES: Dict[str, Dict[str, DiseaseEvidenceProfile]] = {
    # --- Foot Disease ---
    "foot": {
        "early": _profile(
            "foot", "early",
            texture="low",
            density="low",
            color="low",
            clustering="unavailable",
            boundary="unavailable",
        ),
        "moderate": _profile(
            "foot", "moderate",
            texture="medium",
            density="medium",
            color="medium",
            clustering="unavailable",
            boundary="medium",
        ),
        "severe": _profile(
            "foot", "severe",
            texture="high",
            density="high",
            color="high",
            clustering="medium",
            boundary="high",
        ),
    },
    # --- Mouth Disease ---
    "mouth": {
        "early": _profile(
            "mouth", "early",
            texture="low",
            density="low",
            color="low",
            clustering="unavailable",
            boundary="unavailable",
        ),
        "moderate": _profile(
            "mouth", "moderate",
            texture="medium",
            density="medium",
            color="medium",
            clustering="unavailable",
            boundary="medium",
        ),
        "severe": _profile(
            "mouth", "severe",
            texture="high",
            density="high",
            color="high",
            clustering="medium",
            boundary="high",
        ),
    },
    # --- Lumpy Skin Disease ---
    "lumpy": {
        "early": _profile(
            "lumpy", "early",
            texture="medium",
            density="low",
            color="low",
            clustering="unavailable",
            boundary="low",
        ),
        "moderate": _profile(
            "lumpy", "moderate",
            # Example shape matches the project task illustration; still provisional.
            texture="high",
            density="medium",
            color="medium",
            clustering="unavailable",
            boundary="high",
        ),
        "severe": _profile(
            "lumpy", "severe",
            texture="very_high",
            density="high",
            color="high",
            clustering="high",
            boundary="very_high",
        ),
    },
    # --- Mastitis ---
    "mastitis": {
        "early": _profile(
            "mastitis", "early",
            texture="low",
            density="low",
            color="medium",
            clustering="unavailable",
            boundary="unavailable",
        ),
        "moderate": _profile(
            "mastitis", "moderate",
            texture="medium",
            density="medium",
            color="medium",
            clustering="unavailable",
            boundary="medium",
        ),
        "severe": _profile(
            "mastitis", "severe",
            texture="high",
            density="high",
            color="high",
            clustering="medium",
            boundary="high",
        ),
    },
    # --- Healthy (structural stages only; not clinical disease staging) ---
    "healthy": {
        "early": _profile(
            "healthy", "early",
            texture="very_low",
            density="very_low",
            color="low",
            clustering="unavailable",
            boundary="unavailable",
        ),
        "moderate": _profile(
            "healthy", "moderate",
            texture="very_low",
            density="very_low",
            color="low",
            clustering="unavailable",
            boundary="unavailable",
        ),
        "severe": _profile(
            "healthy", "severe",
            texture="low",
            density="very_low",
            color="low",
            clustering="unavailable",
            boundary="unavailable",
        ),
    },
}


def get_disease_profile(disease: str, stage: str) -> DiseaseEvidenceProfile:
    """
    Return the Disease Evidence Profile for a disease and severity stage.

    Parameters
    ----------
    disease:
        One of: foot, mouth, lumpy, mastitis, healthy (case-insensitive).
    stage:
        One of: early, moderate, severe (case-insensitive).

    Returns
    -------
    dict with keys: disease, stage, biomarkers, provisional.

    Raises
    ------
    KeyError
        If disease or stage is not defined in the knowledge base.
    """
    disease_key = disease.strip().lower()
    stage_key = stage.strip().lower()

    if disease_key not in DISEASE_EVIDENCE_PROFILES:
        raise KeyError(
            f"Unknown disease '{disease}'. Expected one of: {', '.join(DISEASES)}"
        )

    stage_profiles = DISEASE_EVIDENCE_PROFILES[disease_key]
    if stage_key not in stage_profiles:
        raise KeyError(
            f"Unknown stage '{stage}' for disease '{disease_key}'. "
            f"Expected one of: {', '.join(STAGES)}"
        )

    # Return a deep copy so callers cannot mutate the shared registry.
    return deepcopy(stage_profiles[stage_key])


def list_disease_profiles(
    disease: Optional[str] = None,
) -> List[DiseaseEvidenceProfile]:
    """
    List all profiles, or all stages for a single disease.

    Useful for future EBDRE modules that need to iterate the knowledge base.
    """
    if disease is None:
        profiles: List[DiseaseEvidenceProfile] = []
        for stage_map in DISEASE_EVIDENCE_PROFILES.values():
            for profile in stage_map.values():
                profiles.append(deepcopy(profile))
        return profiles

    disease_key = disease.strip().lower()
    if disease_key not in DISEASE_EVIDENCE_PROFILES:
        raise KeyError(
            f"Unknown disease '{disease}'. Expected one of: {', '.join(DISEASES)}"
        )
    return [
        deepcopy(profile)
        for profile in DISEASE_EVIDENCE_PROFILES[disease_key].values()
    ]
