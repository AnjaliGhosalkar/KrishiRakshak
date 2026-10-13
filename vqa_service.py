from __future__ import annotations

import os
import re
import string
from functools import lru_cache
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image

MODEL_ID = "Salesforce/blip-vqa-base"
DESCRIPTION_QUESTION = "What animal is shown?"
DESCRIPTION_SIZE = (320, 320)
_UNSUPPORTED_OUTPUT = re.compile(
    r"\b(got shot|was shot|shot by|gunshot|bullet|diagnos(?:ed|is)|treat(?:ment|ed)?|"
    r"antibiotic|medicine|dosage|prescription|lumpy skin disease|mastitis|foot.and.mouth disease)\b",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "a", "an", "and", "are", "can", "could", "does", "from", "how", "in", "is", "it",
    "me", "of", "on", "or", "show", "shown", "sign", "signs", "the", "these", "this",
    "to", "what", "which", "with", "would",
}
_FEATURE_ALIASES = {
    "skin": {"skin", "lump", "lumps", "nodule", "nodules", "lesion", "lesions", "bump", "bumps"},
    "udder": {"udder", "teat", "teats", "swelling", "swollen", "redness", "lesion", "lump"},
    "hoof": {"hoof", "hooves", "foot", "feet", "leg", "legs", "lameness", "swelling"},
    "mouth": {"mouth", "lip", "lips", "tongue", "ulcer", "ulcers", "lesion", "lesions"},
    "color": {"black", "white", "brown", "red", "pink", "gray", "grey", "yellow", "blue"},
}
_ANIMALS = {"cow", "cattle", "calf", "calves", "buffalo", "horse", "sheep", "goat"}


@lru_cache(maxsize=1)
def _load_vqa_model() -> Tuple[Any, Any, Any, Any]:
    """Load and cache the BLIP processor and model on first use."""
    try:
        import torch
        os.environ["USE_TF"] = "0"
        from transformers import BlipForQuestionAnswering, BlipProcessor
    except ImportError as exc:
        raise RuntimeError(
            "BLIP VQA dependencies are unavailable. Install requirements-vqa.txt."
        ) from exc

    try:
        processor = BlipProcessor.from_pretrained(MODEL_ID)
        model = BlipForQuestionAnswering.from_pretrained(MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            f"Could not load BLIP VQA model {MODEL_ID!r}. "
            "Check network access or the Hugging Face model cache."
        ) from exc

    return processor, model, torch, device


def _run_vqa_model(image: Image.Image, question: str) -> str:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    try:
        processor, model, torch, device = _load_vqa_model()
        inputs = processor(
            images=image.convert("RGB"),
            text=question,
            return_tensors="pt",
        )
        inputs = {
            name: value.to(device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=30)
        answer = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("BLIP could not answer the question for this image.") from exc

    return answer


def _is_relevant_observation(question: str, answer: str) -> bool:
    normalized_answer = " ".join(answer.casefold().split())
    if not normalized_answer or len(normalized_answer) > 180 or _UNSUPPORTED_OUTPUT.search(normalized_answer):
        return False
    question_terms = {
        token for token in re.findall(r"[a-z0-9]+", question.casefold())
        if len(token) > 2 and token not in _STOP_WORDS
    }
    answer_terms = set(re.findall(r"[a-z0-9]+", normalized_answer))
    if question_terms & answer_terms:
        return True
    for topic, aliases in _FEATURE_ALIASES.items():
        if topic in question_terms or (topic == "udder" and "mastitis" in question_terms):
            if answer_terms & aliases:
                return True
    if question_terms & {"animal", "cow", "cattle", "livestock"} and answer_terms & _ANIMALS:
        return True
    return False


def analyze_image(image: Image.Image, question: str) -> Dict[str, Any]:
    """Return only short, question-relevant visual suggestions, never diagnoses."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    normalized_question = question.strip()
    answer = _run_vqa_model(image.convert("RGB"), normalized_question)
    if not _is_relevant_observation(normalized_question, answer):
        return {
            "question": normalized_question,
            "status": "unusable",
            "image_observations": None,
        }
    return {
        "question": normalized_question,
        "status": "ok",
        "image_observations": answer,
    }


def answer_question(image: Image.Image, question: str) -> Dict[str, str]:
    """Legacy VQA response, restricted to a relevant non-diagnostic suggestion."""
    result = analyze_image(image, question)
    if not result["image_observations"]:
        raise RuntimeError("BLIP returned no reliable, relevant observation for this question.")
    return {"question": result["question"], "answer": result["image_observations"]}


def describe_image(image: Image.Image) -> str:
    """Describe visible image colors and contrast, using BLIP only for subject ID."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")

    answer = answer_question(image, DESCRIPTION_QUESTION)["answer"].lower().strip()
    subject = answer.strip(string.punctuation + " ")
    if subject not in {"cow", "cattle", "buffalo"}:
        subject = ""

    resized = image.convert("RGB").resize(DESCRIPTION_SIZE)
    rgb = np.asarray(resized)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    median_hue, median_saturation, median_brightness = np.median(hsv, axis=(0, 1))
    if median_saturation < 24:
        color_description = "neutral gray tones"
    elif median_hue <= 12 or median_hue >= 170:
        if median_brightness >= 105 and median_saturation < 120:
            color_description = "warm pink-tan tones"
        else:
            color_description = "reddish tones"
    elif median_hue <= 25:
        color_description = "warm tan-brown tones"
    elif median_hue <= 38:
        color_description = "yellow-brown tones"
    elif median_hue <= 85:
        color_description = "green tones"
    elif median_hue <= 130:
        color_description = "blue tones"
    else:
        color_description = "purple-pink tones"

    dark_threshold = int(np.percentile(grayscale, 22))
    dark_mask = cv2.threshold(
        grayscale,
        dark_threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )[1]
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    _, _, component_stats, _ = cv2.connectedComponentsWithStats(dark_mask)
    image_area = grayscale.shape[0] * grayscale.shape[1]
    large_dark_regions = [
        area
        for area in component_stats[1:, cv2.CC_STAT_AREA]
        if image_area * 0.002 <= area <= image_area * 0.2
    ]

    if len(large_dark_regions) >= 3:
        contrast_description = "several darker, high-contrast areas"
    elif large_dark_regions:
        contrast_description = "some darker, high-contrast areas"
    else:
        contrast_description = "subtle light-and-dark variation"

    if subject:
        subject_description = f"a {subject}" if subject in {"cow", "buffalo"} else subject
        return (
            f"The image shows {subject_description} with {color_description} "
            f"and {contrast_description}."
        )
    return f"The image shows {color_description} with {contrast_description}."
