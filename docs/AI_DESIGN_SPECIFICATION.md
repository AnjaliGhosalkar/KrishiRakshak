# KrishiRakshak Repository Analysis Report

No files were modified. This report is based on a full inspection of the repository at `d:\cow_skin_detection`.

---

## A. Current Architecture

### High-level layout

```
cow_skin_detection/
├── app.py                    ← Production Flask API (entry point for web app)
├── predict.py                ← CLI version of the same prediction logic
├── cnn_random_hybrid.py      ← Training script that produces models used by app.py
├── cnn_svm_hybrid.py         ← Alternate CNN+SVM training (not used by API)
├── train_*.py                ← Additional end-to-end CNN training scripts
├── predict_*.py              ← Standalone CLI predictors (not used by API)
├── rf_model.pkl              ← Random Forest classifier
├── label_encoder.pkl         ← 13-class disease+stage labels
├── scaler.pkl                ← Feature scaler
├── dataset/                  ← Training images (disease/stage folder structure)
├── test_images/              ← CLI test images
├── frontend/                 ← React + Vite SPA
└── docs/
    └── AI_DESIGN_SPECIFICATION.md  ← Empty (1 line)
```

### Frontend structure

| Aspect | Current state |
|--------|---------------|
| Framework | React 19 + Vite 5 |
| Language | **JavaScript (`.jsx`)**, not TypeScript |
| Entry | `frontend/src/main.jsx` → `App.jsx` |
| Styling | `App.css`, `index.css` |
| HTTP client | `axios` |
| Components | **Single monolithic `App.jsx`** — upload, predict, results all in one file |
| API URL | Hardcoded `http://localhost:5000/predict` |

There is no routing, no component library, and no separate services layer.

### Backend structure

| Aspect | Current state |
|--------|---------------|
| Framework | Flask + Flask-CORS |
| Entry point | `app.py` (`app.run(debug=True, host="0.0.0.0", port=5000)`) |
| Routes | **Single route:** `POST /predict` |
| Active model | **MobileNet (feature extractor) + Random Forest (classifier)** |
| Artifacts | `rf_model.pkl`, `label_encoder.pkl`, `scaler.pkl` |
| Dependencies | **No `requirements.txt`** — inferred from imports |

### Disease classes (13 combined labels)

The label encoder defines disease **and** stage as a single combined class:

```
foot_early, foot_moderate, foot_severe,
healthy,
lumpy_early, lumpy_moderate, lumpy_severe,
mastitis_early, mastitis_moderate, mastitis_severe,
mouth_early, mouth_moderate, mouth_severe
```

There is **no separate stage model** — stage is parsed from the combined label string (`disease_stage`).

### Dataset structure

```
dataset/
├── healthy/          ← flat (no stage subfolders)
├── foot/{early,moderate,severe}/
├── lumpy/{early,moderate,severe}/
├── mastitis/{early,moderate,severe}/
└── mouth/{early,moderate,severe}/
```

### Alternate / legacy pipelines (not wired to the web app)

Several scripts exist for experimentation but are **not** used by `app.py` or the React frontend:

- End-to-end CNN models: `train_model.py`, `train_resnet50_model.py`, `train_efficientnet_model.py`
- Saved weights: `cow_skin_*.h5`, `cow_skin_*.keras`, `cow_disease_resnet_model.h5`
- CNN+SVM hybrid: `cnn_svm_hybrid.py` → `main.py`
- Standalone predictors: `predict_resnet.py`, `predict_efficientnet.py`, `predict_single_image.py`

---

## B. Existing Prediction Flow

```
User uploads image (React)
        ↓
POST /predict  (multipart form, field: "image")
        ↓
PIL: open → RGB → resize (224×224)
        ↓
MobileNet preprocess_input → expand_dims
        ↓
MobileNet feature_extractor.predict()
        ↓
reshape → scaler.transform()
        ↓
RandomForest.predict() + predict_proba()
        ↓
Label decode: "lumpy_moderate" → disease="lumpy", stage="moderate"
        ↓
Lookup: DISEASE_INFO, RECOMMENDATIONS, stage-based alert
        ↓
Optional: progression_images (base64 samples from dataset/)
        ↓
JSON response → React results section
```

### Existing response JSON

```json
{
  "disease": "Lumpy",
  "stage": "Moderate",
  "confidence": "94.23%",
  "alert": "Provide basic treatment and observe closely",
  "recommendation": "Provide soft feed and apply wound care to nodules.",
  "description": "Lumpy Skin Disease causes nodules on skin.",
  "progression_images": {
    "early": "data:image/jpeg;base64,...",
    "moderate": "data:image/jpeg;base64,...",
    "severe": "data:image/jpeg;base64,..."
  }
}
```

**Notes:**
- `confidence` is a **string** with `%` suffix, not a numeric field.
- `disease` and `stage` are capitalized strings.
- For `healthy`, stage is `"None"` (capitalized from `"none"`).
- No explainability, biomarker, or evidence fields exist today.

---

## C. Files Involved in Prediction

### Production path (web app)

| Role | File |
|------|------|
| Flask API + orchestration | `app.py` |
| Model artifacts | `rf_model.pkl`, `label_encoder.pkl`, `scaler.pkl` |
| MobileNet weights | Downloaded at runtime via `weights='imagenet'` |
| Disease metadata | `DISEASE_INFO`, `RECOMMENDATIONS` in `app.py` |
| Progression images | `dataset/{disease}/{stage}/` read by `get_progression_images()` |
| Frontend upload + display | `frontend/src/App.jsx` |
| Frontend styles | `frontend/src/App.css` |

### Training path (produces production models)

| Role | File |
|------|------|
| CNN+RF training | `cnn_random_hybrid.py` |
| Dataset loading | `load_images()` in `cnn_random_hybrid.py` (walks `dataset/`) |
| Feature extraction | MobileNet in `cnn_random_hybrid.py` |
| Classifier training | `RandomForestClassifier` → saves `.pkl` files |

### CLI mirror (same logic as API, not used by frontend)

| Role | File |
|------|------|
| Interactive CLI prediction | `predict.py` |

### Treatment / recommendation logic

All in `app.py` (lines 23–53 and 132–143):

- **`RECOMMENDATIONS`**: dict keyed by `disease` → `stage` → text
- **`DISEASE_INFO`**: short description per disease
- **Alerts**: hardcoded strings based on stage (`early` / `moderate` / `severe` / default)
- **Progression images**: random sample from `dataset/` per stage, returned as base64

There is no external knowledge base, vet API, or database.

---

## D. Where EBDRE Should Be Integrated

### Primary integration point: `app.py` → `/predict` handler

EBDRE should run **after** the CNN+RF prediction succeeds and **before** the JSON response is built:

```python
# Conceptual insertion point in app.py (after line ~130)
label = le.inverse_transform(pred)[0]
disease, stage = parse_label(label)
confidence = ...

# --- EBDRE START ---
# Input: original PIL image (img or img_resized), disease, stage
# Output: biomarkers, css, evidence_ranking, clinical_summary
ebdre_result = ebdre_engine.analyze(img, disease, stage)
# --- EBDRE END ---

return jsonify({ ...existing fields..., **ebdre_result })
```

**Why here:**
- The original image is already loaded as a PIL `Image` object.
- Disease and stage are already parsed.
- CNN confidence is already computed separately — EBDRE can add `clinical_support_score` without conflating the two.
- A single API call keeps the frontend simple.

### Recommended modular backend layout

```
ebdre/
├── __init__.py
├── engine.py                 ← Orchestrator: extract → match → score → summarize
├── biomarkers/
│   ├── __init__.py
│   ├── extractor.py          ← Runs all biomarker extractors on image
│   ├── lesion_density.py
│   ├── texture_roughness.py
│   ├── color_variation.py
│   ├── lesion_clustering.py
│   └── boundary_irregularity.py
├── profiles/
│   ├── __init__.py
│   └── disease_profiles.py   ← Expected biomarker ranges per disease/stage
├── matching/
│   ├── __init__.py
│   └── evidence_matcher.py   ← Observed vs expected comparison
├── scoring/
│   ├── __init__.py
│   └── css_calculator.py     ← Clinical Support Score
└── summary/
    ├── __init__.py
    └── evidence_summary.py   ← Ranked evidence + natural-language summary
```

### Secondary integration points (optional, later)

| Location | Purpose |
|----------|---------|
| `predict.py` | Keep CLI and API behavior consistent |
| `frontend/src/App.jsx` | Display CSS, biomarker scores, evidence summary |
| `docs/AI_DESIGN_SPECIFICATION.md` | Document EBDRE design (currently empty) |

### What EBDRE should **not** touch

- MobileNet feature extraction
- Random Forest prediction
- Model loading (`rf_model.pkl`, etc.)
- Existing response fields (unless extending additively)

---

## E. Potential Files to Create

| File | Purpose |
|------|---------|
| `ebdre/engine.py` | Main EBDRE orchestrator |
| `ebdre/biomarkers/extractor.py` | Biomarker extraction pipeline |
| `ebdre/biomarkers/*.py` | Individual biomarker implementations |
| `ebdre/profiles/disease_profiles.py` | Expected profiles per disease/stage |
| `ebdre/matching/evidence_matcher.py` | Observed vs expected matching |
| `ebdre/scoring/css_calculator.py` | Clinical Support Score |
| `ebdre/summary/evidence_summary.py` | Evidence ranking + summary text |
| `requirements.txt` | Pin Flask, TensorFlow, scikit-learn, Pillow, **opencv-python** (if used) |
| `frontend/src/components/EvidencePanel.jsx` | (Optional) Separate UI for EBDRE output |
| `docs/EBDRE_SPECIFICATION.md` | Research module documentation |

---

## F. Potential Files to Modify

| File | Modification |
|------|--------------|
| `app.py` | Import and call EBDRE after prediction; extend JSON response with new fields |
| `frontend/src/App.jsx` | Render CSS, biomarker evidence, clinical summary (below existing results) |
| `frontend/src/App.css` | Styles for new evidence section |
| `predict.py` | (Optional) Mirror EBDRE in CLI for debugging |
| `docs/AI_DESIGN_SPECIFICATION.md` | Fill in project design doc |

**Suggested additive JSON fields** (backward compatible):

```json
{
  "clinical_support_score": "88.00%",
  "biomarkers": {
    "lesion_density": { "observed": 0.72, "expected_range": [0.5, 0.9], "match_score": 0.85 },
    "texture_roughness": { ... }
  },
  "evidence_ranking": [
    { "biomarker": "lesion_density", "support_level": "high", "description": "High lesion density observed." }
  ],
  "clinical_evidence_summary": "High lesion density and rough texture observed. These findings support the predicted disease."
}
```

Existing frontend code will ignore unknown fields until updated.

---

## G. Risks of Breaking the Current System

| Risk | Severity | Detail |
|------|----------|--------|
| **Model load failure at startup** | High | `app.py` loads `.pkl` files and MobileNet at import time. Any import error in EBDRE modules could prevent the server from starting. |
| **Monolithic `app.py`** | Medium | All logic lives in one file. EBDRE integration increases complexity; refactor carefully without changing prediction order. |
| **Duplicate logic** | Medium | `predict.py` mirrors `app.py`. Updating only the API leaves CLI out of sync. |
| **No `requirements.txt`** | Medium | Adding OpenCV or other deps without pinning versions may break existing TensorFlow setup. |
| **Response payload size** | Medium | `progression_images` already embed large base64 strings. Adding biomarker debug images could slow responses. |
| **Healthy class handling** | Medium | `healthy` has no stage subfolders and stage=`none`. EBDRE profiles must define expected biomarkers for healthy vs diseased cases. |
| **Confidence type mismatch** | Low | CNN confidence is a formatted string (`"94.23%"`). EBDRE CSS should follow the same pattern or use numeric fields consistently. |
| **Frontend hardcoded URL** | Low | `localhost:5000` won't work in deployment; unrelated to EBDRE but affects testing. |
| **Vision vs reality: TypeScript** | Low | Project vision says React + TypeScript; codebase is JSX. EBDRE UI can stay in JSX unless you migrate. |
| **Multiple unused model files** | Low | Risk of accidentally wiring EBDRE to ResNet/EfficientNet scripts instead of the active MobileNet+RF pipeline. |
| **No existing tests** | Medium | No test suite found. EBDRE changes have no automated regression guard for `/predict`. |
| **Import-time TensorFlow load** | Medium | MobileNet loads on every server start (~seconds). EBDRE should not add another heavy model at import unless necessary. |

---

## Summary Table: Research Vision vs Current Implementation

| Capability | Status |
|------------|--------|
| Image upload | ✅ React drag-and-drop + file picker |
| Image preprocessing | ✅ PIL resize + MobileNet `preprocess_input` |
| Disease prediction | ✅ MobileNet + Random Forest (13 classes) |
| Stage prediction | ✅ Embedded in combined label (not separate model) |
| CNN confidence | ✅ `predict_proba` max × 100 |
| Treatment/recommendations | ✅ Static dict lookup by disease+stage |
| Progression reference images | ✅ Random samples from `dataset/` |
| Explainability (EBDRE) | ❌ Not implemented |
| Visual biomarkers | ❌ Not implemented |
| Clinical Support Score | ❌ Not implemented |
| Clinical Evidence Summary | ❌ Not implemented |
| SHAP/LIME/Grad-CAM | ❌ Not present |

---

## Recommended EBDRE Integration Strategy (for when you give the go-ahead)

1. Create the `ebdre/` package as a **pure Python module** with no Flask dependency.
2. In `app.py`, pass the **original PIL image** (before or after resize — decide once and document) plus parsed `disease` and `stage` to `ebdre.engine.analyze()`.
3. Extend the JSON response **additively** — do not rename or remove existing fields.
4. Update `App.jsx` to show CSS and evidence summary in a new section, clearly labeled separately from CNN confidence.
5. Add `requirements.txt` before introducing OpenCV or other new dependencies.

---

No code was changed. Ready for your next instruction when you want to proceed with EBDRE implementation or a narrower first step (e.g., biomarker extraction prototype only).