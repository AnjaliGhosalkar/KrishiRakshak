import os
import random
import base64
import json
from uuid import uuid4
from io import BytesIO
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import joblib





from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

from biomarker_extraction import extract_biomarkers
from ebdre.engine import safe_run_ebdre_pipeline
from mobilenet_gradcam import explain_saved_mobilenet
from rf_shap_explainer import explain_rf_prediction
from vqa_service import analyze_image, answer_question, describe_image
from disease_knowledge import DiseaseKnowledgeBase, KnowledgeServiceError

app = Flask(__name__)
CORS(app)
knowledge_base = DiseaseKnowledgeBase()

# -----------------------------
# CONFIG & INFO
# -----------------------------
IMG_SIZE = (224, 224)
ASSISTANT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ASSISTANT_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
ASSISTANT_IMAGE_MAX_BYTES = 12 * 1024 * 1024
ASSISTANT_IMAGE_MAX_PIXELS = 20_000_000
XAI_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xai_outputs")

DISEASE_INFO = {
    "foot": {"description": "Foot disease causes lameness and swelling in cattle."},
    "healthy": {"description": "The cow is healthy."},
    "lumpy": {"description": "Lumpy Skin Disease causes nodules on skin."},
    "mastitis": {"description": "Mastitis affects the udder."},
    "mouth": {"description": "Mouth disease causes sores."}
}

RECOMMENDATIONS = {
    "healthy": {"none": "Continue current feeding and hygiene practices."},
    "foot": {
        "early": "Clean the hooves and apply antiseptic.",
        "moderate": "Trim hooves and apply copper sulfate bath.",
        "severe": "Use antibiotics and seek veterinary hoof care."
    },
    "lumpy": {
        "early": "Isolate animal to prevent spread.",
        "moderate": "Provide soft feed and apply wound care to nodules.",
        "severe": "Administer pain relief and antibiotics under vet guidance."
    },
    "mastitis": {
        "early": "Apply warm compresses and milk out affected quarter.",
        "moderate": "Administer prescribed intramammary antibiotics.",
        "severe": "Isolate the animal and consult a vet for systemic treatment."
    },
    "mouth": {
        "early": "Provide soft feed and monitor eating habits.",
        "moderate": "Apply mouth wash/antiseptic to sores.",
        "severe": "Seek immediate vet assistance for fluid therapy."
    }
}

# -----------------------------
# LOAD MODELS
# -----------------------------
print("Loading models...")
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

rf = joblib.load("rf_model.pkl")
le = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
print("Models loaded successfully.")

def encode_image_base64(img_path):
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        return None


def read_assistant_image(uploaded_file):
    """Decode a bounded, supported image upload for assistant analysis."""
    filename = (uploaded_file.filename or "").strip()
    if not filename:
        raise ValueError("Choose an image file before submitting.")
    if os.path.splitext(filename)[1].lower() not in ASSISTANT_IMAGE_EXTENSIONS:
        raise ValueError("Upload a JPEG, PNG, WEBP, or BMP image.")

    payload = uploaded_file.stream.read(ASSISTANT_IMAGE_MAX_BYTES + 1)
    if not payload:
        raise ValueError("The uploaded image is empty.")
    if len(payload) > ASSISTANT_IMAGE_MAX_BYTES:
        raise ValueError("The image must be 12 MB or smaller.")

    try:
        with Image.open(BytesIO(payload)) as image_check:
            if image_check.format not in ASSISTANT_IMAGE_FORMATS:
                raise ValueError("The uploaded file is not a supported image format.")
            if image_check.width * image_check.height > ASSISTANT_IMAGE_MAX_PIXELS:
                raise ValueError("The image dimensions are too large to analyze.")
            image_check.verify()
        with Image.open(BytesIO(payload)) as source_image:
            source_image.load()
            return source_image.convert("RGB")
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a valid, decodable image.") from exc

def get_progression_images(disease):
    if disease == "healthy" or disease.lower() == "healthy":
        return {}
        
    stages = ["early", "moderate", "severe"]
    progression = {}
    
    for stage in stages:
        stage_dir = os.path.join("dataset", disease.lower(), stage)
        if os.path.exists(stage_dir):
            files = [f for f in os.listdir(stage_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                random_file = random.choice(files)
                b64 = encode_image_base64(os.path.join(stage_dir, random_file))
                if b64:
                    progression[stage] = b64
    return progression

@app.route("/result-images/<path:filename>", methods=["GET"])
def serve_result_image(filename):
    return send_from_directory(XAI_OUTPUT_DIR, filename)

@app.route("/vqa", methods=["POST"])
def answer_vqa():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    uploaded_file = request.files["image"]
    if uploaded_file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    question = request.form.get("question", "")
    if not question.strip():
        return jsonify({"error": "A question is required"}), 400

    try:
        image = read_assistant_image(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        return jsonify(answer_question(image, question))
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

@app.route("/knowledge/status", methods=["GET"])
def knowledge_status():
    return jsonify(knowledge_base.status())

@app.route("/knowledge/reindex", methods=["POST"])
def knowledge_reindex():
    try:
        report = knowledge_base.reindex()
    except KnowledgeServiceError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify(report), 200 if report.get("built") else 422

@app.route("/knowledge/ask", methods=["POST"])
def ask_disease_knowledge():
    try:
        image = None
        prediction_context = None
        if request.mimetype == "multipart/form-data":
            question = request.form.get("question", "")
            history = json.loads(request.form.get("history", "[]"))
            top_k = int(request.form.get("top_k", "4"))
            prediction_value = request.form.get("prediction_context")
            if prediction_value:
                prediction_context = json.loads(prediction_value)
            uploaded_file = request.files.get("image")
            if uploaded_file is not None:
                image = read_assistant_image(uploaded_file)
        else:
            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                return jsonify({"error": "A JSON request body is required"}), 400
            question = body.get("question")
            history = body.get("history", [])
            top_k = body.get("top_k", 4)
            prediction_context = body.get("prediction_context")

        if not isinstance(question, str) or not question.strip():
            return jsonify({"error": "A non-empty question is required"}), 400
        if len(question) > 1000:
            return jsonify({"error": "The question must be 1000 characters or fewer"}), 400
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 8:
            return jsonify({"error": "top_k must be an integer between 1 and 8"}), 400

        image_analysis = analyze_image(image, question) if image is not None else None
        result = knowledge_base.ask(
            question,
            top_k=top_k,
            history=history,
            image_observations=(image_analysis or {}).get("image_observations"),
            prediction_context=prediction_context,
            image_provided=image is not None,
        )
        if image_analysis:
            result["image_analysis_status"] = image_analysis["status"]
            if image_analysis.get("image_observations"):
                result["image_observations"] = image_analysis["image_observations"]
        if prediction_context:
            result["prediction_context"] = prediction_context
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except KnowledgeServiceError as exc:
        return jsonify({"error": str(exc)}), 503
    except RuntimeError as exc:
        return jsonify({"error": f"Image analysis is unavailable: {exc}"}), 503

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files["image"]
    if file.filename == '':
        return jsonify({"error": "Empty file"}), 400
        
    try:
        # Process image
        img = Image.open(file).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        arr = np.array(img_resized)
        arr = preprocess_input(arr.astype("float32"))
        arr = np.expand_dims(arr, axis=0)
        
        # Extract features
        features = feature_extractor.predict(arr, verbose=0)
        features = features.reshape(1, -1)
        features = scaler.transform(features)
        
        # Predict
        pred_proba = rf.predict_proba(features)[0]
        pred = rf.predict(features)
        label = le.inverse_transform(pred)[0]
        confidence = float(np.max(pred_proba) * 100)
        
        if label == "healthy":
            disease = "healthy"
            stage = "none"
        else:
            parts = label.split("_")
            disease = parts[0]
            stage = parts[1] if len(parts) > 1 else "unknown"
            
        # Alerts
        if stage == "early":
            alert = "Monitor the animal regularly"
        elif stage == "moderate":
            alert = "Provide basic treatment and observe closely"
        elif stage == "severe":
            alert = "Immediate veterinary attention required"
        else:
            alert = "Maintain standard care"
            
        recommendation = RECOMMENDATIONS.get(disease, {}).get(stage, "Consult a veterinarian.")
        description = DISEASE_INFO.get(disease, {}).get("description", "")
        
        progression_images = get_progression_images(disease)
        image_rgb = np.asarray(img)
        request_id = uuid4().hex

        # EBDRE reasoning layer (after CNN+RF). Uses original RGB image.
        # Failures here must not break the existing prediction response.
        ebdre_result = safe_run_ebdre_pipeline(
            image_rgb=image_rgb,
            disease=disease,
            stage=stage,
            cnn_confidence=confidence,
        )

        result_images = {}
        try:
            biomarker_result = extract_biomarkers(
                image_rgb,
                debug_output_dir=os.path.join(XAI_OUTPUT_DIR, request_id),
                save_debug=True,
            )
            if biomarker_result.get("debug_image_path"):
                result_images["biomarker_image_url"] = url_for(
                    "serve_result_image",
                    filename=f"{request_id}/array_input_debug.jpg",
                    _external=True,
                )
        except Exception as e:
            result_images["biomarker_image_error"] = str(e)

        try:
            result_images["visual_description"] = describe_image(img)
        except Exception as e:
            result_images["visual_description_error"] = str(e)

        xai_result = {}
        try:
            shap_result = explain_rf_prediction(
                rf=rf,
                label_encoder=le,
                scaled_features=features,
                predicted_class_id=int(pred[0]),
                image_rgb=image_rgb,
                output_path=os.path.join(XAI_OUTPUT_DIR, f"{request_id}_shap.png"),
            )
            shap_result.pop("visualization", None)
            shap_result.pop("saved_path", None)
            shap_result["image_url"] = url_for(
                "serve_result_image",
                filename=f"{request_id}_shap.png",
                _external=True,
            )
            xai_result["random_forest_shap"] = shap_result
        except Exception as e:
            xai_result["random_forest_shap_error"] = str(e)

        try:
            gradcam_result = explain_saved_mobilenet(
                image_rgb=image_rgb,
                output_path=os.path.join(XAI_OUTPUT_DIR, f"{request_id}_gradcam.png"),
            )
            gradcam_result.pop("visualization", None)
            gradcam_result.pop("saved_path", None)
            gradcam_result["image_url"] = url_for(
                "serve_result_image",
                filename=f"{request_id}_gradcam.png",
                _external=True,
            )
            xai_result["saved_mobilenet_gradcam"] = gradcam_result
        except Exception as e:
            xai_result["saved_mobilenet_gradcam_error"] = str(e)
        
        return jsonify({
            "disease": disease.capitalize(),
            "stage": stage.capitalize(),
            # Keep legacy field for current React UI
            "confidence": f"{confidence:.2f}%",
            "alert": alert,
            "recommendation": recommendation,
            "description": description,
            "progression_images": progression_images,
            **result_images,
            "xai": xai_result,
            # EBDRE fields
            "cnn_confidence": ebdre_result.get("cnn_confidence"),
            "clinical_support_score": ebdre_result.get("clinical_support_score"),
            "evidence_strength": ebdre_result.get("evidence_strength"),
            "available_evidence": ebdre_result.get("available_evidence"),
            "confidence_level": ebdre_result.get("confidence_level"),
            "ranked_evidence": ebdre_result.get("ranked_evidence", []),
            "clinical_summary": ebdre_result.get("clinical_summary"),
            "biomarkers": ebdre_result.get("biomarkers"),
            "evidence_matching": ebdre_result.get("evidence_matching"),
            **(
                {"ebdre_error": ebdre_result["ebdre_error"]}
                if ebdre_result.get("ebdre_error")
                else {}
            ),
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
