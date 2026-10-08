import os
import random
import base64
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
from vqa_service import answer_question, describe_image

app = Flask(__name__)
CORS(app)

# -----------------------------
# CONFIG & INFO
# -----------------------------
IMG_SIZE = (224, 224)
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
        with Image.open(uploaded_file) as source_image:
            image = source_image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Uploaded file is not a valid image"}), 400

    try:
        return jsonify(answer_question(image, question))
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

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
