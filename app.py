import os
import random
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

app = Flask(__name__)
CORS(app)

# -----------------------------
# CONFIG & INFO
# -----------------------------
IMG_SIZE = (224, 224)

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
        
        return jsonify({
            "disease": disease.capitalize(),
            "stage": stage.capitalize(),
            "confidence": f"{confidence:.2f}%",
            "alert": alert,
            "recommendation": recommendation,
            "description": description,
            "progression_images": progression_images
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
