# ==========================================
# FAST PREDICTION (WITH DISEASE DETAILS)
# ==========================================

import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

# -----------------------------
# CONFIG
# -----------------------------
TEST_DIR = "test_images"
IMG_SIZE = (224, 224)

# -----------------------------
# DISEASE INFO
# -----------------------------
DISEASE_INFO = {
    "foot": {
        "description": "Foot disease causes lameness and swelling in cattle.",
        "stages": [
            "Early: limping",
            "Moderate: swelling",
            "Severe: walking difficulty"
        ]
    },
    "healthy": {
        "description": "The cow is healthy.",
        "stages": ["No disease"]
    },
    "lumpy": {
        "description": "Lumpy Skin Disease causes nodules on skin.",
        "stages": [
            "Early: small lumps",
            "Moderate: spread",
            "Severe: skin damage"
        ]
    },
    "mastitis": {
        "description": "Mastitis affects the udder.",
        "stages": [
            "Early: swelling",
            "Moderate: pain",
            "Severe: infection"
        ]
    },
    "mouth": {
        "description": "Mouth disease causes sores.",
        "stages": [
            "Early: sores",
            "Moderate: eating issue",
            "Severe: ulcers"
        ]
    }
}

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
RECOMMENDATIONS = {
    "healthy": {
        "none": "Continue current feeding and hygiene practices."
    },
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
# LOAD CNN (ONLY ONCE)
# -----------------------------
print(" Loading CNN...")
base_model = MobileNet(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

feature_extractor = Model(
    inputs=base_model.input,
    outputs=base_model.output
)

# -----------------------------
# LOAD TRAINED MODELS
# -----------------------------
print(" Loading trained models...")
rf = joblib.load("rf_model.pkl")
le = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

print(" Ready for prediction!")

def show_disease_progression(disease):
    if disease == "healthy":
        return
        
    stages = ["early", "moderate", "severe"]
    stage_titles = ["Early Stage", "Moderate Stage", "Severe Stage"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if hasattr(fig.canvas.manager, 'set_window_title'):
        fig.canvas.manager.set_window_title(f"{disease.capitalize()} Progression")
    
    for i, stage in enumerate(stages):
        ax = axes[i]
        stage_dir = os.path.join("dataset", disease, stage)
        img_loaded = False
        
        if os.path.exists(stage_dir):
            files = [f for f in os.listdir(stage_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                random_file = random.choice(files)
                try:
                    stage_img = Image.open(os.path.join(stage_dir, random_file)).convert("RGB")
                    ax.imshow(stage_img)
                    img_loaded = True
                except Exception:
                    pass
        
        ax.set_title(stage_titles[i])
        ax.axis("off")
        if not img_loaded:
            ax.text(0.5, 0.5, 'No Image Available', ha='center', va='center')
            
    plt.tight_layout()

# -----------------------------
# LOOP FOR MULTIPLE PREDICTIONS
# -----------------------------
while True:

    print("\n Enter image name from test_images folder")
    file = input(" Image name (or 'exit'): ")

    if file.lower() == "exit":
        print(" Exiting...")
        break

    path = os.path.join(TEST_DIR, file)

    if not os.path.exists(path):
        print(" Image not found! Check filename.")
        continue

    try:
        # Load image
        img = Image.open(path).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        arr = np.array(img_resized)

        # Preprocess
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
        confidence = np.max(pred_proba) * 100
        
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

        # Recommendation
        recommendation = RECOMMENDATIONS.get(disease, {}).get(stage, "Consult a veterinarian.")

        # Display result
        print("\n" + "="*40)
        print(" PREDICTION RESULTS")
        print("="*40)
        print(f" Disease:        {disease.capitalize()}")
        print(f" Stage:          {stage.capitalize()}")
        print(f" Confidence:     {confidence:.2f}%")
        print(f" Alert:          {alert}")
        print(f" Recommendation: {recommendation}")
        print("="*40)

        info = DISEASE_INFO.get(disease, {})
        print("\n Description:", info.get("description", ""))

        print("\n Stages:")
        for s in info.get("stages", []):
            print("-", s)

        # Show image
        plt.figure()
        plt.imshow(img)
        plt.title(label.upper())
        plt.axis("off")
        
        # Show disease progression
        show_disease_progression(disease)
        
        plt.show()

    except Exception as e:
        print(" Error:", e)