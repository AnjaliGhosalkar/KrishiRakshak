# ==========================================
# CNN + Random Forest (Combined Disease & Stage)
# ==========================================

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib

from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)

# -----------------------------
# LOAD DATASET (Combined Labels)
# -----------------------------
def load_images():
    images = []
    labels = []

    print("\n📂 Loading dataset...")

    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset directory '{DATASET_DIR}' not found!")
        return np.array(images), np.array(labels)

    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, file)
                
                # Automatically create combined labels based on folder structure
                # e.g. dataset/mastitis/early/img.jpg -> 'mastitis_early'
                # e.g. dataset/healthy/img.jpg -> 'healthy'
                rel_path = os.path.relpath(root, DATASET_DIR)
                parts = rel_path.split(os.sep)
                
                if len(parts) == 1:
                    label = parts[0]
                elif len(parts) >= 2:
                    label = f"{parts[0]}_{parts[1]}"
                else:
                    continue

                try:
                    img = Image.open(path).convert("RGB")
                    img = img.resize(IMG_SIZE)
                    images.append(np.array(img))
                    labels.append(label)
                except Exception as e:
                    print(f"Error reading {path}: {e}")

    print("Total images:", len(images))
    
    # Print the distribution of created classes
    if len(labels) > 0:
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\n📊 Label Distribution:")
        for lbl, cnt in zip(unique_labels, counts):
            print(f"  - {lbl}: {cnt}")

    return np.array(images), np.array(labels)


# -----------------------------
# CNN FEATURE EXTRACTOR
# -----------------------------
def get_feature_extractor():
    base_model = MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    return Model(inputs=base_model.input, outputs=base_model.output)


# -----------------------------
# MAIN TRAINING
# -----------------------------
def main():

    print("🧠 Loading CNN...")
    feature_extractor = get_feature_extractor()

    X, y = load_images()

    if len(X) == 0:
        print("❌ No data found!")
        return

    print("\n⚙️ Preprocessing...")
    X = preprocess_input(X.astype("float32"))

    print("🔍 Extracting features...")
    features = feature_extractor.predict(X, verbose=1)

    X_features = features.reshape(features.shape[0], -1)

    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)

    print("🏷️ Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes encoded: {le.classes_}")

    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, test_size=0.2, random_state=42
    )

    print("🌳 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    print("📈 Evaluating...")
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)

    print("\n🎯 Accuracy:", round(acc * 100, 2), "%")

    print("\n💾 Saving models...")
    joblib.dump(rf, "rf_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("✅ Training complete! Models saved.")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()
