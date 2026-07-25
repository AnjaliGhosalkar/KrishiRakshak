# ==========================================
# FAST Hybrid CNN + Linear SVM + Explanation
# ==========================================

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("🚀 Program started...")

import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import joblib


# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = "dataset"
TEST_DIR = "test_images"
IMG_SIZE = (224, 224)

CLASS_NAMES = ["foot", "healthy", "lumpy", "mastitis", "mouth"]


# -----------------------------
# DISEASE INFO
# -----------------------------
DISEASE_INFO = {
    "foot": {
        "description": "Foot disease causes lameness and swelling in cattle.",
        "stages": ["Early: limping", "Moderate: swelling", "Severe: walking difficulty"]
    },
    "healthy": {
        "description": "The cow is healthy.",
        "stages": ["No disease"]
    },
    "lumpy": {
        "description": "Lumpy Skin Disease causes nodules on skin.",
        "stages": ["Early: small lumps", "Moderate: spread", "Severe: skin damage"]
    },
    "mastitis": {
        "description": "Mastitis affects the udder.",
        "stages": ["Early: swelling", "Moderate: pain", "Severe: infection"]
    },
    "mouth": {
        "description": "Mouth disease causes sores.",
        "stages": ["Early: sores", "Moderate: eating issue", "Severe: ulcers"]
    }
}


# -----------------------------
# LOAD IMAGES (RECURSIVE)
# -----------------------------
def load_images():
    images = []
    labels = []

    print("\n📂 Loading dataset...")

    for label in CLASS_NAMES:
        folder = os.path.join(DATASET_DIR, label)

        if not os.path.exists(folder):
            print("Missing:", folder)
            continue

        count = 0

        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    path = os.path.join(root, file)

                    try:
                        img = Image.open(path).convert("RGB")
                        img = img.resize(IMG_SIZE)
                        images.append(np.array(img))
                        labels.append(label)
                        count += 1

                        # 🔥 LIMIT (for speed, remove if not needed)
                        if count >= 300:
                            break

                    except:
                        pass

        print(f"{label}: {count} images")

    print("Total:", len(images))
    return np.array(images), np.array(labels)


# -----------------------------
# FEATURE EXTRACTOR
# -----------------------------
def get_feature_extractor():
    base_model = MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    return Model(inputs=base_model.input, outputs=base_model.output)


# -----------------------------
# PREDICT SINGLE IMAGE
# -----------------------------
def predict_single_image(feature_extractor, svm, le, scaler):

    print("\nEnter image name from test_images folder")
    file = input("👉 Image name: ")

    path = os.path.join(TEST_DIR, file)

    if not os.path.exists(path):
        print("❌ Not found!")
        return

    img = Image.open(path).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized)

    arr = preprocess_input(arr.astype("float32"))
    arr = np.expand_dims(arr, axis=0)

    features = feature_extractor.predict(arr, verbose=0)
    features = features.reshape(1, -1)
    features = scaler.transform(features)

    pred = svm.predict(features)
    label = le.inverse_transform(pred)[0]

    print("\n✅ Prediction:", label)

    info = DISEASE_INFO.get(label, {})
    print("\n📘 Description:", info.get("description", ""))

    print("\n📊 Stages:")
    for s in info.get("stages", []):
        print("-", s)

    plt.imshow(img)
    plt.title(label.upper())
    plt.axis("off")
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
def main():

    model = get_feature_extractor()

    X, y = load_images()

    if len(X) == 0:
        print("No data!")
        return

    print("\nProcessing...")
    X = preprocess_input(X.astype("float32"))

    features = model.predict(X, verbose=1)
    X_features = features.reshape(features.shape[0], -1)

    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_enc, test_size=0.2, random_state=42
    )

    print("\n⚡ Training FAST SVM...")
    svm = LinearSVC()
    svm.fit(X_train, y_train)

    acc = accuracy_score(y_test, svm.predict(X_test))
    print("\n🎯 Accuracy:", round(acc * 100, 2), "%")

    joblib.dump(svm, "svm.pkl")
    joblib.dump(le, "labels.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Saved models!")

    predict_single_image(model, svm, le, scaler)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()