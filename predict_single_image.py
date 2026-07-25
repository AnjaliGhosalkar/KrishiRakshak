import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "cow_disease_resnet_model.h5")
model = load_model(model_path)

# same class order as in training
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
IMG_HEIGHT = 224
IMG_WIDTH = 224

tmp_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
tmp_gen = tmp_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,
    class_mode="categorical"
)
class_indices = tmp_gen.class_indices  # e.g. {"diseased": 0, "healthy": 1}
idx_to_class = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_label = idx_to_class[class_idx]
    confidence = float(np.max(preds[0]))

    print(f"Image: {img_path}")
    print(f"Predicted class: {class_label}")
    print(f"Confidence: {confidence:.4f}")

# change this to any image path you want to test
test_image_path = os.path.join(BASE_DIR, "some_test_image.jpg")
predict_image(test_image_path)