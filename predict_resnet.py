import sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from train_resnet50_model import build_resnet_model, CLASS_NAMES, IMG_SIZE

MODEL_WEIGHTS_PATH = "cow_skin_resnet_model.h5"


def load_trained_model():
    model = build_resnet_model(num_classes=len(CLASS_NAMES))
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model


def preprocess_image(img_path: str):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = tf.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def predict_image(model, img_path: str):
    x = preprocess_image(img_path)
    preds = model.predict(x)
    probs = preds[0]
    top_index = int(tf.argmax(probs).numpy())
    predicted_class = CLASS_NAMES[top_index]
    confidence = float(probs[top_index])

    print(f"Image: {img_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("\nAll class probabilities:")
    for cls_name, p in zip(CLASS_NAMES, probs):
        print(f"  {cls_name}: {float(p):.4f}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_resnet.py path_to_image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    print("Loading ResNet50 model...")
    model = load_trained_model()
    predict_image(model, img_path)


if __name__ == "__main__":
    main()