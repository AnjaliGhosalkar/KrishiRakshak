from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def _load_classifier_and_grad_model() -> tuple[Any, Any, tuple[str, ...]]:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet import preprocess_input
    from tensorflow.keras.models import Model, load_model

    from train_model import CLASS_NAMES

    classifier_path = Path(__file__).resolve().parent / "cow_skin_mobilenet.keras"
    classifier = load_model(classifier_path, compile=False)
    backbone = classifier.get_layer("mobilenet_1.00_224")
    final_convolution = backbone.get_layer("conv_pw_13_relu")
    activation_model = Model(
        inputs=backbone.inputs,
        outputs=[final_convolution.output, backbone.output],
    )

    image_input = tf.keras.Input(shape=(224, 224, 3), name="gradcam_rgb_image")
    activation_maps, backbone_output = activation_model(preprocess_input(image_input))
    classifier_output = backbone_output
    for layer in classifier.layers[2:]:
        classifier_output = layer(classifier_output)

    grad_model = Model(image_input, [activation_maps, classifier_output])
    if int(classifier.output_shape[-1]) != len(CLASS_NAMES):
        raise ValueError(
            "Saved MobileNet output count does not match train_model.CLASS_NAMES"
        )
    return classifier, grad_model, tuple(CLASS_NAMES)


def explain_saved_mobilenet(
    image_rgb: np.ndarray,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create Grad-CAM for the separate saved five-class MobileNet classifier."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf

    classifier, grad_model, class_names = _load_classifier_and_grad_model()
    rgb = np.asarray(image_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Grad-CAM requires an RGB image with shape HxWx3")

    resized = Image.fromarray(rgb.astype(np.uint8)).resize((224, 224))
    input_tensor = np.asarray(resized, dtype=np.float32)[None, ...]

    with tf.GradientTape() as tape:
        activation_maps, scores = grad_model(input_tensor, training=False)
        class_index = int(tf.argmax(scores[0]))
        target_score = scores[:, class_index]

    gradients = tape.gradient(target_score, activation_maps)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients are unavailable for the saved classifier")

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(channel_weights * activation_maps, axis=-1)[0]
    heatmap = tf.nn.relu(heatmap)
    maximum = float(tf.reduce_max(heatmap))
    if not np.isfinite(maximum) or maximum <= 0:
        raise RuntimeError("Grad-CAM produced no positive activation for the selected class")
    heatmap_array = (heatmap / maximum).numpy()

    rgb_224 = np.asarray(resized, dtype=np.uint8)
    heatmap_image = Image.fromarray(np.uint8(heatmap_array * 255)).resize((224, 224))
    heatmap_resized = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    color_map = matplotlib.colormaps["jet"](heatmap_resized)[..., :3]
    overlay = np.clip(0.58 * (rgb_224 / 255.0) + 0.42 * color_map, 0, 1)

    predicted_class = class_names[class_index]
    confidence = float(scores[0, class_index])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(rgb_224)
    axes[0].set_title("Original")
    axes[1].imshow(heatmap_array, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(f"Saved MobileNet classifier: {predicted_class} ({confidence:.1%})")

    image_buffer = BytesIO()
    figure.savefig(image_buffer, format="png", dpi=160)
    plt.close(figure)
    image_bytes = image_buffer.getvalue()
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)

    return {
        "method": "Grad-CAM",
        "explains": "the separate saved five-class MobileNet classifier, not the production Random Forest",
        "model": "cow_skin_mobilenet.keras",
        "target_layer": "mobilenet_1.00_224/conv_pw_13_relu",
        "class": predicted_class,
        "class_index": class_index,
        "class_probability": confidence,
        "heatmap_shape": list(heatmap_array.shape),
        "visualization": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
        "saved_path": str(output_path) if output_path else None,
    }