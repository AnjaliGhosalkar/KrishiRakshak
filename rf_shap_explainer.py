from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image


FEATURE_GRID = (7, 7, 1024)


def explain_rf_prediction(
    rf: Any,
    label_encoder: Any,
    scaled_features: np.ndarray,
    predicted_class_id: int,
    image_rgb: np.ndarray,
    output_path: Optional[str] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Explain one Random Forest prediction using TreeSHAP on its scaled inputs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    features = np.asarray(scaled_features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.shape[0] != 1:
        raise ValueError("SHAP explanation expects exactly one prediction row")
    if features.shape[1] != rf.n_features_in_:
        raise ValueError(
            f"Expected {rf.n_features_in_} Random Forest features, got {features.shape[1]}"
        )
    if features.shape[1] != int(np.prod(FEATURE_GRID)):
        raise ValueError("Random Forest feature count does not match the MobileNet feature map")

    class_matches = np.flatnonzero(np.asarray(rf.classes_) == predicted_class_id)
    if len(class_matches) != 1:
        raise ValueError(f"Predicted class id {predicted_class_id!r} is not a unique forest class")
    class_index = int(class_matches[0])
    predicted_label = str(label_encoder.inverse_transform([predicted_class_id])[0])
    probabilities = np.asarray(rf.predict_proba(features))[0]
    probability = float(probabilities[class_index])

    explainer = shap.TreeExplainer(
        rf,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    shap_values = explainer.shap_values(features, check_additivity=False)
    if isinstance(shap_values, list):
        class_values = np.asarray(shap_values[class_index])[0]
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3 and values.shape == (1, features.shape[1], len(rf.classes_)):
            class_values = values[0, :, class_index]
        elif values.ndim == 3 and values.shape == (1, len(rf.classes_), features.shape[1]):
            class_values = values[0, class_index, :]
        elif values.ndim == 2 and values.shape == (1, features.shape[1]):
            class_values = values[0]
        else:
            raise ValueError(f"Unexpected multiclass SHAP output shape: {values.shape}")

    expected_values = np.asarray(explainer.expected_value).reshape(-1)
    if expected_values.size == 1:
        base_value = float(expected_values[0])
    elif expected_values.size == len(rf.classes_):
        base_value = float(expected_values[class_index])
    else:
        raise ValueError(f"Unexpected SHAP base-value shape: {expected_values.shape}")

    reconstructed_probability = base_value + float(np.sum(class_values))
    if not np.isclose(reconstructed_probability, probability, atol=1e-4, rtol=1e-4):
        raise RuntimeError(
            "TreeSHAP additivity check failed for the predicted class: "
            f"base + contributions={reconstructed_probability:.6f}, "
            f"forest probability={probability:.6f}"
        )

    grid_values = class_values.reshape(FEATURE_GRID)
    spatial_values = np.sum(grid_values, axis=2)
    top_indices = np.argsort(np.abs(class_values))[-max(1, top_k):][::-1]
    top_features = []
    for feature_index in top_indices:
        row, remainder = divmod(int(feature_index), FEATURE_GRID[1] * FEATURE_GRID[2])
        column, channel = divmod(remainder, FEATURE_GRID[2])
        contribution = float(class_values[feature_index])
        top_features.append(
            {
                "feature_index": int(feature_index),
                "feature": f"cell_{row}_{column}_channel_{channel}",
                "row": row,
                "column": column,
                "channel": channel,
                "scaled_value": float(features[0, feature_index]),
                "shap_value": contribution,
                "direction": "toward prediction" if contribution >= 0 else "away from prediction",
            }
        )

    rgb = np.asarray(image_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Visualization requires an RGB image with shape HxWx3")
    background = Image.fromarray(rgb.astype(np.uint8)).resize((224, 224))
    limit = float(np.max(np.abs(spatial_values))) or 1.0

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.imshow(background)
    overlay = axis.imshow(
        spatial_values,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        alpha=0.55,
        interpolation="bilinear",
        extent=(0, 224, 224, 0),
    )
    axis.set_title(f"Random Forest TreeSHAP: {predicted_label}")
    axis.set_axis_off()
    figure.colorbar(overlay, ax=axis, fraction=0.046, pad=0.04, label="SHAP contribution")

    image_buffer = BytesIO()
    figure.savefig(image_buffer, format="png", dpi=160)
    plt.close(figure)
    image_bytes = image_buffer.getvalue()
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)

    return {
        "method": "TreeSHAP",
        "explains": "the production Random Forest prediction",
        "class": predicted_label,
        "class_probability": probability,
        "base_value": base_value,
        "additivity_sum": reconstructed_probability,
        "feature_count": int(features.shape[1]),
        "top_features": top_features,
        "spatial_map": spatial_values.tolist(),
        "visualization": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
        "saved_path": str(output_path) if output_path else None,
    }