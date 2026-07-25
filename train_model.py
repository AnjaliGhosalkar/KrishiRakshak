import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input


DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["foot", "mouth", "lumpy", "mastitis", "healthy"]


def get_datasets():
    """Create training and validation datasets from the folder structure."""
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
    )

    # Use AUTOTUNE for better performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def build_mobilenet_model(num_classes: int):
    """Build a MobileNet-based classification model."""
    base_model = MobileNet(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    # Freeze base model for transfer learning (you can unfreeze later for fine-tuning)
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    train_ds, val_ds = get_datasets()

    model = build_mobilenet_model(num_classes=len(CLASS_NAMES))
    model.summary()

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )
    model.save("cow_skin_disease_model.h5")
    print("Model saved successfully!")

    # Print final training and validation accuracy
    train_acc = history.history.get("accuracy", [None])[-1]
    val_acc = history.history.get("val_accuracy", [None])[-1]

    print(f"\nFinal training accuracy: {train_acc:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")

    # Save the model in the recommended Keras format (.keras) to avoid
    # legacy H5 deserialization issues like Unknown layer: 'TrueDivide'.
    model.save("cow_skin_mobilenet.keras")
    print("Model saved as 'cow_skin_mobilenet.keras'")


if __name__ == "__main__":
    main()