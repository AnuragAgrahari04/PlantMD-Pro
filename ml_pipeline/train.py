"""
Training Pipeline — EfficientNetV2 with Transfer Learning
Run: python ml_pipeline/train.py --dataset ./data/plantvillage --epochs 30
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def build_model(num_classes: int, img_size: int = 224, learning_rate: float = 1e-4):
    """Build EfficientNetV2-S model with transfer learning."""
    import tensorflow as tf

    # Phase 1: Feature extraction (frozen base)
    base = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
    return model, base


def build_data_pipeline(dataset_dir: str, img_size: int, batch_size: int):
    """Build tf.data pipeline with augmentation."""
    import tensorflow as tf

    AUTOTUNE = tf.data.AUTOTUNE

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    train_ds = (
        train_ds
        .map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(1000)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


def train(
    dataset_dir: str,
    output_dir: str = "models",
    img_size: int = 224,
    batch_size: int = 32,
    epochs_phase1: int = 10,
    epochs_phase2: int = 20,
    learning_rate: float = 1e-4,
):
    import tensorflow as tf

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[PlantMD] Loading dataset from: {dataset_dir}")
    train_ds, val_ds, class_names = build_data_pipeline(dataset_dir, img_size, batch_size)
    num_classes = len(class_names)
    print(f"[PlantMD] Classes: {num_classes} — {class_names[:5]}...")

    # Save labels
    labels_path = Path(output_dir) / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[PlantMD] Labels saved to {labels_path}")

    model, base = build_model(num_classes, img_size, learning_rate)
    print(f"[PlantMD] Model parameters: {model.count_params():,}")

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(Path(output_dir) / "plantmd_efficientnet.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(Path(output_dir) / "logs"),
            histogram_freq=1,
        ),
        tf.keras.callbacks.CSVLogger(str(Path(output_dir) / "training_log.csv")),
    ]

    # ── Phase 1: Feature extraction ─────────────────────────────────────────
    print(f"\n[PlantMD] Phase 1: Feature extraction ({epochs_phase1} epochs)")
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_phase1, callbacks=callbacks)

    # ── Phase 2: Fine-tuning (unfreeze top layers) ───────────────────────────
    print(f"\n[PlantMD] Phase 2: Fine-tuning ({epochs_phase2} epochs)")
    base.trainable = True
    # Freeze early layers, fine-tune last 30%
    fine_tune_from = int(len(base.layers) * 0.7)
    for layer in base.layers[:fine_tune_from]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1 + epochs_phase2,
        initial_epoch=epochs_phase1,
        callbacks=callbacks,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[PlantMD] Final evaluation:")
    results = model.evaluate(val_ds, verbose=1)
    metrics = dict(zip(model.metrics_names, results))
    print(f"[PlantMD] Results: {metrics}")

    # Save metrics
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[PlantMD] ✅ Training complete! Model saved to {output_dir}/plantmd_efficientnet.h5")
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PlantMD EfficientNetV2 model")
    parser.add_argument("--dataset", required=True, help="Path to PlantVillage dataset directory")
    parser.add_argument("--output", default="models", help="Output directory for model artifacts")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-phase1", type=int, default=10)
    parser.add_argument("--epochs-phase2", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset,
        output_dir=args.output,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        learning_rate=args.lr,
    )
