"""TinyVGG baseline training entrypoint.

Usage:
    python baseline/train_tinyvgg.py
    python baseline/train_tinyvgg.py configs/tinyvgg.yaml
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import tensorflow as tf
import yaml
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D

try:
    from .data_loader import DataConfig, build_datasets
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import DataConfig, build_datasets

def build_tinyvgg(input_shape: tuple[int, int, int], n_classes: int) -> tf.keras.Model:
    return Sequential(
        [
            Input(shape=input_shape),
            Conv2D(16, (3, 3), padding="same", activation="relu"),
            Conv2D(16, (3, 3), padding="same", activation="relu"),
            MaxPool2D((2, 1)),
            Conv2D(16, (3, 3), padding="same", activation="relu"),
            Conv2D(16, (3, 3), padding="same", activation="relu"),
            MaxPool2D((2, 1)),
            Flatten(),
            Dense(n_classes, activation="softmax"),
        ],
        name="TinyVGG",
    )

def result_path(results_dir: str, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else Path(results_dir) / path)

def keras_model_path(path: str) -> str:
    model_path = Path(path)
    if model_path.suffix not in {".keras", ".h5"}:
        model_path = model_path / "tinyvgg.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return str(model_path)

def write_kubeflow_metrics(path: str, metrics: dict[str, float]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": [
            {"name": name, "numberValue": float(value), "format": "RAW"}
            for name, value in metrics.items()
        ]
    }
    Path(path).write_text(json.dumps(payload), encoding="utf-8")

def main(config_path: str) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    results_cfg = cfg["results"]

    results_dir = results_cfg["dir"]
    os.makedirs(results_dir, exist_ok=True)

    log_path = result_path(results_dir, results_cfg.get("log", "training_history.csv"))
    model_path = keras_model_path(result_path(results_dir, results_cfg.get("model", "tinyvgg.keras")))
    counts_path = result_path(results_dir, results_cfg.get("class_counts", "class_counts.csv"))
    metrics_path = result_path(results_dir, results_cfg.get("metrics", "mlpipeline-metrics.json"))
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(training_cfg.get("seed", 42))

    data_config = DataConfig(
        dataset=data_cfg.get("dataset", "radioml2018"),
        file_path=data_cfg.get("path", data_cfg.get("train_hdf5", data_cfg.get("hdf5"))),
        data_ratio=data_cfg.get("data_ratio", 0.01),
        train_ratio=data_cfg.get("train_ratio", 0.8),
        batch_size=training_cfg.get("batch_size", 16),
        shuffle_buffer=training_cfg.get("shuffle_buffer", 512),
        seed=training_cfg.get("seed", 42),
        eps=data_cfg.get("eps", 1e-6),
        num_classes=model_cfg.get("n_classes", 24),
        stratified=data_cfg.get("stratified", True),
    )

    train_ds, val_ds, class_counts, input_shape, n_classes = build_datasets(data_config)
    Path(counts_path).parent.mkdir(parents=True, exist_ok=True)
    class_counts.to_csv(counts_path, index=False)
    print(class_counts.to_string(index=False))

    model = build_tinyvgg(input_shape, n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training_cfg.get("lr", 1e-3)),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    callbacks = [
        tf.keras.callbacks.CSVLogger(log_path),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training_cfg.get("epochs", 10),
        callbacks=callbacks,
    )

    val_loss, val_accuracy = model.evaluate(val_ds)
    write_kubeflow_metrics(
        metrics_path,
        {"val_loss": val_loss, "val_accuracy": val_accuracy},
    )

    print(f"Best model : {model_path}")
    print(f"Log        : {log_path}")
    print(f"Counts     : {counts_path}")
    print(f"Metrics    : {metrics_path}")

if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/tinyvgg.yaml"
    main(config)
