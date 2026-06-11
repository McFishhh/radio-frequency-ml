"""DL4PL-style Conv1D baseline training entrypoint.

Usage:
    python baseline/train_dl4pl.py
    python baseline/train_dl4pl.py configs/tinyvgg_radioml2016.yaml
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import tensorflow as tf
import yaml
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, MaxPool1D

try:
    from .data_loader import DataConfig, build_datasets, snr_split_table
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import DataConfig, build_datasets, snr_split_table

def build_dl4pl(input_shape: tuple[int, int], n_classes: int) -> tf.keras.Model:
    return Sequential(
        [
            Input(shape=input_shape),
            Conv1D(128, kernel_size=8, padding="valid", activation="relu"),
            MaxPool1D(pool_size=2, strides=2),
            Conv1D(64, kernel_size=16, padding="valid", activation="relu"),
            MaxPool1D(pool_size=2, strides=2),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(n_classes, activation="softmax"),
        ],
        name="DL4PL_Conv1D",
    )

def squeeze_conv2d_channel(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.map(
        lambda x, y: (tf.squeeze(x, axis=-1), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

def result_path(results_dir: str, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else Path(results_dir) / path)

def keras_model_path(path: str) -> str:
    model_path = Path(path)
    if model_path.suffix not in {".keras", ".h5"}:
        model_path = model_path / "dl4pl_conv1d.keras"
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

    results_dir = results_cfg.get("dir", "model_history/dl4pl")
    os.makedirs(results_dir, exist_ok=True)

    log_path = result_path(results_dir, results_cfg.get("log", "training_history.csv"))
    model_path = keras_model_path(result_path(results_dir, results_cfg.get("model", "dl4pl_conv1d.keras")))
    counts_path = result_path(results_dir, results_cfg.get("class_counts", "class_counts.csv"))
    snr_counts_path = result_path(results_dir, results_cfg.get("snr_counts", "snr_counts.csv"))
    metrics_path = result_path(results_dir, results_cfg.get("metrics", "mlpipeline-metrics.json"))
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(training_cfg.get("seed", 42))

    data_config = DataConfig(
        dataset=data_cfg.get("dataset", "radioml2016"),
        file_path=data_cfg.get("path", data_cfg.get("train_hdf5", data_cfg.get("hdf5"))),
        data_ratio=data_cfg.get("data_ratio", 1.0),
        train_ratio=data_cfg.get("train_ratio", 0.8),
        batch_size=training_cfg.get("batch_size", 64),
        shuffle_buffer=training_cfg.get("shuffle_buffer", 512),
        seed=training_cfg.get("seed", 42),
        eps=data_cfg.get("eps", 1e-6),
        num_classes=model_cfg.get("n_classes"),
        stratified=data_cfg.get("stratified", True),
        augment_train=training_cfg.get("augment_train", False),
        awgn_std=training_cfg.get("awgn_std", 0.05),
        rotation_count=training_cfg.get("rotation_count", 4),
    )

    train_ds, val_ds, class_counts, input_shape, n_classes = build_datasets(data_config)
    train_ds = squeeze_conv2d_channel(train_ds)
    val_ds = squeeze_conv2d_channel(val_ds)
    conv1d_input_shape = input_shape[:2]

    Path(counts_path).parent.mkdir(parents=True, exist_ok=True)
    class_counts.to_csv(counts_path, index=False)
    print(class_counts.to_string(index=False))

    if "snr" in class_counts.columns:
        snr_counts = snr_split_table(class_counts)
        Path(snr_counts_path).parent.mkdir(parents=True, exist_ok=True)
        snr_counts.to_csv(snr_counts_path, index=False)
        print(snr_counts.to_string(index=False))

    model = build_dl4pl(conv1d_input_shape, n_classes)
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

    best_model = tf.keras.models.load_model(model_path)
    val_loss, val_accuracy = best_model.evaluate(val_ds)
    write_kubeflow_metrics(metrics_path, {"val_loss": val_loss, "val_accuracy": val_accuracy})

    print(f"Best model : {model_path}")
    print(f"Log        : {log_path}")
    print(f"Counts     : {counts_path}")
    if "snr" in class_counts.columns:
        print(f"SNR counts : {snr_counts_path}")
    print(f"Metrics    : {metrics_path}")

if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/tinyvgg_radioml2016.yaml"
    main(config)
