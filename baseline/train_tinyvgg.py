"""Kubeflow-friendly TinyVGG training entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D

try:
    from .data_loader import DataConfig, build_datasets
except ImportError:
    from data_loader import DataConfig, build_datasets

def build_tinyvgg(num_classes: int) -> tf.keras.Model:
    return Sequential(
        [
            Input(shape=(1024, 2, 1)),
            Conv2D(16, (3, 3), padding="same", activation="relu"),
            Conv2D(16, (3, 3), padding="same", activation="relu"),
            MaxPool2D((2, 1)),
            Conv2D(16, (3, 3), padding="same", activation="relu"),
            Conv2D(16, (3, 3), padding="same", activation="relu"),
            MaxPool2D((2, 1)),
            Flatten(),
            Dense(num_classes, activation="softmax"),
        ],
        name="TinyVGG",
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", required=True, help="Path to GOLD_XYZ_OSC.0001_1024.hdf5.")
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "/tmp/outputs/model/tinyvgg.keras"))
    parser.add_argument("--history-path", default=os.getenv("HISTORY_PATH", "/tmp/outputs/history/training_history.csv"))
    parser.add_argument("--class-counts-path", default=os.getenv("CLASS_COUNTS_PATH", "/tmp/outputs/history/class_counts.csv"))
    parser.add_argument("--metrics-path", default=os.getenv("METRICS_PATH", "/tmp/outputs/metrics/mlpipeline-metrics.json"))
    parser.add_argument("--data-ratio", type=float, default=0.01)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=24)
    parser.add_argument("--random-split", action="store_true")
    return parser.parse_args()

def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def keras_model_path(path: str) -> str:
    output_path = Path(path)
    if output_path.suffix in {".keras", ".h5"}:
        ensure_parent(str(output_path))
        return str(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path / "tinyvgg.keras")

def save_kubeflow_metrics(path: str, metrics: dict[str, float]) -> None:
    ensure_parent(path)
    payload = {
        "metrics": [
            {"name": name, "numberValue": float(value), "format": "RAW"}
            for name, value in metrics.items()
        ]
    }
    Path(path).write_text(json.dumps(payload), encoding="utf-8")

def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    config = DataConfig(
        file_path=args.file_path,
        data_ratio=args.data_ratio,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        num_classes=args.num_classes,
        stratified=not args.random_split,
    )
    train_ds, test_ds, class_counts = build_datasets(config)

    model_path = keras_model_path(args.model_path)
    for path in [args.history_path, args.class_counts_path]:
        ensure_parent(path)

    class_counts.to_csv(args.class_counts_path, index=False)
    print(class_counts.to_string(index=False))

    model = build_tinyvgg(args.num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    history = model.fit(train_ds, validation_data=test_ds, epochs=args.epochs)
    pd.DataFrame({"epoch": range(1, len(history.history["loss"]) + 1), **history.history}).to_csv(
        args.history_path,
        index=False,
    )

    test_loss, test_accuracy = model.evaluate(test_ds)
    model.save(model_path)
    save_kubeflow_metrics(
        args.metrics_path,
        {"test_loss": test_loss, "test_accuracy": test_accuracy},
    )

    print(f"model_path={model_path}")
    print(f"history_path={args.history_path}")
    print(f"class_counts_path={args.class_counts_path}")
    print(f"metrics_path={args.metrics_path}")

if __name__ == "__main__":
    main()
