"""Plot classification accuracy vs SNR for a trained TinyVGG model.

Usage:
    python baseline/plot_snr_accuracy.py
    python baseline/plot_snr_accuracy.py configs/tinyvgg.yaml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

try:
    from .data_loader import DataConfig, hdf5_random_split, hdf5_stratified_split
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import DataConfig, hdf5_random_split, hdf5_stratified_split

def result_path(results_dir: str, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else Path(results_dir) / path)

def keras_model_path(path: str) -> str:
    model_path = Path(path)
    if model_path.suffix not in {".keras", ".h5"}:
        model_path = model_path / "tinyvgg.keras"
    return str(model_path)

def prep_x(x: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return np.expand_dims((x - mean) / (std + eps), -1)

def build_val_indices(config: DataConfig) -> np.ndarray:
    if config.stratified:
        _, val_indices, _ = hdf5_stratified_split(config)
    else:
        _, val_indices = hdf5_random_split(config)
    return val_indices

def predict_by_snr(
    model: tf.keras.Model,
    file_path: str,
    val_indices: np.ndarray,
    batch_size: int,
    eps: float,
) -> pd.DataFrame:
    rows = []

    with h5py.File(file_path, "r") as hf:
        x_ds, y_ds, z_ds = hf["X"], hf["Y"], hf["Z"]

        for start in range(0, len(val_indices), batch_size):
            batch_indices = val_indices[start:start + batch_size]
            x_batch = np.stack([prep_x(x_ds[index], eps) for index in batch_indices])
            y_true = np.array([np.argmax(y_ds[index]) for index in batch_indices])
            snr = np.array([z_ds[index].squeeze() for index in batch_indices])

            y_pred = np.argmax(model.predict(x_batch, verbose=0), axis=1)

            for index, true_label, pred_label, snr_value in zip(batch_indices, y_true, y_pred, snr):
                rows.append({
                    "index": int(index),
                    "snr": int(snr_value),
                    "class_id": int(true_label),
                    "pred_class_id": int(pred_label),
                    "correct": int(true_label == pred_label),
                })

    return pd.DataFrame(rows)

def plot_accuracy(results_df: pd.DataFrame, output_path: str) -> None:
    acc_df = (
        results_df
        .groupby(["class_id", "snr"], as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )

    plt.figure(figsize=(14, 8))
    for class_id, group in acc_df.groupby("class_id"):
        group = group.sort_values("snr")
        plt.plot(group["snr"], group["accuracy"], marker="o", linewidth=1.5, label=f"class {class_id}")

    plt.title("Classification Accuracy vs SNR by Modulation Class")
    plt.xlabel("SNR")
    plt.ylabel("Classification Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.show()

def main(config_path: str) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg["data"].get("dataset", "radioml2018") != "radioml2018":
        raise ValueError("SNR plotting currently expects the RadioML2018 HDF5 dataset with a Z array.")

    results_dir = cfg["results"]["dir"]
    model_path = keras_model_path(result_path(results_dir, cfg["results"].get("model", "tinyvgg.keras")))
    predictions_path = result_path(results_dir, cfg["results"].get("snr_predictions", "snr_predictions.csv"))
    plot_path = result_path(results_dir, cfg["results"].get("snr_plot", "snr_accuracy.png"))

    data_config = DataConfig(
        dataset="radioml2018",
        file_path=cfg["data"]["path"],
        data_ratio=cfg["data"].get("data_ratio", 0.01),
        train_ratio=cfg["data"].get("train_ratio", 0.8),
        batch_size=cfg["training"].get("batch_size", 16),
        shuffle_buffer=cfg["training"].get("shuffle_buffer", 512),
        seed=cfg["training"].get("seed", 42),
        eps=cfg["data"].get("eps", 1e-6),
        num_classes=cfg["model"].get("n_classes", 24),
        stratified=cfg["data"].get("stratified", True),
    )

    model = tf.keras.models.load_model(model_path)
    val_indices = build_val_indices(data_config)
    results_df = predict_by_snr(
        model=model,
        file_path=data_config.file_path,
        val_indices=val_indices,
        batch_size=data_config.batch_size,
        eps=data_config.eps,
    )

    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(predictions_path, index=False)
    plot_accuracy(results_df, plot_path)

    print(f"Predictions: {predictions_path}")
    print(f"Plot       : {plot_path}")

if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/tinyvgg.yaml"
    main(config)
