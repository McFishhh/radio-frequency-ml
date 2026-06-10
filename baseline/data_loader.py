"""Lazy RadioML HDF5 data loading for TinyVGG training."""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

FEATURE_SIGNATURE = tf.TensorSpec(shape=(1024, 2), dtype=tf.float32)
LABEL_SIGNATURE = tf.TensorSpec(shape=(24,), dtype=tf.int32)

@dataclass(frozen=True)
class DataConfig:
    file_path: str
    data_ratio: float = 0.01
    train_ratio: float = 0.8
    batch_size: int = 16
    shuffle_buffer: int = 512
    seed: int = 42
    eps: float = 1e-6
    num_classes: int = 24
    stratified: bool = True

def sample_count(path: str, data_ratio: float) -> int:
    with h5py.File(path, "r") as hf:
        return int(len(hf["X"]) * data_ratio)

def random_split_indices(config: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(sample_count(config.file_path, config.data_ratio), dtype=np.int64)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(indices)
    split = int(len(indices) * config.train_ratio)
    return indices[:split], indices[split:]

def stratified_split_indices(
    config: DataConfig,
    chunk_size: int = 65_536,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    n_samples = sample_count(config.file_path, config.data_ratio)
    by_class: list[list[np.ndarray]] = [[] for _ in range(config.num_classes)]

    with h5py.File(config.file_path, "r") as hf:
        y_ds = hf["Y"]
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            labels = np.argmax(y_ds[start:end], axis=1)
            for class_id in range(config.num_classes):
                local_indices = np.flatnonzero(labels == class_id)
                if len(local_indices):
                    by_class[class_id].append(local_indices + start)

    train_parts, test_parts, rows = [], [], []
    for class_id, parts in enumerate(by_class):
        indices = np.concatenate(parts).astype(np.int64) if parts else np.array([], dtype=np.int64)
        rng.shuffle(indices)
        split = int(len(indices) * config.train_ratio)
        train_idx, test_idx = indices[:split], indices[split:]
        train_parts.append(train_idx)
        test_parts.append(test_idx)
        rows.append({
            "class_id": class_id,
            "train": len(train_idx),
            "test": len(test_idx),
            "total": len(indices),
        })

    train_indices = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    test_indices = np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices, pd.DataFrame(rows)

def label_counts(
    path: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    num_classes: int,
) -> pd.DataFrame:
    counts = {"train": np.zeros(num_classes, dtype=np.int64), "test": np.zeros(num_classes, dtype=np.int64)}

    with h5py.File(path, "r") as hf:
        y_ds = hf["Y"]
        for split, indices in {"train": train_indices, "test": test_indices}.items():
            for index in indices:
                counts[split][int(np.argmax(y_ds[index]))] += 1

    return pd.DataFrame({
        "class_id": np.arange(num_classes),
        "train": counts["train"],
        "test": counts["test"],
        "total": counts["train"] + counts["test"],
    })

def hdf5_generator(path: str, indices: np.ndarray):
    with h5py.File(path, "r") as hf:
        x_ds, y_ds = hf["X"], hf["Y"]
        for index in indices:
            yield x_ds[index].astype(np.float32), y_ds[index].astype(np.int32)

def preprocess(x: tf.Tensor, y: tf.Tensor, eps: float) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.reduce_mean(x, axis=0, keepdims=True)
    std = tf.math.reduce_std(x, axis=0, keepdims=True)
    x = tf.expand_dims((x - mean) / (std + eps), -1)
    return x, tf.cast(y, tf.float32)

def make_dataset(
    config: DataConfig,
    indices: np.ndarray,
    training: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_generator(
        lambda: hdf5_generator(config.file_path, indices),
        output_signature=(FEATURE_SIGNATURE, LABEL_SIGNATURE),
    )
    ds = ds.map(lambda x, y: preprocess(x, y, config.eps), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(config.shuffle_buffer, seed=config.seed, reshuffle_each_iteration=True)
    return ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

def build_datasets(config: DataConfig) -> tuple[tf.data.Dataset, tf.data.Dataset, pd.DataFrame]:
    if config.stratified:
        train_indices, test_indices, counts = stratified_split_indices(config)
    else:
        train_indices, test_indices = random_split_indices(config)
        counts = label_counts(config.file_path, train_indices, test_indices, config.num_classes)

    return (
        make_dataset(config, train_indices, training=True),
        make_dataset(config, test_indices, training=False),
        counts,
    )
