"""RadioML data loading for TinyVGG training.

Supported datasets:
    radioml2018: HDF5 file with X, Y, Z arrays. X is usually (N, 1024, 2).
    radioml2016: Pickle dict keyed by (modulation, snr). Values are usually
                 shaped (N, 2, 128), and are transposed to (128, 2).
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import pickle

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

@dataclass(frozen=True)
class DataConfig:
    dataset: str
    file_path: str
    data_ratio: float = 1.0
    train_ratio: float = 0.8
    batch_size: int = 16
    shuffle_buffer: int = 512
    seed: int = 42
    eps: float = 1e-6
    num_classes: int | None = None
    stratified: bool = True

def preprocess(x: tf.Tensor, y: tf.Tensor, eps: float) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.reduce_mean(x, axis=0, keepdims=True)
    std = tf.math.reduce_std(x, axis=0, keepdims=True)
    x = tf.expand_dims((x - mean) / (std + eps), -1)
    return x, tf.cast(y, tf.float32)

def make_tf_dataset(
    config: DataConfig,
    generator_fn,
    output_shape: tuple[int, int],
    training: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=(
            tf.TensorSpec(shape=output_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(config.num_classes,), dtype=tf.int32),
        ),
    )
    ds = ds.map(lambda x, y: preprocess(x, y, config.eps), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(config.shuffle_buffer, seed=config.seed, reshuffle_each_iteration=True)
    return ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

def hdf5_sample_count(path: str, data_ratio: float) -> int:
    with h5py.File(path, "r") as hf:
        return int(len(hf["X"]) * data_ratio)

def hdf5_total_count(path: str) -> int:
    with h5py.File(path, "r") as hf:
        return len(hf["X"])

def hdf5_random_split(config: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    n = hdf5_total_count(config.file_path)
    rng = random.Random(config.seed)
    indices = rng.sample(range(n), int(n * config.data_ratio))
    split = int(len(indices) * config.train_ratio)
    return np.array(indices[:split], dtype=np.int64), np.array(indices[split:], dtype=np.int64)

def hdf5_stratified_split(config: DataConfig, chunk_size: int = 65_536):
    rng = np.random.default_rng(config.seed)
    by_class: list[list[np.ndarray]] = [[] for _ in range(config.num_classes)]

    with h5py.File(config.file_path, "r") as hf:
        y_ds = hf["Y"]
        n_total = len(hf["X"])

        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            labels = np.argmax(y_ds[start:end], axis=1)
            for class_id in range(config.num_classes):
                local_indices = np.flatnonzero(labels == class_id)
                if len(local_indices):
                    by_class[class_id].append(local_indices + start)

    train_parts, val_parts, rows = [], [], []
    for class_id, parts in enumerate(by_class):
        indices = np.concatenate(parts).astype(np.int64) if parts else np.array([], dtype=np.int64)
        rng.shuffle(indices)
        sample_count = int(len(indices) * config.data_ratio)
        sampled_indices = indices[:sample_count]
        split = int(len(sampled_indices) * config.train_ratio)
        train_idx, val_idx = sampled_indices[:split], sampled_indices[split:]
        train_parts.append(train_idx)
        val_parts.append(val_idx)
        rows.append({
            "class_id": class_id,
            "train": len(train_idx),
            "val": len(val_idx),
            "total": len(sampled_indices),
            "available": len(indices),
        })

    train_indices = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_indices = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices, pd.DataFrame(rows)

def hdf5_label_counts(config: DataConfig, train_indices: np.ndarray, val_indices: np.ndarray) -> pd.DataFrame:
    counts = {"train": np.zeros(config.num_classes, dtype=np.int64), "val": np.zeros(config.num_classes, dtype=np.int64)}
    with h5py.File(config.file_path, "r") as hf:
        y_ds = hf["Y"]
        for split, indices in {"train": train_indices, "val": val_indices}.items():
            for index in indices:
                counts[split][int(np.argmax(y_ds[index]))] += 1
    return pd.DataFrame({
        "class_id": np.arange(config.num_classes),
        "train": counts["train"],
        "val": counts["val"],
        "total": counts["train"] + counts["val"],
    })

def hdf5_generator(path: str, indices: np.ndarray):
    with h5py.File(path, "r") as hf:
        x_ds, y_ds = hf["X"], hf["Y"]
        for index in indices:
            yield x_ds[index].astype(np.float32), y_ds[index].astype(np.int32)

def build_radioml2018(config: DataConfig):
    if config.num_classes is None:
        config = DataConfig(**{**config.__dict__, "num_classes": 24})

    if config.stratified:
        train_indices, val_indices, counts = hdf5_stratified_split(config)
    else:
        train_indices, val_indices = hdf5_random_split(config)
        counts = hdf5_label_counts(config, train_indices, val_indices)

    train_ds = make_tf_dataset(config, lambda: hdf5_generator(config.file_path, train_indices), (1024, 2), True)
    val_ds = make_tf_dataset(config, lambda: hdf5_generator(config.file_path, val_indices), (1024, 2), False)
    return train_ds, val_ds, counts, (1024, 2, 1), config.num_classes

def load_radioml2016_records(path: str, data_ratio: float):
    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    mods = sorted({key[0] for key in raw.keys()})
    mod_to_id = {mod: index for index, mod in enumerate(mods)}
    records = []

    for key in sorted(raw.keys(), key=lambda item: (item[0], item[1])):
        mod, snr = key
        sample_count = int(len(raw[key]) * data_ratio)
        for local_index in range(sample_count):
            records.append((key, local_index, mod_to_id[mod], snr))

    return raw, records, mods

def split_records(records: list[tuple], num_classes: int, train_ratio: float, seed: int, stratified: bool):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records), dtype=np.int64)

    if not stratified:
        rng.shuffle(indices)
        split = int(len(indices) * train_ratio)
        return indices[:split], indices[split:]

    train_parts, val_parts = [], []
    labels = np.array([record[2] for record in records])
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        split = int(len(class_indices) * train_ratio)
        train_parts.append(class_indices[:split])
        val_parts.append(class_indices[split:])

    train_indices = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_indices = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices

def radioml2016_counts(records: list[tuple], train_indices: np.ndarray, val_indices: np.ndarray, num_classes: int):
    counts = {"train": np.zeros(num_classes, dtype=np.int64), "val": np.zeros(num_classes, dtype=np.int64)}
    for split, indices in {"train": train_indices, "val": val_indices}.items():
        for index in indices:
            counts[split][records[index][2]] += 1
    return pd.DataFrame({
        "class_id": np.arange(num_classes),
        "train": counts["train"],
        "val": counts["val"],
        "total": counts["train"] + counts["val"],
    })

def radioml2016_generator(raw: dict, records: list[tuple], indices: np.ndarray, num_classes: int):
    for index in indices:
        key, local_index, class_id, _ = records[index]
        x = raw[key][local_index].astype(np.float32)
        if x.shape[0] == 2:
            x = np.transpose(x, (1, 0))
        y = np.zeros(num_classes, dtype=np.int32)
        y[class_id] = 1
        yield x, y

def build_radioml2016(config: DataConfig):
    raw, records, mods = load_radioml2016_records(config.file_path, config.data_ratio)
    num_classes = config.num_classes or len(mods)
    config = DataConfig(**{**config.__dict__, "num_classes": num_classes})
    train_indices, val_indices = split_records(records, num_classes, config.train_ratio, config.seed, config.stratified)
    counts = radioml2016_counts(records, train_indices, val_indices, num_classes)

    train_ds = make_tf_dataset(
        config,
        lambda: radioml2016_generator(raw, records, train_indices, num_classes),
        (128, 2),
        True,
    )
    val_ds = make_tf_dataset(
        config,
        lambda: radioml2016_generator(raw, records, val_indices, num_classes),
        (128, 2),
        False,
    )
    return train_ds, val_ds, counts, (128, 2, 1), num_classes

def build_datasets(config: DataConfig):
    dataset = config.dataset.lower()
    if dataset == "radioml2018":
        return build_radioml2018(config)
    if dataset == "radioml2016":
        return build_radioml2016(config)
    raise ValueError(f"Unsupported dataset: {config.dataset}")
