"""RadioML data loading for TinyVGG training."""

from __future__ import annotations

from dataclasses import dataclass
import pickle

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from .augmentation import augment_dataset
except ImportError:
    from augmentation import augment_dataset

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
    augment_train: bool = False
    awgn_std: float = 0.05
    rotation_count: int = 4

def preprocess(x: tf.Tensor, y: tf.Tensor, eps: float) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.reduce_mean(x, axis=0, keepdims=True)
    std = tf.math.reduce_std(x, axis=0, keepdims=True)
    return (x - mean) / (std + eps), tf.cast(y, tf.float32)

def split_indices(
    labels: np.ndarray,
    snrs: np.ndarray,
    config: DataConfig,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    train_parts, val_parts, rows = [], [], []
    groups = [(None, None)] if not config.stratified else sorted(set(zip(labels, snrs)))

    for class_id, snr in groups:
        indices = (
            np.arange(len(labels), dtype=np.int64)
            if class_id is None
            else np.flatnonzero((labels == class_id) & (snrs == snr))
        )
        rng.shuffle(indices)
        keep = int(len(indices) * config.data_ratio)
        if config.data_ratio > 0 and len(indices):
            keep = max(1, keep)
        indices = indices[:keep]
        split = int(len(indices) * config.train_ratio)
        train_idx, val_idx = indices[:split], indices[split:]
        train_parts.append(train_idx)
        val_parts.append(val_idx)

        if class_id is None:
            continue
        rows.append({
            "class_id": int(class_id),
            "snr": int(snr),
            "train": len(train_idx),
            "val": len(val_idx),
            "total": len(indices),
            "available": int(np.sum((labels == class_id) & (snrs == snr))),
        })

    train_indices = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_indices = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    if rows:
        return train_indices, val_indices, pd.DataFrame(rows)
    return train_indices, val_indices, count_split(labels, snrs, train_indices, val_indices)

def count_split(
    labels: np.ndarray,
    snrs: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for class_id, snr in sorted(set(zip(labels, snrs))):
        train = int(np.sum((labels[train_indices] == class_id) & (snrs[train_indices] == snr)))
        val = int(np.sum((labels[val_indices] == class_id) & (snrs[val_indices] == snr)))
        if train + val:
            rows.append({
                "class_id": int(class_id),
                "snr": int(snr),
                "train": train,
                "val": val,
                "total": train + val,
                "available": int(np.sum((labels == class_id) & (snrs == snr))),
            })
    return pd.DataFrame(rows)

def make_tf_dataset(
    config: DataConfig,
    output_shape: tuple[int, int],
    generator_fn,
    training: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=(
            tf.TensorSpec(shape=output_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(config.num_classes,), dtype=tf.int32),
        ),
    )
    if training and config.augment_train:
        ds = augment_dataset(ds, config.awgn_std, config.rotation_count)
    ds = ds.map(lambda x, y: preprocess(x, y, config.eps), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(config.shuffle_buffer, seed=config.seed, reshuffle_each_iteration=True)
    return ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

def load_radioml2018(config: DataConfig):
    with h5py.File(config.file_path, "r") as hf:
        labels = np.argmax(hf["Y"][:], axis=1)
        snrs = np.asarray(hf["Z"][:]).reshape(-1).astype(int)

    config = DataConfig(**{**config.__dict__, "num_classes": config.num_classes or 24})
    train_idx, val_idx, counts = split_indices(labels, snrs, config)

    def gen(indices):
        with h5py.File(config.file_path, "r") as hf:
            for index in indices:
                yield hf["X"][index].astype(np.float32), hf["Y"][index].astype(np.int32)

    return (
        make_tf_dataset(config, (1024, 2), lambda: gen(train_idx), True),
        make_tf_dataset(config, (1024, 2), lambda: gen(val_idx), False),
        counts,
        (1024, 2),
        config.num_classes,
    )

def load_radioml2016(config: DataConfig):
    with open(config.file_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    mods = sorted({key[0] for key in raw})
    mod_to_id = {mod: i for i, mod in enumerate(mods)}
    records = [
        (key, i, mod_to_id[key[0]], key[1])
        for key in sorted(raw, key=lambda item: (item[0], item[1]))
        for i in range(len(raw[key]))
    ]
    labels = np.array([record[2] for record in records])
    snrs = np.array([record[3] for record in records])
    config = DataConfig(**{**config.__dict__, "num_classes": config.num_classes or len(mods)})
    train_idx, val_idx, counts = split_indices(labels, snrs, config)

    def gen(indices):
        for index in indices:
            key, local_index, class_id, _ = records[index]
            x = raw[key][local_index].astype(np.float32)
            if x.shape[0] == 2:
                x = np.transpose(x, (1, 0))
            y = np.zeros(config.num_classes, dtype=np.int32)
            y[class_id] = 1
            yield x, y

    return (
        make_tf_dataset(config, (128, 2), lambda: gen(train_idx), True),
        make_tf_dataset(config, (128, 2), lambda: gen(val_idx), False),
        counts,
        (128, 2),
        config.num_classes,
    )

def build_datasets(config: DataConfig):
    loaders = {
        "radioml2018": load_radioml2018,
        "radioml2016": load_radioml2016,
    }
    try:
        return loaders[config.dataset.lower()](config)
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {config.dataset}") from exc

def snr_split_table(split_counts: pd.DataFrame) -> pd.DataFrame:
    summary = (
        split_counts
        .groupby("snr", as_index=False)[["train", "val", "total"]]
        .sum()
        .sort_values("snr")
    )
    summary["train_ratio"] = summary["train"] / summary["total"]
    summary["val_ratio"] = summary["val"] / summary["total"]
    return summary
