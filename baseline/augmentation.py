"""IQ data augmentation utilities for tf.data pipelines."""

from __future__ import annotations

import tensorflow as tf

def add_awgn(x: tf.Tensor, noise_std: float) -> tf.Tensor:
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std, dtype=x.dtype)
    return x + noise

def rotate_iq(x: tf.Tensor, theta: tf.Tensor) -> tf.Tensor:
    cos_theta = tf.cos(theta)
    sin_theta = tf.sin(theta)
    rotation = tf.stack(
        [
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta],
        ]
    )
    return tf.matmul(x, rotation)

def equally_spaced_angles(rotation_count: int, dtype: tf.DType) -> tf.Tensor:
    step = tf.constant(2.0 * 3.141592653589793, dtype=dtype) / tf.cast(rotation_count + 1, dtype)
    return tf.cast(tf.range(1, rotation_count + 1), dtype) * step

def augment_iq_sample(
    x: tf.Tensor,
    y: tf.Tensor,
    awgn_std: float = 0.05,
    rotation_count: int = 4,
) -> tf.data.Dataset:
    augmented_x = [x, add_awgn(x, awgn_std)]
    angles = equally_spaced_angles(rotation_count, x.dtype)
    augmented_x.extend(rotate_iq(x, angle) for angle in tf.unstack(angles))

    x_out = tf.stack(augmented_x, axis=0)
    y_out = tf.repeat(tf.expand_dims(y, axis=0), repeats=1 + 1 + rotation_count, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_out, y_out))

def augment_dataset(
    ds: tf.data.Dataset,
    awgn_std: float = 0.05,
    rotation_count: int = 4,
) -> tf.data.Dataset:
    return ds.flat_map(
        lambda x, y: augment_iq_sample(
            x,
            y,
            awgn_std=awgn_std,
            rotation_count=rotation_count,
        )
    )
