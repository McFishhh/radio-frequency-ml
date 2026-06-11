"""Preprocess RadioML2016.10a pickle data into an HDF5 training file.

This follows the structure of the PyTorch prepare_data.py outline:
    - load raw RML2016.10a_dict.pkl
    - select AWGN samples at a target SNR, default 18 dB
    - take the first train_ratio samples per modulation
    - normalize each sample to unit RMS
    - shuffle deterministically
    - save x/y arrays to HDF5

Usage:
    python baseline/preprocessing_2.py
    python baseline/preprocessing_2.py --raw-path data/raw/RML2016.10a_dict.pkl --out-path data/processed/train_awgn18.hdf5
"""

from __future__ import annotations

import argparse
import os
import pickle

import h5py
import numpy as np

MODS = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]
MOD2IDX = {mod: idx for idx, mod in enumerate(MODS)}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess RadioML2016.10a into HDF5.")
    parser.add_argument("--raw-path", default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--out-path", default="data/processed/train_awgn18.hdf5")
    parser.add_argument("--snr", type=int, default=18)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()

def unit_rms_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=(1, 2), keepdims=True))
    return x / (rms + eps)

def build_awgn_snr_split(
    data: dict,
    snr: int,
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_parts = []
    y_parts = []
    snr_parts = []

    for mod in MODS:
        key = (mod, snr)
        if key not in data:
            raise KeyError(f"Missing RadioML2016 key: {key}")

        samples = data[key].astype(np.float32)
        n_train = int(len(samples) * train_ratio)

        x_parts.append(samples[:n_train])
        y_parts.extend([MOD2IDX[mod]] * n_train)
        snr_parts.extend([snr] * n_train)

    x = np.concatenate(x_parts, axis=0).astype(np.float32)
    y = np.asarray(y_parts, dtype=np.int64)
    snrs = np.asarray(snr_parts, dtype=np.int64)
    return x, y, snrs

def shuffle_arrays(
    x: np.ndarray,
    y: np.ndarray,
    snrs: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    return x[perm], y[perm], snrs[perm]

def save_hdf5(
    out_path: str,
    x: np.ndarray,
    y: np.ndarray,
    snrs: np.ndarray,
    snr: int,
    train_ratio: float,
    seed: int,
    normalized: bool,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("x", data=x.astype(np.float32))
        f.create_dataset("y", data=y.astype(np.int64))
        f.create_dataset("snr", data=snrs.astype(np.int64))
        f.create_dataset("mods", data=np.asarray(MODS, dtype="S"))

        f.attrs["source_dataset"] = "RadioML2016.10a"
        f.attrs["target_snr"] = snr
        f.attrs["train_ratio"] = train_ratio
        f.attrs["seed"] = seed
        f.attrs["unit_rms_normalized"] = normalized
        f.attrs["x_shape"] = str(x.shape)
        f.attrs["y_shape"] = str(y.shape)

def main() -> None:
    args = parse_args()

    with open(args.raw_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    x, y, snrs = build_awgn_snr_split(
        data=data,
        snr=args.snr,
        train_ratio=args.train_ratio,
    )

    if not args.no_normalize:
        x = unit_rms_normalize(x)

    x, y, snrs = shuffle_arrays(x, y, snrs, args.seed)
    save_hdf5(
        out_path=args.out_path,
        x=x,
        y=y,
        snrs=snrs,
        snr=args.snr,
        train_ratio=args.train_ratio,
        seed=args.seed,
        normalized=not args.no_normalize,
    )

    print(f"Saved: {args.out_path}")
    print(f"x: {x.shape} float32")
    print(f"y: {y.shape} int64")
    print(f"snr: {snrs.shape} int64")
    print(f"class counts: {np.bincount(y, minlength=len(MODS)).tolist()}")

if __name__ == "__main__":
    main()
