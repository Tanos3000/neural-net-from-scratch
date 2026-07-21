"""Parses the raw MNIST IDX file format into NumPy arrays.

MNIST files aren't CSV or JSON - they're a simple custom binary format
(a few header numbers, then a big block of raw bytes). Parsing it by hand
here, rather than using a library, is one more piece of "understanding the
whole pipeline", which is the point of this project.
"""

import gzip
import struct
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n_images, n_rows, n_cols = struct.unpack(">IIII", f.read(16))
        buffer = f.read()
    images = np.frombuffer(buffer, dtype=np.uint8).reshape(n_images, n_rows * n_cols)
    return images.astype(np.float64)


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        buffer = f.read()
    return np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)


def load_mnist():
    """Returns (X_train, y_train, X_test, y_test).

    X arrays: shape (n_samples, 784), pixel values scaled to [0, 1].
    y arrays: shape (n_samples,), integer digit labels 0-9.
    """
    X_train = _read_images(DATA_DIR / "train-images-idx3-ubyte.gz") / 255.0
    y_train = _read_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
    X_test = _read_images(DATA_DIR / "t10k-images-idx3-ubyte.gz") / 255.0
    y_test = _read_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")
    return X_train, y_train, X_test, y_test
