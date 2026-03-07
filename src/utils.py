"""
utils.py — Shared helpers used across the pipeline.
"""

import os
import json
import numpy as np


# ──────────────────────────────────────────────
# Directory helpers
# ──────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# ──────────────────────────────────────────────
# JSON helpers
# ──────────────────────────────────────────────

def save_json(data: dict, path: str) -> None:
    """Serialize *data* to a JSON file at *path*."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="latin-1") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    """Load and return a JSON file as a dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="latin-1") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Array validation
# ──────────────────────────────────────────────

def assert_3d(arr: np.ndarray, name: str = "X") -> None:
    """Raise ValueError if *arr* is not 3-dimensional."""
    if arr.ndim != 3:
        raise ValueError(
            f"Expected {name} to be 3-D (N, window, features), "
            f"got shape {arr.shape}"
        )


def assert_1d(arr: np.ndarray, name: str = "y") -> None:
    """Raise ValueError if *arr* is not 1-dimensional."""
    if arr.ndim != 1:
        raise ValueError(
            f"Expected {name} to be 1-D (N,), got shape {arr.shape}"
        )


def assert_lengths_match(a: np.ndarray, b: np.ndarray,
                          name_a: str = "X", name_b: str = "y") -> None:
    """Raise ValueError if first dimensions of *a* and *b* differ."""
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Length mismatch: {name_a} has {a.shape[0]} samples "
            f"but {name_b} has {b.shape[0]}."
        )


# ──────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────

def section(title: str) -> None:
    """Print a visible section header to stdout."""
    bar = "─" * 52
    print(f"\n{bar}\n  {title}\n{bar}")


def log_info(msg: str) -> None:
    """Print an info-level message."""
    print(f"  [INFO]  {msg}")


def log_warn(msg: str) -> None:
    """Print a warning-level message."""
    print(f"  [WARN]  {msg}")


def log_error(msg: str) -> None:
    """Print an error-level message."""
    print(f"  [ERROR] {msg}")