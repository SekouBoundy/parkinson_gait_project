"""
utils.py — Shared helper functions used across the pipeline.
"""

import os
import json
import numpy as np


# ──────────────────────────────────────────────
# Directory helpers
# ──────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """Create a directory (and all parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# ──────────────────────────────────────────────
# JSON helpers
# ──────────────────────────────────────────────

def save_json(data: dict, path: str) -> None:
    """
    Save a dictionary to a JSON file.

    Parameters
    ----------
    data : dict   Any JSON-serialisable dictionary.
    path : str    Full file path to write to.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    """Load and return a JSON file as a Python dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Array validation
# ──────────────────────────────────────────────

def validate_X_y(X: np.ndarray, y: np.ndarray) -> None:
    """
    Assert that X and y have the expected shapes for the deep-learning pipeline:
      X : (N, window_size, features)  — 3-D
      y : (N,)                        — 1-D
    Raises ValueError with a clear message on any violation.
    """
    if X.ndim != 3:
        raise ValueError(
            f"X must be 3-D (N, window_size, features), got shape {X.shape}"
        )
    if y.ndim != 1:
        raise ValueError(
            f"y must be 1-D (N,), got shape {y.shape}"
        )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Sample count mismatch: X has {X.shape[0]} rows, y has {y.shape[0]} rows"
        )
    if X.shape[0] == 0:
        raise ValueError("X and y are empty — no windows were generated.")


# ──────────────────────────────────────────────
# Pretty-print helpers
# ──────────────────────────────────────────────

def section(title: str) -> None:
    """Print a titled section divider to stdout."""
    bar = "─" * 56
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def kv(key: str, value) -> None:
    """Print a single key/value line, aligned for readability."""
    print(f"  {key:<30} {value}")