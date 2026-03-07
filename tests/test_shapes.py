"""
test_shapes.py — Validate processed output arrays before model training.

Run:
  python tests/test_shapes.py
or:
  pytest tests/test_shapes.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

X_PATH    = os.path.join(config.PROCESSED_DIR, "X.npy")
Y_PATH    = os.path.join(config.PROCESSED_DIR, "y.npy")
META_PATH = os.path.join(config.PROCESSED_DIR, "meta.json")


def _load():
    for p in [X_PATH, Y_PATH, META_PATH]:
        assert os.path.exists(p), (
            f"Missing: {p}\n"
            "-> Run  python -m src.make_dataset  first."
        )
    return (
        np.load(X_PATH),
        np.load(Y_PATH),
        json.load(open(META_PATH)),
    )


# ── Shape tests ───────────────────────────────

def test_X_is_3d():
    X, _, _ = _load()
    assert X.ndim == 3, f"X must be 3-D (N, window_size, features), got {X.shape}"


def test_y_is_1d():
    _, y, _ = _load()
    assert y.ndim == 1, f"y must be 1-D (N,), got {y.shape}"


def test_N_matches():
    X, y, _ = _load()
    assert X.shape[0] == y.shape[0], (
        f"Sample count mismatch: X has {X.shape[0]}, y has {y.shape[0]}"
    )


def test_window_size():
    X, _, _ = _load()
    assert X.shape[1] == config.WINDOW_SIZE, (
        f"Window size: expected {config.WINDOW_SIZE}, got {X.shape[1]}"
    )


def test_at_least_one_feature():
    X, _, _ = _load()
    assert X.shape[2] >= 1, f"Expected >= 1 feature, got {X.shape[2]}"


def test_not_empty():
    X, y, _ = _load()
    assert X.shape[0] > 0, "X is empty — no windows were generated"
    assert y.shape[0] > 0, "y is empty — no labels were generated"


def test_labels_binary():
    _, y, _ = _load()
    unique = set(np.unique(y).tolist())
    assert unique.issubset({0, 1}), (
        f"Labels must be 0 or 1, found: {unique}"
    )


def test_meta_consistent():
    X, y, m = _load()
    assert m["X_shape"] == list(X.shape), (
        f"meta X_shape {m['X_shape']} != actual {list(X.shape)}"
    )
    assert m["window_size"] == config.WINDOW_SIZE, (
        f"meta window_size {m['window_size']} != config {config.WINDOW_SIZE}"
    )
    assert m["total_windows"] == int(X.shape[0]), (
        f"meta total_windows {m['total_windows']} != X.shape[0] {X.shape[0]}"
    )


# ── Runner ────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_X_is_3d,
        test_y_is_1d,
        test_N_matches,
        test_window_size,
        test_at_least_one_feature,
        test_not_empty,
        test_labels_binary,
        test_meta_consistent,
    ]

    passed = 0
    print("\nRunning shape tests...\n")
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}\n         {e}")

    print(f"\n{'='*40}")
    print(f"  {passed}/{len(tests)} tests passed")
    print(f"{'='*40}\n")
    sys.exit(0 if passed == len(tests) else 1)