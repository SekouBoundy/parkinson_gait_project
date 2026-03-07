import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

X_PATH    = os.path.join(config.PROCESSED_DIR, "X.npy")
Y_PATH    = os.path.join(config.PROCESSED_DIR, "y.npy")
META_PATH = os.path.join(config.PROCESSED_DIR, "meta.json")

def _load():
    for p in [X_PATH, Y_PATH, META_PATH]:
        assert os.path.exists(p), f"Missing: {p}\n-> Run python -m src.preprocess_and_window first"
    return np.load(X_PATH), np.load(Y_PATH), json.load(open(META_PATH))

def test_X_is_3d():
    X, _, _ = _load(); assert X.ndim == 3, f"Expected 3D, got {X.shape}"

def test_y_is_1d():
    _, y, _ = _load(); assert y.ndim == 1, f"Expected 1D, got {y.shape}"

def test_N_matches():
    X, y, _ = _load(); assert X.shape[0] == y.shape[0], "N mismatch between X and y"

def test_window_size():
    X, _, _ = _load(); assert X.shape[1] == config.WINDOW_SIZE

def test_labels_binary():
    _, y, _ = _load(); assert set(np.unique(y).tolist()).issubset({0, 1})

def test_meta_consistent():
    X, y, m = _load()
    assert m["X_shape"] == list(X.shape)
    assert m["window_size"] == config.WINDOW_SIZE

if __name__ == "__main__":
    tests = [test_X_is_3d, test_y_is_1d, test_N_matches,
             test_window_size, test_labels_binary, test_meta_consistent]
    passed = 0
    for t in tests:
        try:
            t(); print(f"  OK  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
