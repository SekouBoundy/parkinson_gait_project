"""
Filter -> Z-score -> Sliding windows -> saves X.npy, y.npy, meta.json
Usage:
    python -m src.preprocess_and_window
    python -m src.preprocess_and_window --no_filter
"""
import os, sys, json, logging, argparse
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.inspect_dataset import _load_file, _infer_label

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def lowpass_filter(signal: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        log.warning("cutoff_hz >= Nyquist – skipping filter"); return signal
    b, a = butter(N=4, Wn=cutoff_hz / nyq, btype="low")
    if signal.ndim == 1:
        return filtfilt(b, a, signal).astype(np.float32)
    out = np.empty_like(signal)
    for c in range(signal.shape[1]):
        out[:, c] = filtfilt(b, a, signal[:, c])
    return out.astype(np.float32)


def standardize(signal: np.ndarray) -> np.ndarray:
    if signal.ndim == 1:
        std = signal.std()
        return ((signal - signal.mean()) / std).astype(np.float32) if std > 1e-8 else signal
    out = np.empty_like(signal, dtype=np.float32)
    for c in range(signal.shape[1]):
        col = signal[:, c]; std = col.std()
        out[:, c] = (col - col.mean()) / std if std > 1e-8 else col
    return out


def sliding_windows(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    T, C = signal.shape
    if T < window_size:
        return np.empty((0, window_size, C), dtype=np.float32)
    starts = range(0, T - window_size + 1, stride)
    return np.stack([signal[s:s+window_size] for s in starts]).astype(np.float32)


def process_file(fp, window_size, stride, fs, cutoff_hz, apply_filter):
    arr, err = _load_file(fp)
    if err:
        log.warning("  x %s -> %s", fp.name, err); return None
    label = _infer_label(fp)
    if label is None:
        log.warning("  ? %s -> label unknown – skipping", fp.name); return None
    if apply_filter:
        arr = lowpass_filter(arr, fs, cutoff_hz)
    arr = standardize(arr)
    windows = sliding_windows(arr, window_size, stride)
    if windows.shape[0] == 0:
        log.warning("  ! %s -> too short for one window (T=%d)", fp.name, arr.shape[0]); return None
    log.info("  OK %-40s windows=%d shape=%s label=%d", fp.name, windows.shape[0], windows.shape, label)
    return windows, label


def run(raw_dir, processed_dir, window_size, stride, fs, cutoff_hz, apply_filter):
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        log.error("RAW_DIR not found: %s", raw_path); sys.exit(1)
    all_files = sorted([f for f in raw_path.rglob("*") if f.is_file()])
    if not all_files:
        log.error("No files in %s", raw_path); sys.exit(1)

    X_list, y_list, skipped = [], [], 0
    for fp in all_files:
        result = process_file(fp, window_size, stride, fs, cutoff_hz, apply_filter)
        if result is None:
            skipped += 1; continue
        windows, label = result
        X_list.append(windows)
        y_list.extend([label] * len(windows))

    if not X_list:
        log.error("No usable data – check formats and folder names."); sys.exit(1)

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    unique, counts = np.unique(y, return_counts=True)
    class_balance = {int(k): int(v) for k, v in zip(unique, counts)}

    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, "X.npy"), X)
    np.save(os.path.join(processed_dir, "y.npy"), y)
    meta = {"X_shape": list(X.shape), "y_shape": list(y.shape),
            "window_size": window_size, "stride": stride, "fs": fs,
            "cutoff_hz": cutoff_hz, "filter_applied": apply_filter,
            "files_processed": len(X_list), "files_skipped": skipped,
            "class_balance": class_balance, "channels_C": X.shape[2]}
    with open(os.path.join(processed_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "="*60)
    print("  PREPROCESSING REPORT")
    print("="*60)
    print(f"  Files processed : {len(X_list)}")
    print(f"  Files skipped   : {skipped}")
    print(f"  Total windows N : {X.shape[0]}")
    print(f"  X shape         : {X.shape}  (N, window_size, C)")
    print(f"  Class balance   : {class_balance}  (0=control, 1=parkinson)")
    print(f"  Filter applied  : {apply_filter} ({cutoff_hz} Hz)")
    print(f"  Saved to        : {processed_dir}/")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",       default=config.RAW_DIR)
    parser.add_argument("--processed_dir", default=config.PROCESSED_DIR)
    parser.add_argument("--window_size",   type=int,   default=config.WINDOW_SIZE)
    parser.add_argument("--stride",        type=int,   default=config.STRIDE)
    parser.add_argument("--fs",            type=float, default=config.FS)
    parser.add_argument("--cutoff_hz",     type=float, default=config.CUTOFF_HZ)
    parser.add_argument("--no_filter",     action="store_true")
    args = parser.parse_args()
    run(args.raw_dir, args.processed_dir, args.window_size,
        args.stride, args.fs, args.cutoff_hz, not args.no_filter)

if __name__ == "__main__":
    main()
