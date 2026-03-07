"""
make_dataset.py — End-to-end dataset builder for the Parkinson gait pipeline.

Usage
-----
python -m src.make_dataset                        # use config.py defaults
python -m src.make_dataset --no_filter            # skip low-pass filter
python -m src.make_dataset --raw_dir data/raw --processed_dir data/processed

Outputs
-------
  data/processed/X.npy        shape (N, window_size, features)
  data/processed/y.npy        shape (N,)
  data/processed/meta.json    processing summary
"""

import argparse
import os
import sys
import numpy as np

# Allow running as  python -m src.make_dataset  from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data_loader   import discover_files, load_signal, infer_label
from src.preprocessing import lowpass_filter, standardize_signal, sliding_windows
from src.utils         import ensure_dir, save_json, validate_X_y, section, kv


# ──────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build processed dataset from raw gait files."
    )
    p.add_argument("--raw_dir",       default=config.RAW_DIR)
    p.add_argument("--processed_dir", default=config.PROCESSED_DIR)
    p.add_argument("--window_size",   type=int,   default=config.WINDOW_SIZE)
    p.add_argument("--stride",        type=int,   default=config.STRIDE)
    p.add_argument("--fs",            type=float, default=config.FS)
    p.add_argument("--cutoff_hz",     type=float, default=config.CUTOFF_HZ)
    p.add_argument(
        "--no_filter",
        action="store_true",
        help="Skip the low-pass filter step.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────
# Per-file processing
# ──────────────────────────────────────────────

def process_file(
    file_path:   str,
    window_size: int,
    stride:      int,
    fs:          float,
    cutoff_hz:   float,
    apply_filter: bool,
    raw_dir:     str = config.RAW_DIR,
) -> tuple[np.ndarray, int, dict] | None:
    """
    Run the full preprocessing chain on a single file.

    Returns
    -------
    (windows, label, file_meta)  on success
    None                         if the file should be skipped (with a warning)
    """
    try:
        label = infer_label(file_path, raw_dir=raw_dir)
    except ValueError as e:
        print(f"  [SKIP] {os.path.basename(file_path)}: {e}")
        return None

    try:
        signal, file_meta = load_signal(file_path)
    except (ValueError, NotImplementedError) as e:
        print(f"  [SKIP] {os.path.basename(file_path)}: {e}")
        return None

    # ── Signal conditioning ──
    if apply_filter:
        try:
            signal = lowpass_filter(signal, fs=fs, cutoff_hz=cutoff_hz)
        except ValueError as e:
            print(f"  [WARN] Filter skipped for {os.path.basename(file_path)}: {e}")

    signal = standardize_signal(signal)

    # ── Windowing ──
    windows = sliding_windows(signal, window_size=window_size, stride=stride)

    if windows.shape[0] == 0:
        print(
            f"  [SKIP] {os.path.basename(file_path)}: signal too short "
            f"({file_meta['num_rows']} rows < window_size={window_size})"
        )
        return None

    file_meta["n_windows"] = windows.shape[0]
    file_meta["label"]     = label
    return windows, label, file_meta


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def build_dataset(args: argparse.Namespace) -> None:
    apply_filter = (not args.no_filter)

    section("Parkinson Gait — Dataset Builder")
    kv("raw_dir",       args.raw_dir)
    kv("processed_dir", args.processed_dir)
    kv("window_size",   args.window_size)
    kv("stride",        args.stride)
    kv("fs",            f"{args.fs} Hz")
    kv("cutoff_hz",     f"{args.cutoff_hz} Hz")
    kv("low-pass filter", "ON" if apply_filter else "OFF")

    # ── Discover files ──
    section("Step 1 — Discovering files")
    files = discover_files(args.raw_dir)
    print(f"  Found {len(files)} supported file(s).\n")

    # ── Process each file ──
    section("Step 2 — Processing files")
    all_windows: list[np.ndarray] = []
    all_labels:  list[int]        = []
    file_reports: list[dict]      = []
    n_success = 0

    for fp in files:
        result = process_file(
            fp,
            window_size  = args.window_size,
            stride       = args.stride,
            fs           = args.fs,
            cutoff_hz    = args.cutoff_hz,
            apply_filter = apply_filter,
            raw_dir      = args.raw_dir,
        )
        if result is None:
            continue

        windows, label, fmeta = result
        all_windows.append(windows)
        all_labels.extend([label] * windows.shape[0])
        file_reports.append(fmeta)
        n_success += 1
        print(
            f"  OK  {os.path.basename(fp):<40} "
            f"label={label}  windows={windows.shape[0]}"
        )

    if n_success == 0:
        raise RuntimeError(
            "No files were successfully processed.\n"
            "Check that your data files are in the correct folder, "
            "have supported extensions, and contain numeric columns."
        )

    # ── Assemble arrays ──
    X = np.concatenate(all_windows, axis=0)   # (N, window_size, features)
    y = np.array(all_labels, dtype=np.int64)  # (N,)

    validate_X_y(X, y)

    # ── Class balance ──
    unique, counts = np.unique(y, return_counts=True)
    class_balance = {int(k): int(v) for k, v in zip(unique, counts)}

    # ── Feature names (consistent across files?) ──
    all_feature_names = [r["feature_names"] for r in file_reports]
    consistent = all(fn == all_feature_names[0] for fn in all_feature_names)
    feature_names = all_feature_names[0] if consistent else None

    # ── Build meta ──
    meta = {
        "X_shape":         list(X.shape),
        "y_shape":         list(y.shape),
        "number_of_files": len(files),
        "files_processed": n_success,
        "total_windows":   int(X.shape[0]),
        "class_balance":   class_balance,
        "window_size":     args.window_size,
        "stride":          args.stride,
        "fs":              args.fs,
        "cutoff_hz":       args.cutoff_hz,
        "apply_filter":    apply_filter,
        "num_features":    int(X.shape[2]),
        "feature_names":   feature_names,
        "per_file":        file_reports,
    }

    # ── Save outputs ──
    section("Step 3 — Saving outputs")
    ensure_dir(args.processed_dir)
    np.save(os.path.join(args.processed_dir, "X.npy"), X)
    np.save(os.path.join(args.processed_dir, "y.npy"), y)
    save_json(meta, os.path.join(args.processed_dir, "meta.json"))

    # ── Final report ──
    section("Done ✓")
    kv("Files found",       len(files))
    kv("Files processed",   n_success)
    kv("Files skipped",     len(files) - n_success)
    kv("Total windows (N)", X.shape[0])
    kv("Class balance",     class_balance)
    kv("X shape",           X.shape)
    kv("y shape",           y.shape)
    kv("Saved to",          args.processed_dir)
    print()


if __name__ == "__main__":
    args = _parse_args()
    build_dataset(args)