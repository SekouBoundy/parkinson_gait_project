"""
preprocess_and_window.py — End-to-end dataset builder.

Run from the project root:
    python -m src.preprocess_and_window
    python -m src.preprocess_and_window --no_filter
    python -m src.preprocess_and_window --raw_dir data/raw --window_size 256
"""

import os
import sys
import argparse
import numpy as np

# ── Make sure project root is on the path when run as a script ────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data_loader   import discover_files, load_signal, infer_label
from src.preprocessing import lowpass_filter, standardize_signal, sliding_windows
from src.utils         import (
    ensure_dir, save_json,
    section, log_info, log_warn, log_error,
)


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build processed numpy arrays from raw gait files."
    )
    p.add_argument("--raw_dir",       default=config.RAW_DIR)
    p.add_argument("--processed_dir", default=config.PROCESSED_DIR)
    p.add_argument("--window_size",   type=int,   default=config.WINDOW_SIZE)
    p.add_argument("--stride",        type=int,   default=config.STRIDE)
    p.add_argument("--fs",            type=int,   default=config.FS)
    p.add_argument("--cutoff_hz",     type=float, default=config.CUTOFF_HZ)
    p.add_argument(
        "--no_filter",
        action="store_true",
        help="Skip the low-pass filter step.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ──────────────────────────────────────────────────────────────────────────────

def process_file(
    file_path:   str,
    window_size: int,
    stride:      int,
    fs:          int,
    cutoff_hz:   float,
    apply_filter: bool,
) -> tuple[np.ndarray, int, dict] | None:
    """
    Run the full preprocessing pipeline on a single file.

    Returns
    -------
    (windows, label, file_meta)  on success, or None on failure.
    """
    try:
        label   = infer_label(file_path)
        signal, file_meta = load_signal(file_path)

        if apply_filter:
            signal = lowpass_filter(signal, fs, cutoff_hz)

        signal  = standardize_signal(signal)
        windows = sliding_windows(signal, window_size, stride)

        if windows.shape[0] == 0:
            log_warn(
                f"Signal too short to produce any windows "
                f"({file_meta['num_rows']} rows < window_size={window_size}). "
                f"Skipped: {os.path.basename(file_path)}"
            )
            return None

        file_meta["n_windows"] = windows.shape[0]
        file_meta["label"]     = label
        return windows, label, file_meta

    except (ValueError, NotImplementedError) as exc:
        log_error(str(exc))
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    section("Parkinson Gait — Dataset Builder")
    log_info(f"raw_dir       : {args.raw_dir}")
    log_info(f"processed_dir : {args.processed_dir}")
    log_info(f"window_size   : {args.window_size}")
    log_info(f"stride        : {args.stride}")
    log_info(f"fs            : {args.fs} Hz")
    log_info(f"cutoff_hz     : {args.cutoff_hz} Hz")
    log_info(f"filter        : {'OFF' if args.no_filter else 'ON'}")

    # ── 1. Discover files ──────────────────────────────────────────────────
    section("Step 1 · Discovering files")
    try:
        files = discover_files(args.raw_dir)
    except FileNotFoundError as exc:
        log_error(str(exc))
        sys.exit(1)

    log_info(f"Found {len(files)} file(s)")

    # ── 2. Process each file ───────────────────────────────────────────────
    section("Step 2 · Processing files")

    all_windows:  list[np.ndarray] = []
    all_labels:   list[np.ndarray] = []
    file_metas:   list[dict]       = []
    n_features_set: set            = set()

    for fp in files:
        result = process_file(
            fp,
            window_size=args.window_size,
            stride=args.stride,
            fs=args.fs,
            cutoff_hz=args.cutoff_hz,
            apply_filter=not args.no_filter,
        )
        if result is None:
            continue

        windows, label, meta = result
        all_windows.append(windows)
        all_labels.append(np.full(windows.shape[0], label, dtype=np.int64))
        file_metas.append(meta)
        n_features_set.add(windows.shape[2])
        log_info(
            f"OK  {os.path.basename(fp):40s}  "
            f"windows={windows.shape[0]:4d}  label={label}"
        )

    n_ok = len(file_metas)
    log_info(f"\n{n_ok}/{len(files)} file(s) processed successfully.")

    # ── 3. Validate & concatenate ──────────────────────────────────────────
    section("Step 3 · Assembling arrays")

    if n_ok == 0:
        raise RuntimeError(
            "No valid windows were generated.\n"
            "  -> Check that your files are correctly formatted and placed in:\n"
            f"     {args.raw_dir}/parkinson/  and  {args.raw_dir}/control/"
        )

    if len(n_features_set) > 1:
        log_warn(
            f"Files have inconsistent feature counts: {n_features_set}.\n"
            "  -> Only files with the same number of columns can be stacked.\n"
            "  -> Fix your dataset or select a consistent column subset in config."
        )
        # Keep only the most common feature count
        dominant = max(n_features_set, key=lambda n: sum(
            1 for m in file_metas if m["num_features"] == n
        ))
        keep_idx = [
            i for i, m in enumerate(file_metas)
            if m["num_features"] == dominant
        ]
        all_windows = [all_windows[i] for i in keep_idx]
        all_labels  = [all_labels[i]  for i in keep_idx]
        file_metas  = [file_metas[i]  for i in keep_idx]
        log_warn(f"  Keeping {len(keep_idx)} file(s) with {dominant} feature(s).")

    X = np.concatenate(all_windows, axis=0)   # (N, window_size, features)
    y = np.concatenate(all_labels,  axis=0)   # (N,)

    # ── 4. Save outputs ────────────────────────────────────────────────────
    section("Step 4 · Saving outputs")

    ensure_dir(args.processed_dir)
    x_path    = os.path.join(args.processed_dir, "X.npy")
    y_path    = os.path.join(args.processed_dir, "y.npy")
    meta_path = os.path.join(args.processed_dir, "meta.json")

    np.save(x_path, X)
    np.save(y_path, y)

    unique, counts   = np.unique(y, return_counts=True)
    class_balance    = {int(k): int(v) for k, v in zip(unique, counts)}
    feature_names    = file_metas[0]["feature_names"] if file_metas else []
    per_file_windows = {
        os.path.basename(m["file_path"]): m["n_windows"]
        for m in file_metas
    }

    meta = {
        "number_of_files":    n_ok,
        "total_windows":      int(X.shape[0]),
        "class_balance":      class_balance,
        "window_size":        args.window_size,
        "stride":             args.stride,
        "fs":                 args.fs,
        "cutoff_hz":          args.cutoff_hz,
        "filter_applied":     not args.no_filter,
        "num_features":       int(X.shape[2]),
        "feature_names":      feature_names,
        "X_shape":            list(X.shape),
        "y_shape":            list(y.shape),
        "per_file_windows":   per_file_windows,
    }
    save_json(meta, meta_path)

    log_info(f"Saved: {x_path}")
    log_info(f"Saved: {y_path}")
    log_info(f"Saved: {meta_path}")

    # ── 5. Final report ────────────────────────────────────────────────────
    section("Done ✓")
    log_info(f"Files found        : {len(files)}")
    log_info(f"Files processed    : {n_ok}")
    log_info(f"Total windows      : {X.shape[0]}")
    log_info(f"Class balance      : { {0: 'control', 1: 'parkinson'}}")
    for cls, cnt in class_balance.items():
        label_name = "control" if cls == 0 else "parkinson"
        log_info(f"  label {cls} ({label_name:9s}) : {cnt} windows")
    log_info(f"X shape            : {X.shape}  (N, window_size, features)")
    log_info(f"y shape            : {y.shape}  (N,)")
    print()


if __name__ == "__main__":
    main()