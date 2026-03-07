"""
inspect_dataset.py — Quick sanity check for the raw dataset folder.

Prints a readable summary without running the full pipeline.

Usage:
    python -m src.inspect_dataset
    python -m src.inspect_dataset --raw_dir data/raw
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data_loader import discover_files, infer_label
from src.utils import section, log_info, log_warn, log_error


def inspect_file(file_path: str) -> dict | None:
    """
    Read one file and return basic statistics without full preprocessing.
    Returns None if the file cannot be read.
    """
    ext = os.path.splitext(file_path)[1].lower()
    delimiter = "\t" if ext == ".txt" else ","

    try:
        df = pd.read_csv(
            file_path,
            sep=delimiter,
            header=None,
            comment="#",
            engine="python",
            on_bad_lines="warn",
            encoding="latin-1",
        )
    except Exception as exc:
        log_error(f"Cannot read {os.path.basename(file_path)}: {exc}")
        return None

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        log_warn(f"No numeric columns in: {os.path.basename(file_path)}")
        return None

    try:
        label = infer_label(file_path)
    except ValueError:
        label = "?"

    return {
        "file":       os.path.basename(file_path),
        "rows":       len(df),
        "cols_total": len(df.columns),
        "cols_num":   len(numeric_df.columns),
        "label":      label,
        "has_nan":    bool(numeric_df.isnull().any().any()),
        "min":        round(float(numeric_df.min().min()), 4),
        "max":        round(float(numeric_df.max().max()), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect raw gait dataset.")
    p.add_argument("--raw_dir", default=config.RAW_DIR)
    args = p.parse_args()

    section("Dataset Inspector")
    log_info(f"Scanning: {args.raw_dir}\n")

    try:
        files = discover_files(args.raw_dir)
    except FileNotFoundError as exc:
        log_error(str(exc))
        sys.exit(1)

    log_info(f"Total files found: {len(files)}\n")

    results   = []
    label_counts = {0: 0, 1: 0, "?": 0}

    for fp in files:
        info = inspect_file(fp)
        if info:
            results.append(info)
            lbl = info["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # ── Summary table ──────────────────────────────────────────────────────
    section("File Summary")
    header = f"  {'File':<40} {'Rows':>6} {'NumCols':>7} {'Label':>6} {'NaN':>5} {'Min':>10} {'Max':>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for r in results:
        label_str = {0: "control", 1: "parkinson"}.get(r["label"], "UNKNOWN")
        nan_str   = "YES" if r["has_nan"] else "no"
        print(
            f"  {r['file']:<40} {r['rows']:>6} {r['cols_num']:>7} "
            f"{label_str:>9} {nan_str:>5} {r['min']:>10} {r['max']:>10}"
        )

    # ── Class balance ──────────────────────────────────────────────────────
    section("Class Balance")
    log_info(f"  Parkinson (label 1) : {label_counts.get(1, 0)} file(s)")
    log_info(f"  Control   (label 0) : {label_counts.get(0, 0)} file(s)")
    if label_counts.get("?", 0):
        log_warn(f"  Unknown label       : {label_counts['?']} file(s)  "
                 "-> Check folder names match config.LABEL_RULES")

    # ── Readiness check ────────────────────────────────────────────────────
    section("Readiness Check")
    n_windows_estimate = sum(
        max(0, (r["rows"] - config.WINDOW_SIZE) // config.STRIDE + 1)
        for r in results
    )
    log_info(f"Estimated windows (window={config.WINDOW_SIZE}, stride={config.STRIDE}): "
             f"~{n_windows_estimate}")

    short = [r["file"] for r in results if r["rows"] < config.WINDOW_SIZE]
    if short:
        log_warn(
            f"{len(short)} file(s) shorter than window_size={config.WINDOW_SIZE} "
            f"— they will be SKIPPED:\n    " + "\n    ".join(short)
        )
    else:
        log_info("All files are long enough to produce at least one window. ✓")

    if label_counts.get(1, 0) == 0 or label_counts.get(0, 0) == 0:
        log_warn(
            "Only one class detected — both 'parkinson/' and 'control/' "
            "subfolders must exist with data."
        )
    else:
        log_info("Both classes detected. ✓")

    print()


if __name__ == "__main__":
    main()