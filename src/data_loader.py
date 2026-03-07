"""
data_loader.py — File discovery, signal loading, and label inference.

Supports .csv and .txt files (tab/comma separated) as defined in config.
"""

import os
import numpy as np
import pandas as pd

from src import config
from src.utils import log_warn


# ──────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────

def discover_files(raw_dir: str) -> list[str]:
    """
    Recursively find all supported gait data files under *raw_dir*.

    Supported extensions are defined in config.SUPPORTED_EXTENSIONS.
    Unsupported extensions (e.g. .dat, .hea) are logged as warnings.

    Returns
    -------
    list[str]
        Sorted list of absolute file paths.

    Raises
    ------
    FileNotFoundError
        If *raw_dir* does not exist or contains no supported files.
    """
    found = [f for f in found if os.path.basename(f) not in config.EXCLUDED_FILENAMES]

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Raw data directory not found: '{raw_dir}'\n"
            "  -> Create it and place your dataset files inside."
        )

    found, skipped = [], []
    for root, _, files in os.walk(raw_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in config.SUPPORTED_EXTENSIONS:
                found.append(os.path.join(root, fname))
            elif ext in config.UNSUPPORTED_EXTENSIONS:
                skipped.append(fname)

    if skipped:
        log_warn(
            f"{len(skipped)} unsupported file(s) ignored "
            f"({', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''})"
        )

    if not found:
        raise FileNotFoundError(
            f"No supported files found in '{raw_dir}'.\n"
            f"  Supported extensions: {config.SUPPORTED_EXTENSIONS}\n"
            "  -> Download the PhysioNet dataset and place .txt/.csv files "
            "inside data/raw/parkinson/ and data/raw/control/."
        )

    return sorted(found)


# ──────────────────────────────────────────────
# Signal loading
# ──────────────────────────────────────────────

def load_signal(file_path: str) -> tuple[np.ndarray, dict]:
    """
    Load a single gait data file and return its numeric signal.

    The function auto-detects the delimiter (comma or tab).
    Only numeric columns are kept; non-numeric columns are dropped silently.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to a .csv or .txt file.

    Returns
    -------
    signal : np.ndarray
        Shape (time_steps, features). Always 2-D — even for a single column.
    meta : dict
        Keys: file_path, num_rows, num_features, feature_names.

    Raises
    ------
    NotImplementedError
        If the file extension is not .csv or .txt.
    ValueError
        If the file contains no numeric columns, or is completely empty.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise NotImplementedError(
            f"Unsupported file type '{ext}' for file: {file_path}\n"
            f"  Supported: {config.SUPPORTED_EXTENSIONS}"
        )

    # ── Auto-detect delimiter ──────────────────
    delimiter = "\t" if ext == ".txt" else ","

    try:
        df = pd.read_csv(
            file_path,
            sep=delimiter,
            header=None,          # PhysioNet files have no header row
            comment="#",          # skip comment lines if present
            engine="python",
            on_bad_lines="warn",
            encoding="latin-1",   # handles Windows-1252 / ISO-8859-1 bytes
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to read file: {file_path}\n  Reason: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(f"File is empty: {file_path}")

    # ── Keep numeric columns only ──────────────
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError(
            f"No numeric columns found in: {file_path}\n"
            "  -> Check the file format; it may be malformed or non-tabular."
        )

    signal = numeric_df.to_numpy(dtype=np.float32)   # (time_steps, features)
    feature_names = [str(c) for c in numeric_df.columns.tolist()]

    meta = {
        "file_path":    file_path,
        "num_rows":     signal.shape[0],
        "num_features": signal.shape[1],
        "feature_names": feature_names,
    }

    return signal, meta


# ──────────────────────────────────────────────
# Label inference
# ──────────────────────────────────────────────

def infer_label(file_path: str) -> int:
    """
    Derive a binary label (0 or 1) from the file path using config rules.

    The full path is checked case-insensitively against each keyword list
    defined in config.LABEL_RULES.

    Parameters
    ----------
    file_path : str
        Path whose directory names or file name encode the class.

    Returns
    -------
    int
        1 for Parkinson patient, 0 for healthy control.

    Raises
    ------
    ValueError
        If no rule in config.LABEL_RULES matches the path.
    """
    # Only check the immediate parent folder name + filename
    # Checking the full path is unsafe — project/repo names may contain keywords.
    fname      = os.path.basename(file_path)
    parent     = os.path.basename(os.path.dirname(file_path))
    path_lower = (parent + "/" + fname).lower()

    for label, keywords in config.LABEL_RULES.items():
        if any(kw in path_lower for kw in keywords):
            return label

    raise ValueError(
        f"Cannot infer label for: {file_path}\n"
        "  -> The path must contain one of the keywords defined in "
        "config.LABEL_RULES.\n"
        f"  Current rules: {config.LABEL_RULES}\n"
        "  Example layout:  data/raw/parkinson/subject_01.txt\n"
        "                   data/raw/control/subject_02.txt"
    )