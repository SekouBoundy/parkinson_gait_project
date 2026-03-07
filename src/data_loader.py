"""
data_loader.py — File discovery, signal loading, and label inference.

Responsibilities
----------------
- Recursively find supported gait data files under raw_dir
- Load each file as a (time_steps, features) NumPy array
- Infer the binary label (0 / 1) from the file path
"""

import os
import numpy as np
import pandas as pd

from src import config


# ──────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────

def discover_files(raw_dir: str = config.RAW_DIR) -> list[str]:
    """
    Recursively scan *raw_dir* and return the paths of every supported file.

    Supported extensions are defined in config.SUPPORTED_EXTENSIONS.
    Files with unsupported extensions are logged as warnings and skipped.

    Parameters
    ----------
    raw_dir : str
        Root directory to search.

    Returns
    -------
    list[str]
        Sorted list of absolute file paths.

    Raises
    ------
    FileNotFoundError
        If *raw_dir* does not exist or contains no supported files.
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Create the folder and place your gait data files inside it."
        )

    found: list[str] = []
    skipped: list[str] = []

    for root, _, files in os.walk(raw_dir):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            full = os.path.join(root, fname)
            if ext in config.SUPPORTED_EXTENSIONS:
                found.append(full)
            elif ext in config.UNSUPPORTED_EXTENSIONS:
                skipped.append(full)

    if skipped:
        print(
            f"  [WARN] Skipped {len(skipped)} unsupported file(s) "
            f"({', '.join(config.UNSUPPORTED_EXTENSIONS)}). "
            "Only CSV/TXT files are currently supported."
        )

    if not found:
        raise FileNotFoundError(
            f"No supported files found in: {raw_dir}\n"
            f"Accepted extensions: {config.SUPPORTED_EXTENSIONS}\n"
            "Make sure you have placed your dataset files in the correct folder."
        )

    return found


# ──────────────────────────────────────────────
# Signal loading
# ──────────────────────────────────────────────

def load_signal(file_path: str) -> tuple[np.ndarray, dict]:
    """
    Load a gait data file and return a (time_steps, features) array.

    The loader:
    - Accepts CSV and whitespace-delimited TXT files.
    - Keeps only numeric columns (non-numeric columns are silently dropped).
    - If SELECTED_COLUMNS is set in config, restricts to those columns.
    - Raises clear errors for empty files or files with no numeric data.

    Parameters
    ----------
    file_path : str
        Path to the gait data file.

    Returns
    -------
    signal : np.ndarray, shape (time_steps, features)
    meta   : dict with keys:
               file_path, num_rows, num_features, feature_names

    Raises
    ------
    NotImplementedError
        If the file extension is not supported.
    ValueError
        If the file contains no numeric columns, or is empty.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in config.SUPPORTED_EXTENSIONS:
        raise NotImplementedError(
            f"Unsupported file type '{ext}' for: {file_path}\n"
            f"Supported types: {config.SUPPORTED_EXTENSIONS}"
        )

    # ── Load raw file ──
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:  # .txt — try tab/space separation
            df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    except Exception as exc:
        raise ValueError(
            f"Could not parse file: {file_path}\nOriginal error: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(f"File is empty: {file_path}")

    # ── Keep only numeric columns ──
    df_numeric = df.select_dtypes(include=[np.number])

    if df_numeric.empty:
        raise ValueError(
            f"No numeric columns found in: {file_path}\n"
            f"Columns present: {list(df.columns)}"
        )

    # ── Optional column selection ──
    if config.SELECTED_COLUMNS is not None:
        missing = [c for c in config.SELECTED_COLUMNS if c not in df_numeric.columns]
        if missing:
            raise ValueError(
                f"Selected columns {missing} not found in: {file_path}\n"
                f"Available numeric columns: {list(df_numeric.columns)}"
            )
        df_numeric = df_numeric[config.SELECTED_COLUMNS]

    signal = df_numeric.to_numpy(dtype=np.float32)  # (time_steps, features)

    meta = {
        "file_path":    file_path,
        "num_rows":     signal.shape[0],
        "num_features": signal.shape[1],
        "feature_names": list(df_numeric.columns),
    }

    return signal, meta


# ──────────────────────────────────────────────
# Label inference
# ──────────────────────────────────────────────

def infer_label(file_path: str, raw_dir: str = config.RAW_DIR) -> int:
    """
    Infer the binary classification label from the file path.

    The check is performed only on the path **relative to raw_dir** so that
    parent directory names (e.g. the project folder itself) cannot accidentally
    trigger a rule.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the data file.
    raw_dir : str
        Root data directory; used to compute the relative sub-path.

    Returns
    -------
    int
        0 for healthy/control, 1 for Parkinson's patient.

    Raises
    ------
    ValueError
        If no rule matches the path.
    """
    # Use only the portion of the path below raw_dir for matching.
    abs_file = os.path.abspath(file_path)
    abs_raw  = os.path.abspath(raw_dir)

    try:
        rel_path = os.path.relpath(abs_file, abs_raw)
    except ValueError:
        # On Windows, relpath can fail across drives — fall back to full path
        rel_path = file_path

    search_str = rel_path.lower().replace("\\", "/")

    for label, keywords in config.LABEL_RULES.items():
        for kw in keywords:
            if kw.lower() in search_str:
                return label

    all_keywords = [kw for kws in config.LABEL_RULES.values() for kw in kws]
    raise ValueError(
        f"Cannot infer label from path: {file_path}\n"
        f"Checked relative path: '{rel_path}'\n"
        f"Expected one of these keywords (case-insensitive): {all_keywords}\n"
        "Rename the file or its parent folder, or update LABEL_RULES in config.py."
    )