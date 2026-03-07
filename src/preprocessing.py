"""
preprocessing.py — Signal conditioning and windowing.

Three independent, composable transforms:
  1. lowpass_filter   — remove high-frequency noise
  2. standardize      — zero-mean / unit-variance per feature column
  3. sliding_windows  — split a long signal into fixed-length windows
"""

import numpy as np
from scipy.signal import butter, filtfilt


# ──────────────────────────────────────────────
# 1. Low-pass filter
# ──────────────────────────────────────────────

def lowpass_filter(
    signal: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth low-pass filter to every feature column.

    Parameters
    ----------
    signal     : np.ndarray, shape (time_steps, features)
    fs         : float  — sampling frequency in Hz
    cutoff_hz  : float  — cut-off frequency in Hz
    order      : int    — filter order (default 4)

    Returns
    -------
    np.ndarray, same shape as *signal*

    Raises
    ------
    ValueError
        If cutoff_hz >= fs/2 (Nyquist limit violation).
    """
    nyquist = fs / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"cutoff_hz ({cutoff_hz} Hz) must be less than the Nyquist frequency "
            f"({nyquist} Hz).  Lower cutoff_hz or raise fs in config.py."
        )

    b, a = butter(order, cutoff_hz / nyquist, btype="low", analog=False)

    filtered = np.empty_like(signal)
    for col in range(signal.shape[1]):
        filtered[:, col] = filtfilt(b, a, signal[:, col])

    return filtered.astype(np.float32)


# ──────────────────────────────────────────────
# 2. Standardisation (z-score, per column)
# ──────────────────────────────────────────────

def standardize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization independently to each feature column.

    Columns with zero variance are left unchanged (all zeros after mean
    subtraction) to avoid division-by-zero.

    Parameters
    ----------
    signal : np.ndarray, shape (time_steps, features)

    Returns
    -------
    np.ndarray, same shape as *signal*
    """
    mean = signal.mean(axis=0)          # shape: (features,)
    std  = signal.std(axis=0)           # shape: (features,)

    # Avoid division by zero for constant-value columns
    std_safe = np.where(std == 0.0, 1.0, std)

    return ((signal - mean) / std_safe).astype(np.float32)


# ──────────────────────────────────────────────
# 3. Sliding-window segmentation
# ──────────────────────────────────────────────

def sliding_windows(
    signal: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """
    Segment a signal into overlapping fixed-length windows.

    Parameters
    ----------
    signal      : np.ndarray, shape (time_steps, features)
    window_size : int — number of time-steps per window
    stride      : int — step between consecutive window starts

    Returns
    -------
    np.ndarray, shape (n_windows, window_size, features)

    If the signal is shorter than *window_size*, returns an empty array
    with shape (0, window_size, features) — the caller is responsible
    for deciding whether to skip or raise an error.
    """
    time_steps, features = signal.shape

    if time_steps < window_size:
        return np.empty((0, window_size, features), dtype=np.float32)

    # Calculate number of complete windows
    n_windows = 1 + (time_steps - window_size) // stride

    windows = np.empty((n_windows, window_size, features), dtype=np.float32)
    for i in range(n_windows):
        start = i * stride
        windows[i] = signal[start : start + window_size]

    return windows