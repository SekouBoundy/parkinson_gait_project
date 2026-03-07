"""
preprocessing.py — Signal filtering, normalization, and windowing.

All functions operate on 2-D arrays of shape (time_steps, features)
and preserve that shape unless noted otherwise.
"""

import numpy as np
from scipy.signal import butter, filtfilt


# ──────────────────────────────────────────────
# Standardization
# ──────────────────────────────────────────────

def standardize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization independently to each feature column.

    Zero-variance columns are left unchanged (all-zero after mean removal)
    to avoid division-by-zero errors.

    Parameters
    ----------
    signal : np.ndarray
        Shape (time_steps, features).

    Returns
    -------
    np.ndarray
        Same shape as input, float32.
    """
    signal = signal.astype(np.float32)
    mean = signal.mean(axis=0)
    std  = signal.std(axis=0)

    # Avoid division by zero for constant columns
    std_safe = np.where(std == 0, 1.0, std)

    return (signal - mean) / std_safe


# ──────────────────────────────────────────────
# Low-pass filter
# ──────────────────────────────────────────────

def lowpass_filter(signal: np.ndarray,
                   fs: int,
                   cutoff_hz: float,
                   order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter to each feature column independently.

    Uses zero-phase filtering (filtfilt) so no phase distortion is introduced.

    Parameters
    ----------
    signal : np.ndarray
        Shape (time_steps, features).
    fs : int
        Sampling frequency in Hz.
    cutoff_hz : float
        Cut-off frequency in Hz. Must be less than fs / 2.
    order : int
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Same shape as input, float32.

    Raises
    ------
    ValueError
        If cutoff_hz >= fs / 2 (Nyquist limit violated).
    """
    nyquist = fs / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"cutoff_hz ({cutoff_hz} Hz) must be less than the Nyquist "
            f"frequency ({nyquist} Hz).  Lower it in config.py."
        )

    # Minimum signal length for filtfilt: 3 * (order + 1) samples per edge
    min_len = 3 * (order + 1)
    if signal.shape[0] < min_len * 2:
        # Signal too short to filter safely — return as-is
        return signal.astype(np.float32)

    b, a = butter(order, cutoff_hz / nyquist, btype="low", analog=False)
    filtered = np.apply_along_axis(
        lambda col: filtfilt(b, a, col),
        axis=0,
        arr=signal,
    )
    return filtered.astype(np.float32)


# ──────────────────────────────────────────────
# Sliding window segmentation
# ──────────────────────────────────────────────

def sliding_windows(signal: np.ndarray,
                    window_size: int,
                    stride: int) -> np.ndarray:
    """
    Segment a signal into overlapping fixed-length windows.

    Parameters
    ----------
    signal : np.ndarray
        Shape (time_steps, features).
    window_size : int
        Number of time steps per window.
    stride : int
        Step size between consecutive windows.

    Returns
    -------
    np.ndarray
        Shape (n_windows, window_size, features).
        Returns shape (0, window_size, features) if the signal is
        shorter than *window_size*.
    """
    time_steps, features = signal.shape

    if time_steps < window_size:
        return np.empty((0, window_size, features), dtype=np.float32)

    starts  = range(0, time_steps - window_size + 1, stride)
    windows = np.stack([signal[s: s + window_size] for s in starts], axis=0)
    return windows.astype(np.float32)