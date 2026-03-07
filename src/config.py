"""
config.py — Central configuration for the Parkinson gait pipeline.

All other modules import from here.  Change values here rather than
hunting through the codebase.
"""

import os

# ──────────────────────────────────────────────
# Directory layout
# ──────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# ──────────────────────────────────────────────
# Signal processing
# ──────────────────────────────────────────────

FS           = 100      # sampling frequency in Hz
CUTOFF_HZ    = 20       # low-pass filter cut-off frequency
APPLY_FILTER = True     # set False to skip filtering (e.g., --no_filter flag)

# ──────────────────────────────────────────────
# Windowing
# ──────────────────────────────────────────────

WINDOW_SIZE = 128   # number of time-steps per window
STRIDE      = 64    # step between consecutive windows  (50 % overlap)

# ──────────────────────────────────────────────
# Label inference rules  (case-insensitive substring match)
# ──────────────────────────────────────────────
#   key   = integer label (0 = healthy, 1 = Parkinson)
#   value = list of substrings that trigger that label

LABEL_RULES: dict[int, list[str]] = {
    1: ["parkinson", "parkinsons", "pd", "gapt", "sipt"],
    0: ["control", "healthy", "gaco"],
}

# ──────────────────────────────────────────────
# Feature column selection
# ──────────────────────────────────────────────
#   None  -> use ALL numeric columns found in the file
#   list  -> restrict to these column names (e.g. ["VGRF_L", "VGRF_R"])

SELECTED_COLUMNS: list[str] | None = None

# ──────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────

SUPPORTED_EXTENSIONS   = {".csv", ".txt"}
UNSUPPORTED_EXTENSIONS = {".dat", ".hea", ".edf", ".mat"}