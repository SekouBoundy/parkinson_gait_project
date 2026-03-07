import os

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

FS           = 100
CUTOFF_HZ    = 20
APPLY_FILTER = True
WINDOW_SIZE  = 128
STRIDE       = 64

LABEL_RULES = {
    1: ["parkinson", "parkinsons"],
    0: ["control", "healthy"],
}

SUPPORTED_EXTENSIONS   = {".csv", ".txt"}
UNSUPPORTED_EXTENSIONS = {".dat", ".hea", ".edf", ".mat"}
