"""
Recursively scans RAW_DIR, profiles every file, saves inspect_report.json.
Usage:
    python -m src.inspect_dataset
    python -m src.inspect_dataset --raw_dir data/raw
"""
import os, sys, json, logging, argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _infer_label(file_path: Path):
    candidate_parts = [p.lower() for p in file_path.parts[-3:-1]]
    for label, keywords in config.LABEL_RULES.items():
        for kw in keywords:
            for part in candidate_parts:
                if kw == part or kw in part:
                    return label
    return None


def _load_file(file_path: Path):
    ext = file_path.suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        if ext in config.UNSUPPORTED_EXTENSIONS:
            return None, f"Unsupported binary/WFDB format '{ext}'"
        return None, f"Unknown extension '{ext}' – skipping"

    for sep in [",", "\t", " ", ";"]:
        try:
            df = pd.read_csv(file_path, sep=sep, header=None, comment="#")
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                df = pd.read_csv(file_path, sep=sep, comment="#")
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                arr = df[num_cols].to_numpy(dtype=np.float32)
                if arr.shape[0] < 2:
                    return None, "File has fewer than 2 rows"
                return arr, None
        except Exception:
            continue
    return None, "Could not parse file with any separator"


def inspect(raw_dir: str) -> dict:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        log.error("RAW_DIR does not exist: %s", raw_path); sys.exit(1)

    all_files = sorted([f for f in raw_path.rglob("*") if f.is_file()])
    if not all_files:
        log.warning("No files found in %s", raw_path)

    ext_counts, loaded_shapes = defaultdict(int), []
    missing_values, load_errors = 0, []
    label_counts = defaultdict(int)
    first_example = None

    for fp in all_files:
        ext_counts[fp.suffix.lower()] += 1

    log.info("Found %d file(s) | Extensions: %s", len(all_files), dict(ext_counts))

    for fp in all_files:
        arr, err = _load_file(fp)
        if err:
            log.warning("  x %s -> %s", fp.name, err)
            load_errors.append({"file": str(fp), "reason": err}); continue

        T, C = arr.shape
        nan_count = int(np.isnan(arr).sum())
        missing_values += nan_count
        loaded_shapes.append((T, C))
        label = _infer_label(fp)
        label_counts[str(label)] += 1
        log.info("  OK %-40s shape=(%d,%d) NaN=%d label=%s", fp.name, T, C, nan_count, label)

        if first_example is None:
            first_example = {"file": str(fp), "shape": [T, C],
                             "first_3_rows": arr[:3].tolist(), "label": label}

    lengths  = [s[0] for s in loaded_shapes]
    channels = [s[1] for s in loaded_shapes]
    stats = {}
    if lengths:
        stats = {
            "length_T": {"min": int(np.min(lengths)), "max": int(np.max(lengths)),
                         "mean": float(np.mean(lengths))},
            "channels_C": {"unique_values": sorted(set(channels))},
        }

    report = {
        "raw_dir": str(raw_path.resolve()), "total_files": len(all_files),
        "extension_counts": dict(ext_counts), "loaded_ok": len(loaded_shapes),
        "load_errors": load_errors, "missing_values_total": missing_values,
        "label_counts": dict(label_counts), "signal_stats": stats,
        "first_example": first_example,
    }

    print("\n" + "="*60)
    print("  DATASET INSPECTION REPORT")
    print("="*60)
    print(f"  Total files       : {len(all_files)}")
    print(f"  Extensions        : {dict(ext_counts)}")
    print(f"  Loaded OK         : {len(loaded_shapes)}")
    print(f"  Failed            : {len(load_errors)}")
    print(f"  Missing values    : {missing_values}")
    print(f"  Label distribution: {dict(label_counts)}")
    if stats:
        lt = stats["length_T"]
        print(f"  Signal length (T) : min={lt['min']}  max={lt['max']}  mean={lt['mean']:.1f}")
        print(f"  Channels (C)      : {stats['channels_C']['unique_values']}")
    if first_example:
        print(f"\n  Example: {Path(first_example['file']).name}  shape={first_example['shape']}")
        print(f"  First 3 rows:\n{np.array(first_example['first_3_rows'])}")
    print("="*60 + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",  default=config.RAW_DIR)
    parser.add_argument("--out_dir",  default=config.PROCESSED_DIR)
    args = parser.parse_args()
    report = inspect(args.raw_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "inspect_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved -> %s", out)

if __name__ == "__main__":
    main()
