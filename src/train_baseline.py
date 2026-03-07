"""
train_baseline.py — Baseline ML classifier for Parkinson gait windows.

What this does:
  1. Load X.npy / y.npy produced by preprocess_and_window
  2. Flatten each window (N, 128, 19) → (N, 128*19)
  3. Stratified 70/15/15 train/val/test split
  4. Handle class imbalance via class_weight='balanced' (no data thrown away)
  5. Train Random Forest + Logistic Regression
  6. Evaluate on held-out test set with full classification report
  7. Save results to data/processed/baseline_results.json

Why two models?
  - Logistic Regression: fast sanity check — if features carry no signal it'll fail
  - Random Forest:       strong non-linear baseline — sets the bar for the LSTM

Usage:
    python -m src.train_baseline
    python -m src.train_baseline --n_estimators 200
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble          import RandomForestClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import StratifiedShuffleSplit
from sklearn.metrics           import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.preprocessing     import StandardScaler

from src import config
from src.utils import section, log_info, log_warn, save_json


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ML baseline models.")
    p.add_argument("--processed_dir",  default=config.PROCESSED_DIR)
    p.add_argument("--n_estimators",   type=int, default=100,
                   help="Number of trees in Random Forest (default 100)")
    p.add_argument("--val_size",       type=float, default=0.15)
    p.add_argument("--test_size",      type=float, default=0.15)
    p.add_argument("--random_state",   type=int, default=42)
    p.add_argument("--max_features",   type=int, default=None,
                   help="Optional: limit features passed to models (for speed)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & splitting
# ─────────────────────────────────────────────────────────────────────────────

def load_arrays(processed_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load X and y from disk, validate shapes."""
    x_path = os.path.join(processed_dir, "X.npy")
    y_path = os.path.join(processed_dir, "y.npy")

    for p in [x_path, y_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "  -> Run python -m src.preprocess_and_window first."
            )

    X = np.load(x_path)
    y = np.load(y_path)

    if X.ndim != 3:
        raise ValueError(f"Expected X to be 3-D, got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected y to be 1-D, got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have different number of samples.")

    return X, y


def flatten(X: np.ndarray) -> np.ndarray:
    """
    Collapse the time and feature axes into one flat vector per window.
    (N, window_size, features)  →  (N, window_size * features)
    """
    N = X.shape[0]
    return X.reshape(N, -1)


def split_data(
    X_flat: np.ndarray,
    y: np.ndarray,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple:
    """
    Stratified split preserving class ratios in every subset.

    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First cut: separate test set
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(sss_test.split(X_flat, y))

    X_tv, y_tv = X_flat[train_val_idx], y[train_val_idx]
    X_test, y_test = X_flat[test_idx], y[test_idx]

    # Second cut: separate val from remaining train
    relative_val = val_size / (1.0 - test_size)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_val, random_state=random_state
    )
    train_idx, val_idx = next(sss_val.split(X_tv, y_tv))

    X_train, y_train = X_tv[train_idx], y_tv[train_idx]
    X_val,   y_val   = X_tv[val_idx],   y_tv[val_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    X_val:  np.ndarray,
    y_val:  np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """
    Print and return metrics for val and test sets.
    Reports per-class precision / recall / F1 + ROC-AUC.
    """
    section(f"Results — {model_name}")

    results = {}
    for split_name, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(Xs)
        y_prob = (
            model.predict_proba(Xs)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        report = classification_report(
            ys, y_pred,
            target_names=["control (0)", "parkinson (1)"],
            output_dict=True,
        )
        cm  = confusion_matrix(ys, y_pred)
        f1  = f1_score(ys, y_pred)
        auc = roc_auc_score(ys, y_prob) if y_prob is not None else None

        # ── Pretty print ─────────────────────────────────────────────────
        print(f"\n  [{split_name.upper()}]")
        print(classification_report(
            ys, y_pred,
            target_names=["control (0)", "parkinson (1)"],
        ))
        print(f"  Confusion matrix:\n"
              f"                    Pred control  Pred parkinson\n"
              f"  Actual control       {cm[0,0]:>6}         {cm[0,1]:>6}\n"
              f"  Actual parkinson     {cm[1,0]:>6}         {cm[1,1]:>6}\n")
        if auc is not None:
            print(f"  ROC-AUC : {auc:.4f}")

        results[split_name] = {
            "f1_parkinson":       round(f1, 4),
            "roc_auc":            round(auc, 4) if auc else None,
            "classification_report": report,
            "confusion_matrix":   cm.tolist(),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── 1. Load data ───────────────────────────────────────────────────────
    section("Baseline ML — Loading data")
    try:
        X, y = load_arrays(args.processed_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}"); sys.exit(1)

    log_info(f"X shape   : {X.shape}  (N, window_size, features)")
    log_info(f"y shape   : {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        name = "control" if cls == 0 else "parkinson"
        pct  = 100 * cnt / len(y)
        log_info(f"  label {cls} ({name:9s}) : {cnt:6d} windows  ({pct:.1f}%)")

    # ── 2. Flatten ─────────────────────────────────────────────────────────
    section("Step 1 · Flattening windows")
    X_flat = flatten(X)
    log_info(f"Flattened shape : {X_flat.shape}  "
             f"({X.shape[1]} timesteps × {X.shape[2]} features)")

    # Optional feature cap for speed (useful on slow machines)
    if args.max_features and args.max_features < X_flat.shape[1]:
        X_flat = X_flat[:, :args.max_features]
        log_warn(f"Feature cap applied — using first {args.max_features} features.")

    # ── 3. Split ───────────────────────────────────────────────────────────
    section("Step 2 · Stratified train / val / test split")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_flat, y,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    log_info(f"Train : {X_train.shape[0]} windows")
    log_info(f"Val   : {X_val.shape[0]} windows")
    log_info(f"Test  : {X_test.shape[0]} windows")

    # Verify stratification held
    for split_name, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
        u, c = np.unique(ys, return_counts=True)
        dist  = {int(k): f"{100*v/len(ys):.1f}%" for k, v in zip(u, c)}
        log_info(f"  {split_name} class dist : {dist}")

    # ── 4. Scale for Logistic Regression ──────────────────────────────────
    section("Step 3 · Scaling features (for Logistic Regression)")
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    log_info("StandardScaler fitted on train set only. ✓")

    # ── 5. Train models ────────────────────────────────────────────────────
    section("Step 4 · Training models")

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=args.random_state,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=args.n_estimators,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
            max_depth=20,
        ),
    }

    all_results = {}

    for name, model in models.items():
        log_info(f"Training {name}...")
        # LR uses scaled data; RF is scale-invariant
        Xtr = X_train_sc if "Logistic" in name else X_train
        Xv  = X_val_sc   if "Logistic" in name else X_val
        Xt  = X_test_sc  if "Logistic" in name else X_test

        model.fit(Xtr, y_train)
        log_info(f"{name} trained. ✓")

        result = evaluate(model, Xv, y_val, Xt, y_test, name)
        all_results[name] = result

    # ── 6. Head-to-head summary ────────────────────────────────────────────
    section("Head-to-Head Summary (Test Set)")
    print(f"  {'Model':<25} {'F1-Parkinson':>13} {'ROC-AUC':>9}")
    print("  " + "─" * 50)
    for name, res in all_results.items():
        f1  = res["test"]["f1_parkinson"]
        auc = res["test"]["roc_auc"] or "N/A"
        print(f"  {name:<25} {f1:>13.4f} {str(auc):>9}")

    # ── 7. Save results ────────────────────────────────────────────────────
    section("Saving results")
    out_path = os.path.join(args.processed_dir, "baseline_results.json")

    save_json({
        "split": {
            "train": int(X_train.shape[0]),
            "val":   int(X_val.shape[0]),
            "test":  int(X_test.shape[0]),
            "val_size":  args.val_size,
            "test_size": args.test_size,
        },
        "class_weight": "balanced",
        "models": all_results,
    }, out_path)

    log_info(f"Results saved → {out_path}")

    # ── 8. Honest interpretation ───────────────────────────────────────────
    section("What these numbers mean")
    best_f1 = max(r["test"]["f1_parkinson"] for r in all_results.values())

    if best_f1 >= 0.90:
        verdict = "Excellent signal in the features. LSTM should do even better."
    elif best_f1 >= 0.75:
        verdict = "Good signal. Deep learning will likely improve this further."
    elif best_f1 >= 0.60:
        verdict = "Moderate signal. Feature engineering or more data may help."
    else:
        verdict = ("Weak signal on flat features. Check preprocessing — "
                   "raw features may need domain-specific engineering.")

    log_info(f"Best test F1 (parkinson): {best_f1:.4f}")
    log_info(f"Interpretation: {verdict}")
    log_info("")
    log_info("NOTE: F1-parkinson is the key metric here, NOT accuracy.")
    log_info("      A model predicting 'parkinson' for everything would")
    log_info("      get high accuracy but zero recall on control class.")
    print()


if __name__ == "__main__":
    main()