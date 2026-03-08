"""
train.py — LSTM training pipeline for Parkinson gait classification.

What this script does
---------------------
  1. Load X.npy / y.npy  produced by make_dataset.py
  2. Stratified 70 / 15 / 15  train / val / test split
  3. Build PyTorch DataLoaders
  4. Handle class imbalance via pos_weight in BCEWithLogitsLoss
  5. Train the LSTMClassifier with early stopping
  6. Evaluate the best checkpoint on the held-out test set
  7. Save: best_model.pt  +  training_results.json

Usage
-----
  python -m src.train
  python -m src.train --epochs 50 --hidden_size 64 --num_layers 1
  python -m src.train --no_early_stop --epochs 30
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.model import LSTMClassifier
from src.utils import section, kv, save_json, ensure_dir


# ─────────────────────────────────────────────────────────────────────────────
# Device selection — auto-detect GPU, fall back to CPU
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return CUDA if available, MPS (Apple Silicon) if available, else CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"  [Device] GPU detected → {name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("  [Device] Apple Silicon MPS detected")
    else:
        dev = torch.device("cpu")
        print("  [Device] No GPU found → using CPU")
    return dev


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM on processed gait data.")

    # Data
    p.add_argument("--processed_dir", default=config.PROCESSED_DIR)
    p.add_argument("--val_size",      type=float, default=0.15)
    p.add_argument("--test_size",     type=float, default=0.15)

    # Model architecture
    p.add_argument("--hidden_size",   type=int,   default=128)
    p.add_argument("--num_layers",    type=int,   default=2)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--bidirectional", action="store_true",
                   help="Use bidirectional LSTM (doubles parameters).")

    # Training
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4,
                   help="L2 regularisation strength.")

    # Early stopping
    p.add_argument("--no_early_stop", action="store_true",
                   help="Disable early stopping and run for all epochs.")
    p.add_argument("--patience",      type=int,   default=8,
                   help="Early stopping: epochs without val-loss improvement.")

    # Misc
    p.add_argument("--seed",          type=int,   default=42)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_arrays(processed_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate X.npy, y.npy, and groups.npy."""
    x_path = os.path.join(processed_dir, "X.npy")
    y_path = os.path.join(processed_dir, "y.npy")
    g_path = os.path.join(processed_dir, "groups.npy")

    for p in [x_path, y_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "  -> Run  python -m src.make_dataset  first."
            )

    # groups.npy may not exist if dataset was built before this update
    if not os.path.exists(g_path):
        raise FileNotFoundError(
            f"Missing: {g_path}\n"
            "  -> Re-run  python -m src.make_dataset  to regenerate with subject IDs.\n"
            "  This file is required for the patient-level split."
        )

    X = np.load(x_path)   # (N, window_size, features)
    y = np.load(y_path)   # (N,)
    g = np.load(g_path)   # (N,)  subject index per window
    return X, y, g


def patient_level_split(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    val_size:  float,
    test_size: float,
    seed:      int,
) -> tuple:
    """
    Split at the PATIENT level so no subject appears in more than one subset.

    Why this matters
    ----------------
    Each patient produces ~188 windows.  A window-level shuffle lets the model
    see patient A in both train and test — it learns that specific person's gait,
    not the general disease pattern.  Patient-level splitting prevents this.

    Strategy
    --------
    1. Collect one label per subject (majority vote — all windows share the same
       label so this is always unambiguous).
    2. Stratified-shuffle-split the *subject list* into train / val / test.
    3. Expand back to window indices.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    unique_subjects = np.unique(g)
    # One label per subject — safe because all windows of a subject share a label
    subject_labels = np.array([y[g == s][0] for s in unique_subjects])

    # Step 1 — carve out test subjects
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tv_subj_idx, test_subj_idx = next(sss1.split(unique_subjects, subject_labels))

    tv_subjects   = unique_subjects[tv_subj_idx]
    test_subjects = unique_subjects[test_subj_idx]
    tv_labels     = subject_labels[tv_subj_idx]

    # Step 2 — split remaining subjects into train / val
    relative_val = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    train_subj_idx, val_subj_idx = next(sss2.split(tv_subjects, tv_labels))

    train_subjects = tv_subjects[train_subj_idx]
    val_subjects   = tv_subjects[val_subj_idx]

    # Step 3 — expand to window indices
    def windows_for(subjects):
        mask = np.isin(g, subjects)
        return X[mask], y[mask]

    X_train, y_train = windows_for(train_subjects)
    X_val,   y_val   = windows_for(val_subjects)
    X_test,  y_test  = windows_for(test_subjects)

    return X_train, X_val, X_test, y_train, y_val, y_test, \
           len(train_subjects), len(val_subjects), len(test_subjects)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Wrap numpy arrays in a TensorDataset and return a DataLoader."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,       # 0 is safest on Windows
        pin_memory=False,
    )


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """
    Compute BCEWithLogitsLoss pos_weight to compensate for class imbalance.

    pos_weight = n_negative / n_positive

    This tells the loss to penalise missing a Parkinson case more than
    missing a healthy case — proportional to how rare positives are.
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    weight = n_neg / max(n_pos, 1)
    print(f"  Class imbalance  → pos_weight = {weight:.3f}  "
          f"(neg={n_neg}, pos={n_pos})")
    return torch.tensor([weight], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)          # (batch,)
        loss   = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping — keeps training stable for RNNs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item() * len(X_batch)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loader(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """
    Run inference on a DataLoader.

    Returns
    -------
    mean_loss : float
    f1_score  : float   (Parkinson class, label = 1)
    """
    model.eval()
    running_loss = 0.0
    all_labels   = []
    all_preds    = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        running_loss += loss.item() * len(X_batch)

        probs  = torch.sigmoid(logits)
        preds  = (probs >= 0.5).long()

        all_labels.extend(y_batch.cpu().long().tolist())
        all_preds.extend(preds.cpu().tolist())

    mean_loss = running_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return mean_loss, f1


# ─────────────────────────────────────────────────────────────────────────────
# Final test-set evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def full_evaluation(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run inference on the test set and return a full metrics dictionary.
    Prints a readable classification report and confusion matrix.
    """
    model.eval()
    all_labels = []
    all_probs  = []

    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(y_batch.long().tolist())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= 0.5).astype(int)

    report = classification_report(
        all_labels, all_preds,
        target_names=["control (0)", "parkinson (1)"],
        output_dict=True,
    )
    cm  = confusion_matrix(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    # ── Pretty print ─────────────────────────────────────────────────────
    print(classification_report(
        all_labels, all_preds,
        target_names=["control (0)", "parkinson (1)"],
    ))
    print(f"  Confusion matrix:")
    print(f"                       Pred control   Pred parkinson")
    print(f"  Actual control          {cm[0,0]:>6}           {cm[0,1]:>6}")
    print(f"  Actual parkinson        {cm[1,0]:>6}           {cm[1,1]:>6}")
    print(f"\n  ROC-AUC         : {auc:.4f}")
    print(f"  F1 (parkinson)  : {f1:.4f}")

    return {
        "f1_parkinson":          round(float(f1),  4),
        "roc_auc":               round(float(auc), 4),
        "classification_report": report,
        "confusion_matrix":      cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()

    # ── 1. Load data ───────────────────────────────────────────────────────
    section("Step 1 — Loading data")
    try:
        X, y, g = load_arrays(args.processed_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    kv("X shape",    X.shape)
    kv("y shape",    y.shape)
    kv("Subjects",   len(np.unique(g)))
    n_features = X.shape[2]

    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        name = "control" if cls == 0 else "parkinson"
        kv(f"  label {cls} ({name})", f"{cnt} windows ({100*cnt/len(y):.1f}%)")

    # ── 2. Split ───────────────────────────────────────────────────────────
    section("Step 2 — Patient-level split  (70 / 15 / 15 of SUBJECTS)")
    print("  NOTE: splitting by subject — no patient appears in two subsets.\n")

    X_train, X_val, X_test, y_train, y_val, y_test, \
        n_train_subj, n_val_subj, n_test_subj = patient_level_split(
            X, y, g,
            val_size  = args.val_size,
            test_size = args.test_size,
            seed      = args.seed,
        )

    kv("Train", f"{X_train.shape[0]} windows  ({n_train_subj} subjects)")
    kv("Val",   f"{X_val.shape[0]} windows  ({n_val_subj} subjects)")
    kv("Test",  f"{X_test.shape[0]} windows  ({n_test_subj} subjects)")

    # ── 3. DataLoaders ─────────────────────────────────────────────────────
    section("Step 3 — Building DataLoaders")
    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   args.batch_size, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  args.batch_size, shuffle=False)
    kv("Batch size", args.batch_size)
    kv("Train batches", len(train_loader))

    # ── 4. Model ───────────────────────────────────────────────────────────
    section("Step 4 — Building model")
    model = LSTMClassifier(
        input_size    = n_features,
        hidden_size   = args.hidden_size,
        num_layers    = args.num_layers,
        dropout       = args.dropout,
        bidirectional = args.bidirectional,
    ).to(device)
    model.summary()

    # ── 5. Loss, optimiser, scheduler ─────────────────────────────────────
    section("Step 5 — Loss / optimiser / scheduler")
    pos_weight = compute_pos_weight(y_train).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Reduce LR if val loss plateaus for 3 epochs
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    kv("Optimizer",    "Adam")
    kv("Learning rate", args.lr)
    kv("Weight decay",  args.weight_decay)
    kv("LR scheduler",  "ReduceLROnPlateau (factor=0.5, patience=3)")
    kv("Early stopping", f"patience={args.patience}" if not args.no_early_stop else "OFF")

    # ── 6. Training loop ───────────────────────────────────────────────────
    section(f"Step 6 — Training  (max {args.epochs} epochs)")

    best_val_loss   = float("inf")
    best_epoch      = 0
    patience_count  = 0
    history: list[dict] = []

    checkpoint_path = os.path.join(args.processed_dir, "best_model.pt")
    ensure_dir(args.processed_dir)

    print(f"\n  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>9}  "
          f"{'Val F1':>7}  {'LR':>9}  {'Note'}")
    print("  " + "─" * 64)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss          = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1    = evaluate_loader(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        note   = ""

        # ── Save best checkpoint ───────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_epoch     = epoch
            patience_count = 0
            note           = "✓ saved"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_f1":      val_f1,
                "args":        vars(args),
            }, checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= args.patience and not args.no_early_stop:
                note = "early stop"

        elapsed = time.time() - t0
        print(f"  {epoch:>5}  {train_loss:>11.4f}  {val_loss:>9.4f}  "
              f"{val_f1:>7.4f}  {lr_now:>9.2e}  {note}")

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "val_f1":     round(val_f1,     6),
            "lr":         lr_now,
        })

        if patience_count >= args.patience and not args.no_early_stop:
            print(f"\n  Early stopping triggered at epoch {epoch}. "
                  f"Best epoch was {best_epoch}.")
            break

    # ── 7. Load best checkpoint and evaluate on test set ──────────────────
    section(f"Step 7 — Test evaluation  (best checkpoint: epoch {best_epoch})")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics = full_evaluation(model, test_loader, device)

    # ── 8. Save results ────────────────────────────────────────────────────
    section("Step 8 — Saving results")
    results = {
        "model": {
            "input_size":    n_features,
            "hidden_size":   args.hidden_size,
            "num_layers":    args.num_layers,
            "dropout":       args.dropout,
            "bidirectional": args.bidirectional,
            "trainable_params": model.count_parameters(),
        },
        "training": {
            "epochs_run":   len(history),
            "best_epoch":   best_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "batch_size":   args.batch_size,
            "lr":           args.lr,
            "weight_decay": args.weight_decay,
        },
        "split": {
            "method":        "patient-level (no subject leakage)",
            "train_windows": int(X_train.shape[0]),
            "val_windows":   int(X_val.shape[0]),
            "test_windows":  int(X_test.shape[0]),
            "train_subjects": n_train_subj,
            "val_subjects":   n_val_subj,
            "test_subjects":  n_test_subj,
        },
        "test_metrics": test_metrics,
        "history":      history,
    }

    results_path = os.path.join(args.processed_dir, "training_results.json")
    save_json(results, results_path)

    # ── 9. Final summary ───────────────────────────────────────────────────
    section("Done ✓")
    kv("Best epoch",          best_epoch)
    kv("Best val loss",       f"{best_val_loss:.4f}")
    kv("Test F1 (parkinson)", f"{test_metrics['f1_parkinson']:.4f}")
    kv("Test ROC-AUC",        f"{test_metrics['roc_auc']:.4f}")
    kv("Model saved",         checkpoint_path)
    kv("Results saved",       results_path)

    # ── Honest interpretation ──────────────────────────────────────────────
    f1 = test_metrics["f1_parkinson"]
    if f1 >= 0.90:
        verdict = "Excellent. The LSTM has learned strong gait patterns."
    elif f1 >= 0.75:
        verdict = "Good. Consider tuning hidden_size, layers, or dropout next."
    elif f1 >= 0.60:
        verdict = "Moderate. Try more epochs, bidirectional LSTM, or CNN+LSTM."
    else:
        verdict = "Weak. Check the data, class balance, and preprocessing first."

    print(f"\n  Interpretation: {verdict}")
    print()


if __name__ == "__main__":
    main()