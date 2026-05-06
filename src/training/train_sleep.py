"""
train_sleep.py
==============
Training Pipeline for DenseSleepNet (Sleep Stage Classification)

What this script does:
  1. Loads preprocessed EEG epochs from data/processed/sleep_X.npy
  2. Creates sliding-window sequences of epochs (context window)
  3. Splits data into train / validation / test sets
  4. Trains DenseSleepNet with early stopping
  5. Evaluates on the test set and prints a classification report
  6. Saves the best model to models/saved/dense_sleep_net.pt

Usage:
  python src/training/train_sleep.py
  python src/training/train_sleep.py --epochs 100 --batch_size 16
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import yaml
import time

# Add project root to path so we can import our modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.dense_sleep_net import build_sleep_model


# ─── Config ───────────────────────────────────────────────────────────────────
def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class SleepSequenceDataset(Dataset):
    """
    Creates overlapping sequences of sleep epochs for the Transformer.

    Instead of classifying epochs one-by-one, we feed SEQ_LEN consecutive
    epochs as context. The model predicts a label for EVERY epoch in the
    sequence (sequence-to-sequence classification).

    Example with seq_len=5:
      Epochs:  [e0, e1, e2, e3, e4, e5, e6, ...]
      Sample0: [e0, e1, e2, e3, e4] → labels [y0, y1, y2, y3, y4]
      Sample1: [e1, e2, e3, e4, e5] → labels [y1, y2, y3, y4, y5]
      (stride=1 for maximum data, stride=seq_len for non-overlapping)
    """

    def __init__(self, X, y, seq_len=21, stride=1):
        """
        Args:
            X:       (n_epochs, n_channels, epoch_samples) float32
            y:       (n_epochs,) int64 labels
            seq_len: number of consecutive epochs per sample
            stride:  step between sequence starts
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        self.stride  = stride

        # Compute valid start indices
        n_epochs = len(X)
        self.indices = list(range(0, n_epochs - seq_len + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end   = start + self.seq_len
        # x: (seq_len, n_channels, epoch_samples)
        # y: (seq_len,)
        return self.X[start:end], self.y[start:end]


# ─── Training Utilities ────────────────────────────────────────────────────────
class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def compute_class_weights(y, n_classes=5):
    """
    Compute inverse-frequency class weights to handle class imbalance.

    Sleep datasets are heavily imbalanced (lots of N2, little REM/N1).
    Weighting rare classes higher prevents the model from ignoring them.
    """
    counts  = np.bincount(y, minlength=n_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes   # normalise
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one pass over the training data."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)   # (B, seq_len, C, T)
        y_batch = y_batch.to(device)   # (B, seq_len)

        optimizer.zero_grad()

        logits = model(X_batch)        # (B, seq_len, n_classes)

        # Flatten seq dimension for loss computation
        B, S, C = logits.shape
        loss = criterion(logits.view(B * S, C), y_batch.view(B * S))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        preds = logits.argmax(dim=-1)                            # (B, S)
        total_correct += (preds == y_batch).sum().item()
        total_samples += B * S
        total_loss    += loss.item() * B

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        B, S, C = logits.shape
        loss = criterion(logits.view(B * S, C), y_batch.view(B * S))

        preds = logits.argmax(dim=-1).view(-1).cpu().numpy()
        labels = y_batch.view(-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)
        total_loss += loss.item() * B

    avg_loss = total_loss / len(loader)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ─── Main Training Function ────────────────────────────────────────────────────
def train(args):
    config = load_config(args.config)
    train_cfg = config["training"]

    # Reproducibility
    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    processed = Path(config["paths"]["processed"])
    X_path = processed / "sleep_X.npy"
    y_path = processed / "sleep_y.npy"

    if not X_path.exists():
        print("❌ Preprocessed data not found. Run one of:")
        print("   python src/preprocessing/generate_synthetic.py  (no download)")
        print("   python src/preprocessing/preprocess.py --dataset sleep-edf")
        return

    print(f"\n📂 Loading data from {processed}...")
    X = np.load(X_path)   # (n_epochs, n_channels, epoch_samples)
    y = np.load(y_path)   # (n_epochs,)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Class counts: {np.bincount(y)}")

    # ── Train / Val / Test Split ───────────────────────────────────────────────
    # Split by INDEX to preserve temporal continuity within each split
    n = len(X)
    test_n  = int(n * train_cfg["test_split"])
    val_n   = int(n * train_cfg["val_split"])
    train_n = n - val_n - test_n

    X_train, y_train = X[:train_n], y[:train_n]
    X_val,   y_val   = X[train_n:train_n+val_n], y[train_n:train_n+val_n]
    X_test,  y_test  = X[train_n+val_n:], y[train_n+val_n:]

    print(f"\n   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} epochs")

    seq_len = config["sleep_model"]["sequence_length"]

    train_ds = SleepSequenceDataset(X_train, y_train, seq_len=seq_len, stride=1)
    val_ds   = SleepSequenceDataset(X_val,   y_val,   seq_len=seq_len, stride=seq_len)
    test_ds  = SleepSequenceDataset(X_test,  y_test,  seq_len=seq_len, stride=seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"   Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_sleep_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 DenseSleepNet loaded — {n_params:,} trainable parameters")

    # ── Loss with class weighting ──────────────────────────────────────────────
    class_weights = compute_class_weights(y_train, n_classes=config["sleep_model"]["n_classes"])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    # Reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    early_stop = EarlyStopping(patience=train_cfg["patience"])

    # ── Training Loop ─────────────────────────────────────────────────────────
    os.makedirs(config["paths"]["models"],      exist_ok=True)
    os.makedirs(config["paths"]["checkpoints"], exist_ok=True)
    os.makedirs(config["paths"]["logs"],        exist_ok=True)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n🚀 Training for up to {args.epochs} epochs...\n")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'Time':>6}")
    print("─" * 58)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.2%} {val_loss:>10.4f} {val_acc:>8.2%} {elapsed:>5.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = Path(config["paths"]["models"]) / "dense_sleep_net.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "config":      config,
            }, model_path)
            print(f"        ✅ Best model saved (val_loss={val_loss:.4f})")

        scheduler.step(val_loss)
        early_stop.step(val_loss)
        if early_stop.should_stop:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
            break

    # ── Test Evaluation ───────────────────────────────────────────────────────
    print("\n📊 Loading best model for test evaluation...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    class_names = config["sleep_model"]["class_names"]
    print(f"\n🎯 Test Accuracy: {test_acc:.2%}\n")
    print("Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))

    # Save training history for dashboard
    np.save(Path(config["paths"]["logs"]) / "sleep_history.npy", history)
    print(f"\n✅ Training complete! Model saved to: {model_path}")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenseSleepNet")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    train(args)