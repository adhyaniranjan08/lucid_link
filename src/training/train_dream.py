"""
train_dream.py
==============
Training Pipeline for EEGNet (Dream Content Classification)

What this script does:
  1. Loads preprocessed EEG epochs from data/processed/dream_X.npy
  2. Splits into train / val / test
  3. Applies data augmentation (Gaussian noise, channel dropout)
  4. Trains EEGNet with cosine annealing LR schedule
  5. Reports per-class accuracy (important: some dream categories are harder)
  6. Saves best model to models/saved/eegnet_dream.pt

Usage:
  python src/training/train_dream.py
  python src/training/train_dream.py --attention   # use EEGNet + attention
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
import yaml
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.eegnet import build_dream_model


# ─── Config ───────────────────────────────────────────────────────────────────
def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Dataset with Augmentation ────────────────────────────────────────────────
class DreamEEGDataset(Dataset):
    """
    EEG dataset for dream content classification.

    Augmentation (training only):
      - Gaussian noise: adds small random noise to simulate electrode variability
      - Channel dropout: randomly zeros out one channel (simulates bad electrodes)
      - Amplitude scaling: randomly scales signal amplitude ±20%
    """

    def __init__(self, X, y, augment=False, noise_std=0.1):
        """
        Args:
            X:         (n_epochs, n_channels, temporal_length) float32
            y:         (n_epochs,) int64 class labels
            augment:   whether to apply augmentation (True during training)
            noise_std: std of Gaussian noise relative to signal std
        """
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()   # (n_channels, T)
        y = self.y[idx]

        if self.augment:
            # 1. Add Gaussian noise
            if torch.rand(1).item() < 0.5:
                noise = torch.randn_like(x) * self.noise_std * x.std()
                x = x + noise

            # 2. Random channel dropout (zero out 1 channel)
            if torch.rand(1).item() < 0.3:
                ch = torch.randint(0, x.shape[0], (1,)).item()
                x[ch] = 0.0

            # 3. Random amplitude scaling [0.8, 1.2]
            if torch.rand(1).item() < 0.5:
                scale = 0.8 + torch.rand(1).item() * 0.4
                x = x * scale

        return x, y


# ─── Weighted Sampler for Class Imbalance ─────────────────────────────────────
def make_weighted_sampler(y):
    """
    Create a sampler that upsamples minority classes.

    Instead of class weighting in the loss, we oversample rare classes
    so every training batch has roughly equal class representation.
    """
    class_counts = np.bincount(y)
    weights_per_class = 1.0 / (class_counts + 1e-6)
    sample_weights = weights_per_class[y]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(y),
        replacement=True,
    )
    return sampler


# ─── Training Utilities ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)          # (B, n_classes)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == y_batch).sum().item()
        total_samples += len(y_batch)
        total_loss    += loss.item() * len(y_batch)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        preds = logits.argmax(dim=-1)
        total_correct += (preds == y_batch).sum().item()
        total_samples += len(y_batch)
        total_loss    += loss.item() * len(y_batch)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience    = patience
        self.counter     = 0
        self.best_loss   = float("inf")
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ─── Main Training Function ────────────────────────────────────────────────────
def train(args):
    config    = load_config(args.config)
    train_cfg = config["training"]

    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    processed = Path(config["paths"]["processed"])
    X_path = processed / "dream_X.npy"
    y_path = processed / "dream_y.npy"

    if not X_path.exists():
        print("❌ Dream data not found. Run one of:")
        print("   python src/preprocessing/generate_synthetic.py  (no download)")
        print("   python src/preprocessing/preprocess.py --dataset eeg-imagenet")
        return

    print(f"\n📂 Loading dream EEG data from {processed}...")
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Class counts: {np.bincount(y.astype(int))}")

    # ── Adjust model config to match actual data shape ─────────────────────────
    # This handles the case where synthetic data has different dims than config
    actual_channels = X.shape[1]
    actual_length   = X.shape[2]
    config["dream_model"]["n_channels"]      = actual_channels
    config["dream_model"]["temporal_length"] = actual_length
    config["dream_model"]["n_classes"]       = len(np.unique(y))

    # ── Train / Val / Test Split ───────────────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=train_cfg["test_split"],
        random_state=train_cfg["seed"],
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=train_cfg["val_split"] / (1 - train_cfg["test_split"]),
        random_state=train_cfg["seed"],
        stratify=y_temp,
    )
    print(f"\n   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Datasets & Loaders ────────────────────────────────────────────────────
    train_ds = DreamEEGDataset(X_train, y_train, augment=True)
    val_ds   = DreamEEGDataset(X_val,   y_val,   augment=False)
    test_ds  = DreamEEGDataset(X_test,  y_test,  augment=False)

    # Use weighted sampler for balanced batches
    sampler = make_weighted_sampler(y_train.astype(int))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,   num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,     num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,     num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_dream_model(config, use_attention=args.attention).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "EEGNet + Attention" if args.attention else "EEGNet"
    print(f"\n🧠 {model_name} loaded — {n_params:,} trainable parameters")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)   # label smoothing reduces overconfidence

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Cosine annealing: smoothly decays LR to near-zero and restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    early_stop  = EarlyStopping(patience=train_cfg["patience"])
    best_val_acc = 0.0

    os.makedirs(config["paths"]["models"],      exist_ok=True)
    os.makedirs(config["paths"]["logs"],        exist_ok=True)

    model_path = Path(config["paths"]["models"]) / "eegnet_dream.pt"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n🚀 Training for up to {args.epochs} epochs...\n")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10} {'Time':>6}")
    print("─" * 68)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.2%} {val_loss:>10.4f} {val_acc:>8.2%} {current_lr:>10.2e} {elapsed:>5.1f}s")

        # Save best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_acc":     val_acc,
                "config":      config,
            }, model_path)
            print(f"        ✅ Best model saved (val_acc={val_acc:.2%})")

        early_stop.step(val_loss)
        if early_stop.should_stop:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
            break

    # ── Test Evaluation ───────────────────────────────────────────────────────
    print("\n📊 Loading best model for test evaluation...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    class_names = config["dream_model"]["class_names"]
    # Ensure class_names matches actual number of classes
    n_actual = config["dream_model"]["n_classes"]
    if len(class_names) < n_actual:
        class_names = class_names + [f"Class_{i}" for i in range(len(class_names), n_actual)]

    print(f"\n🎯 Test Accuracy: {test_acc:.2%}\n")
    print("Classification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=class_names[:n_actual],
        zero_division=0
    ))

    np.save(Path(config["paths"]["logs"]) / "dream_history.npy", history)
    print(f"\n✅ Training complete! Model saved to: {model_path}")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEGNet for Dream Classification")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--epochs",     type=int,  default=50)
    parser.add_argument("--batch_size", type=int,  default=32)
    parser.add_argument("--attention",  action="store_true",
                        help="Use EEGNet with channel attention module")
    args = parser.parse_args()
    train(args)