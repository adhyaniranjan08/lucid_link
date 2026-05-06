"""
test_pipeline.py
================
End-to-end sanity check — no datasets or trained models needed.

Tests:
  1. Synthetic data generation
  2. Preprocessing (bandpass filter, artifact removal, epoching)
  3. DenseSleepNet forward pass (random weights)
  4. EEGNet forward pass (random weights)
  5. Full inference pipeline (mock models)
  6. Hypnogram generation

Run:
  python tests/test_pipeline.py
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def separator(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print('─'*55)


def test_preprocessing():
    separator("TEST 1: Preprocessing")
    from src.preprocessing.preprocess import bandpass_filter, reject_artifacts, segment_into_epochs, normalize_epochs

    # Simulate 60s of 2-channel EEG at 100Hz
    fs = 100
    duration = 60
    signal = np.random.randn(2, duration * fs) * 50   # 50µV amplitude

    # Bandpass filter
    filtered = bandpass_filter(signal, 1.0, 40.0, fs)
    assert filtered.shape == signal.shape, "Filter changed shape!"
    print("  ✅ Bandpass filter: OK")

    # Segment into 30s epochs
    epochs = segment_into_epochs(filtered, epoch_duration=30, fs=fs)
    assert epochs.shape == (2, 2, 3000), f"Expected (2, 2, 3000), got {epochs.shape}"
    print(f"  ✅ Segmentation: {epochs.shape} — OK")

    # Artifact rejection
    clean, mask = reject_artifacts(epochs, threshold_uv=100.0)
    print(f"  ✅ Artifact rejection: {clean.shape[0]}/{epochs.shape[0]} epochs kept — OK")

    # Normalisation
    normed = normalize_epochs(clean)
    assert abs(normed.mean()) < 0.5, "Normalisation failed: mean too large"
    print(f"  ✅ Normalisation: mean={normed.mean():.4f}, std={normed.std():.4f} — OK")


def test_dense_sleep_net():
    separator("TEST 2: DenseSleepNet")
    from src.models.dense_sleep_net import DenseSleepNet

    model = DenseSleepNet(n_channels=2, n_classes=5, sequence_length=5,
                          d_model=32, n_heads=2, n_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Batch of 2, sequence of 5 epochs, 2 channels, 3000 samples
    x = torch.randn(2, 5, 2, 3000)
    out = model(x)
    assert out.shape == (2, 5, 5), f"Expected (2,5,5), got {out.shape}"
    print(f"  ✅ Forward pass: {x.shape} → {out.shape} — OK")

    # Single epoch predict
    epoch = np.random.randn(2, 3000).astype(np.float32)
    stage, probs = model.predict_single(epoch)
    assert 0 <= stage <= 4
    assert abs(probs.sum() - 1.0) < 1e-5
    print(f"  ✅ Single predict: stage={stage} ({['Wake','N1','N2','N3','REM'][stage]}), "
          f"confidence={probs.max():.2%} — OK")


def test_eegnet():
    separator("TEST 3: EEGNet")
    from src.models.eegnet import EEGNet, EEGNetWithAttention

    for ModelClass in [EEGNet, EEGNetWithAttention]:
        model = ModelClass(n_channels=14, n_classes=6, temporal_length=128)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {ModelClass.__name__}: {n_params:,} parameters")

        x = torch.randn(8, 14, 128)
        out = model(x)
        assert out.shape == (8, 6), f"Expected (8,6), got {out.shape}"
        print(f"  ✅ Forward pass: {x.shape} → {out.shape} — OK")

        epoch = np.random.randn(14, 128).astype(np.float32)
        cat, probs, conf = model.predict(epoch)
        assert 0 <= cat <= 5
        assert abs(probs.sum() - 1.0) < 1e-5
        print(f"  ✅ Predict: category={cat}, confidence={conf:.2%} — OK")


def test_synthetic_generator():
    separator("TEST 4: Synthetic Data Generator")
    from src.preprocessing.generate_synthetic import make_eeg_signal

    for stage in range(5):
        sig = make_eeg_signal(30, 100, stage, n_channels=2)
        assert sig.shape == (2, 3000), f"Stage {stage}: wrong shape {sig.shape}"
        assert not np.any(np.isnan(sig)), f"Stage {stage}: NaN values!"
    print("  ✅ All 5 sleep stages synthesized correctly — OK")

    sig = make_eeg_signal(1, 128, stage=4, n_channels=14)
    assert sig.shape == (14, 128)
    print(f"  ✅ Dream epoch generation: {sig.shape} — OK")


def test_inference_pipeline():
    separator("TEST 5: Inference Pipeline (mock models)")
    from src.inference.pipeline import EEGDreamPipeline
    from unittest.mock import patch, MagicMock
    import torch.nn.functional as F

    # Create pipeline without loading from disk
    pipeline = EEGDreamPipeline.__new__(EEGDreamPipeline)
    pipeline.config = {
        "preprocessing": {
            "sample_rate": 100, "dream_sample_rate": 128,
            "epoch_duration": 30, "bandpass_low": 1.0,
            "bandpass_high": 40.0,
        },
        "sleep_model": {
            "n_channels": 2, "n_classes": 5, "sequence_length": 5,
            "d_model": 32, "n_heads": 2, "n_layers": 2, "dropout": 0.1,
        },
        "dream_model": {
            "n_channels": 14, "n_classes": 6, "temporal_length": 128,
            "F1": 8, "D": 2, "F2": 16, "kernel_length": 64, "dropout": 0.5,
        },
        "paths": {"models": "models/saved"},
    }
    pipeline.device       = torch.device("cpu")
    pipeline.sleep_fs     = 100
    pipeline.dream_fs     = 128
    pipeline.epoch_dur    = 30
    pipeline.low          = 1.0
    pipeline.high         = 40.0
    pipeline.epoch_samples= 3000
    pipeline.seq_len      = 5
    pipeline._epoch_buffer= []

    # Use randomly-initialized models
    from src.models.dense_sleep_net import build_sleep_model
    from src.models.eegnet import build_dream_model
    pipeline.sleep_model = build_sleep_model(pipeline.config).to(pipeline.device)
    pipeline.dream_model = build_dream_model(pipeline.config).to(pipeline.device)
    pipeline.sleep_model.eval()
    pipeline.dream_model.eval()

    # Test process_epoch
    epoch = np.random.randn(2, 3000).astype(np.float32)
    result = pipeline.process_epoch(epoch, epoch_idx=0)
    assert 0 <= result.sleep_stage <= 4
    assert result.sleep_probs.shape == (5,)
    print(f"  ✅ process_epoch: stage={result.sleep_stage_name}, REM={result.is_rem} — OK")

    # Test full run with short signal (10 epochs)
    eeg = np.random.randn(2, 30000).astype(np.float32)  # 10 × 30s epochs
    output = pipeline.run(eeg, verbose=False)
    assert output.total_epochs == 10
    assert len(output.sleep_timeline) == 10
    print(f"  ✅ pipeline.run: {output.total_epochs} epochs processed — OK")
    print(f"     REM detected: {len(output.rem_epochs)} epochs")
    print(f"     Dream predictions: {len(output.dream_predictions)}")


def test_dataset_classes():
    separator("TEST 6: Dataset Classes")
    from src.training.train_sleep import SleepSequenceDataset
    from src.training.train_dream import DreamEEGDataset

    # Sleep dataset
    X = np.random.randn(100, 2, 3000).astype(np.float32)
    y = np.random.randint(0, 5, 100).astype(np.int64)
    ds = SleepSequenceDataset(X, y, seq_len=5, stride=1)
    assert len(ds) == 96   # 100 - 5 + 1
    x_s, y_s = ds[0]
    assert x_s.shape == (5, 2, 3000)
    assert y_s.shape == (5,)
    print(f"  ✅ SleepSequenceDataset: {len(ds)} samples, x={x_s.shape} — OK")

    # Dream dataset
    X_d = np.random.randn(200, 14, 128).astype(np.float32)
    y_d = np.random.randint(0, 6, 200).astype(np.int64)
    ds_d = DreamEEGDataset(X_d, y_d, augment=True)
    x_d, y_d_item = ds_d[0]
    assert x_d.shape == (14, 128)
    print(f"  ✅ DreamEEGDataset: {len(ds_d)} samples, x={x_d.shape} — OK")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  🧠 EEG DREAM DETECTION — PIPELINE TEST SUITE")
    print("═"*55)

    tests = [
        test_preprocessing,
        test_dense_sleep_net,
        test_eegnet,
        test_synthetic_generator,
        test_inference_pipeline,
        test_dataset_classes,
    ]

    passed, failed = 0, 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"\n  ❌ FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "═"*55)
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  🎉 All tests passed! Pipeline is ready.")
    else:
        print("  ⚠️  Some tests failed. Check errors above.")
    print("═"*55 + "\n")