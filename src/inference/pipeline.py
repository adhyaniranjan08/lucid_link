"""
pipeline.py
===========
End-to-End EEG Dream Detection & Classification Pipeline

This is the main inference engine that ties everything together:

  EEG Signal
      │
      ▼
  Preprocessing (bandpass filter → normalize)
      │
      ▼
  DenseSleepNet → Sleep Stage (Wake / N1 / N2 / N3 / REM)
      │
      ├── NOT REM → log stage, wait for next epoch
      │
      └── REM ────────────────────────────────────────────────────────►
                                                                      │
                                                              EEGNet (Dream Classifier)
                                                                      │
                                                              Dream Category + Confidence

Usage:
  # Run on a numpy file
  python src/inference/pipeline.py --input data/processed/sample_eeg.npy

  # Run demo with synthetic data
  python src/inference/pipeline.py --demo
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.dense_sleep_net import build_sleep_model
from src.models.eegnet import build_dream_model
from src.preprocessing.preprocess import bandpass_filter, normalize_epochs


# ─── Config ───────────────────────────────────────────────────────────────────
def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Result Dataclass ─────────────────────────────────────────────────────────
@dataclass
class EpochResult:
    """Holds the result of processing one EEG epoch through the pipeline."""
    epoch_idx:         int
    timestamp_s:       float                    # seconds from recording start
    sleep_stage:       int                      # 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
    sleep_stage_name:  str
    sleep_probs:       np.ndarray               # (5,) probabilities
    is_rem:            bool
    dream_category:    Optional[int]   = None   # None if not REM
    dream_category_name: Optional[str] = None
    dream_probs:       Optional[np.ndarray] = None
    dream_confidence:  Optional[float] = None


@dataclass
class PipelineOutput:
    """Aggregated output of the full pipeline run."""
    results:           List[EpochResult] = field(default_factory=list)
    sleep_timeline:    List[int]         = field(default_factory=list)   # stage per epoch
    rem_epochs:        List[int]         = field(default_factory=list)   # epoch indices
    dream_predictions: List[dict]        = field(default_factory=list)   # REM predictions
    total_epochs:      int = 0
    rem_percentage:    float = 0.0

    def summary(self):
        """Print a human-readable summary."""
        print("\n" + "═" * 60)
        print("  🧠 EEG DREAM DETECTION PIPELINE — RESULTS")
        print("═" * 60)
        print(f"  Total epochs processed : {self.total_epochs}")
        print(f"  REM epochs detected    : {len(self.rem_epochs)} ({self.rem_percentage:.1f}%)")
        print(f"  Dream predictions      : {len(self.dream_predictions)}")

        if self.dream_predictions:
            print("\n  🌙 Dream Content Predictions:")
            print("  " + "─" * 40)
            for p in self.dream_predictions:
                bar_len = int(p["confidence"] * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  Epoch {p['epoch_idx']:>4}  │{bar}│ {p['category_name']:<12} {p['confidence']:.1%}")

        print("\n  Sleep Stage Distribution:")
        from collections import Counter
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]
        counts = Counter(self.sleep_timeline)
        for stage_id, name in enumerate(stage_names):
            cnt = counts.get(stage_id, 0)
            pct = 100 * cnt / max(self.total_epochs, 1)
            bar = "█" * int(pct / 5)
            print(f"  {name:<5} │{bar:<20}│ {cnt:>4} epochs ({pct:.1f}%)")
        print("═" * 60)


# ─── Pipeline Class ───────────────────────────────────────────────────────────
class EEGDreamPipeline:
    """
    Main inference pipeline.

    Loads both models from disk and provides:
      - process_epoch():    classify a single epoch
      - run():              process a full recording
      - stream_epoch():     generator for real-time simulation
    """

    STAGE_NAMES  = ["Wake", "N1", "N2", "N3", "REM"]
    DREAM_CATEGORIES = ["Face", "Object", "Animal", "Scene", "Text", "Movement"]

    def __init__(self, config_path="configs/config.yaml", device=None):
        self.config = load_config(config_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🖥️  Pipeline running on: {self.device}")

        self._load_sleep_model()
        self._load_dream_model()

        # Preprocessing parameters
        self.sleep_fs     = self.config["preprocessing"]["sample_rate"]
        self.dream_fs     = self.config["preprocessing"]["dream_sample_rate"]
        self.epoch_dur    = self.config["preprocessing"]["epoch_duration"]
        self.low          = self.config["preprocessing"]["bandpass_low"]
        self.high         = self.config["preprocessing"]["bandpass_high"]
        self.epoch_samples = int(self.epoch_dur * self.sleep_fs)

        # Buffer for sequence context (DenseSleepNet needs seq_len epochs)
        self.seq_len      = self.config["sleep_model"]["sequence_length"]
        self._epoch_buffer: List[np.ndarray] = []

    def _load_sleep_model(self):
        """Load DenseSleepNet from checkpoint."""
        model_path = Path(self.config["paths"]["models"]) / "dense_sleep_net.pt"

        if not model_path.exists():
            print(f"⚠️  Sleep model not found at {model_path}")
            print("   Using randomly-initialized model (train first for real predictions)")
            self.sleep_model = build_sleep_model(self.config).to(self.device)
        else:
            checkpoint = torch.load(str(model_path), map_location=self.device)
            self.sleep_model = build_sleep_model(checkpoint["config"]).to(self.device)
            self.sleep_model.load_state_dict(checkpoint["model_state"])
            print(f"✅ Sleep model loaded (val_acc={checkpoint.get('val_acc', '?')})")

        self.sleep_model.eval()

    def _load_dream_model(self):
        """Load EEGNet from checkpoint."""
        model_path = Path(self.config["paths"]["models"]) / "eegnet_dream.pt"

        if not model_path.exists():
            print(f"⚠️  Dream model not found at {model_path}")
            print("   Using randomly-initialized model (train first for real predictions)")
            self.dream_model = build_dream_model(self.config).to(self.device)
        else:
            checkpoint = torch.load(str(model_path), map_location=self.device)
            self.dream_model = build_dream_model(checkpoint["config"]).to(self.device)
            self.dream_model.load_state_dict(checkpoint["model_state"])
            print(f"✅ Dream model loaded (val_acc={checkpoint.get('val_acc', '?')})")

        self.dream_model.eval()

    # ── Sleep Stage Classification ────────────────────────────────────────────
    def classify_sleep_stage(self, epoch: np.ndarray):
        """
        Classify a single EEG epoch into a sleep stage.

        Maintains an internal rolling buffer so the Transformer
        always gets seq_len epochs for context.

        Args:
            epoch: (n_channels, epoch_samples)

        Returns:
            stage:      int (0–4)
            stage_name: str
            probs:      np.ndarray (5,)
        """
        # Add to rolling buffer
        self._epoch_buffer.append(epoch)
        if len(self._epoch_buffer) > self.seq_len:
            self._epoch_buffer.pop(0)

        # Pad with zeros if we don't have a full sequence yet
        buffer = list(self._epoch_buffer)
        while len(buffer) < self.seq_len:
            buffer.insert(0, np.zeros_like(epoch))

        # Stack → (1, seq_len, n_channels, epoch_samples)
        seq = np.stack(buffer, axis=0)                    # (seq_len, C, T)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.sleep_model(seq_tensor)         # (1, seq_len, 5)
            # Take prediction for the LAST (most recent) epoch
            probs = F.softmax(logits[0, -1], dim=-1).cpu().numpy()

        stage = int(probs.argmax())
        return stage, self.STAGE_NAMES[stage], probs

    # ── Dream Content Classification ──────────────────────────────────────────
    def classify_dream(self, epoch: np.ndarray):
        """
        Classify the dream content of a REM epoch.

        Args:
            epoch: (n_channels, temporal_length) — may need resampling if
                   sleep_fs ≠ dream_fs

        Returns:
            category:      int
            category_name: str
            probs:         np.ndarray (n_classes,)
            confidence:    float
        """
        # Resample if the two models use different sampling rates
        if self.sleep_fs != self.dream_fs:
            from scipy.signal import resample
            target_len = int(epoch.shape[-1] * self.dream_fs / self.sleep_fs)
            epoch = resample(epoch, target_len, axis=-1)

        # Adjust channels if needed (dream model may expect more channels)
        dream_n_ch = self.config["dream_model"]["n_channels"]
        if epoch.shape[0] < dream_n_ch:
            # Tile channels to match expected count
            reps = (dream_n_ch // epoch.shape[0]) + 1
            epoch = np.tile(epoch, (reps, 1))[:dream_n_ch]
        elif epoch.shape[0] > dream_n_ch:
            epoch = epoch[:dream_n_ch]

        # Adjust temporal length
        target_t = self.config["dream_model"]["temporal_length"]
        if epoch.shape[-1] != target_t:
            from scipy.signal import resample
            epoch = resample(epoch, target_t, axis=-1)

        x = torch.tensor(epoch, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.dream_model(x)                  # (1, n_classes)
            probs  = F.softmax(logits[0], dim=-1).cpu().numpy()

        category   = int(probs.argmax())
        confidence = float(probs.max())
        n_classes  = len(probs)

        # Build safe class name list
        names = self.DREAM_CATEGORIES[:n_classes]
        while len(names) < n_classes:
            names.append(f"Category_{len(names)}")

        return category, names[category], probs, confidence

    # ── Process One Epoch ─────────────────────────────────────────────────────
    def process_epoch(self, epoch: np.ndarray, epoch_idx: int) -> EpochResult:
        """
        Full pipeline for a single epoch.

        Args:
            epoch:     (n_channels, epoch_samples) — raw (unfiltered) EEG
            epoch_idx: sequential index

        Returns:
            EpochResult with all fields filled
        """
        # Preprocess
        epoch = bandpass_filter(epoch, self.low, self.high, self.sleep_fs)
        epoch = (epoch - epoch.mean()) / (epoch.std() + 1e-8)

        timestamp_s = epoch_idx * self.epoch_dur

        # Stage 1: Sleep classification
        stage, stage_name, sleep_probs = self.classify_sleep_stage(epoch)
        is_rem = (stage == 4)

        result = EpochResult(
            epoch_idx       = epoch_idx,
            timestamp_s     = timestamp_s,
            sleep_stage     = stage,
            sleep_stage_name= stage_name,
            sleep_probs     = sleep_probs,
            is_rem          = is_rem,
        )

        # Stage 2: Dream classification (only if REM)
        if is_rem:
            cat, cat_name, dream_probs, confidence = self.classify_dream(epoch)
            result.dream_category      = cat
            result.dream_category_name = cat_name
            result.dream_probs         = dream_probs
            result.dream_confidence    = confidence

        return result

    # ── Run Full Recording ────────────────────────────────────────────────────
    def run(self, eeg_signal: np.ndarray, verbose=True) -> PipelineOutput:
        """
        Process a full EEG recording.

        Args:
            eeg_signal: (n_channels, n_samples) — continuous raw EEG
            verbose:    print progress

        Returns:
            PipelineOutput
        """
        # Segment into epochs
        n_ch, n_samples = eeg_signal.shape
        epoch_samples   = int(self.epoch_dur * self.sleep_fs)
        n_epochs        = n_samples // epoch_samples

        print(f"\n🔍 Processing {n_epochs} epochs ({n_epochs * self.epoch_dur / 60:.1f} minutes)...")

        output = PipelineOutput()

        for i in range(n_epochs):
            epoch = eeg_signal[:, i * epoch_samples: (i + 1) * epoch_samples]
            result = self.process_epoch(epoch, i)
            output.results.append(result)
            output.sleep_timeline.append(result.sleep_stage)

            if result.is_rem:
                output.rem_epochs.append(i)
                if result.dream_category is not None:
                    output.dream_predictions.append({
                        "epoch_idx":     i,
                        "timestamp_s":   result.timestamp_s,
                        "category":      result.dream_category,
                        "category_name": result.dream_category_name,
                        "confidence":    result.dream_confidence,
                        "probs":         result.dream_probs.tolist(),
                    })

            if verbose and (i + 1) % 20 == 0:
                print(f"  Epoch {i+1}/{n_epochs} — Stage: {result.sleep_stage_name}"
                      + (f" → 🌙 {result.dream_category_name} ({result.dream_confidence:.0%})"
                         if result.is_rem else ""))

        output.total_epochs   = n_epochs
        output.rem_percentage = 100 * len(output.rem_epochs) / max(n_epochs, 1)

        return output

    # ── Streaming / Real-time Mode ────────────────────────────────────────────
    def stream_epochs(self, eeg_signal: np.ndarray, delay_s=0.1):
        """
        Generator that simulates real-time processing.

        Yields one EpochResult per epoch with an optional delay to
        mimic live data from a headband.

        Usage:
            for result in pipeline.stream_epochs(eeg):
                print(result.sleep_stage_name)
        """
        n_ch, n_samples = eeg_signal.shape
        epoch_samples   = int(self.epoch_dur * self.sleep_fs)
        n_epochs        = n_samples // epoch_samples

        for i in range(n_epochs):
            epoch = eeg_signal[:, i * epoch_samples: (i + 1) * epoch_samples]
            result = self.process_epoch(epoch, i)
            time.sleep(delay_s)    # simulate real-time pacing
            yield result


# ─── CLI Entry Point ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EEG Dream Detection Pipeline")
    parser.add_argument("--input",  type=str, help="Path to .npy EEG file (n_channels, n_samples)")
    parser.add_argument("--demo",   action="store_true", help="Run with synthetic data")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    pipeline = EEGDreamPipeline(config_path=args.config)

    if args.demo or args.input is None:
        print("\n🎮 Running DEMO with synthetic EEG data...")
        # Generate a synthetic 2-hour sleep session
        fs           = pipeline.sleep_fs
        n_channels   = pipeline.config["sleep_model"]["n_channels"]
        duration_min = 30    # 30 minutes for quick demo
        n_samples    = duration_min * 60 * fs

        # Simulate sleep stages in the signal using different frequency content
        from src.preprocessing.generate_synthetic import make_eeg_signal
        epoch_dur = pipeline.epoch_dur
        epoch_smp = int(epoch_dur * fs)
        n_epochs  = n_samples // epoch_smp

        # Simple hypnogram for demo
        demo_hypnogram = (
            [0] * 5 + [1] * 3 + [2] * 8 + [3] * 10 +
            [4] * 8 + [2] * 5 + [4] * 10 + [0] * 3   # two REM periods
        )[:n_epochs]

        eeg_chunks = []
        for stage in demo_hypnogram:
            eeg_chunks.append(make_eeg_signal(epoch_dur, fs, stage, n_ch=n_channels))
        eeg_signal = np.concatenate(eeg_chunks, axis=-1)

    else:
        print(f"\n📂 Loading EEG from {args.input}...")
        eeg_signal = np.load(args.input)
        if eeg_signal.ndim == 1:
            eeg_signal = eeg_signal[np.newaxis, :]   # add channel dim

    output = pipeline.run(eeg_signal)
    output.summary()

    # Save results
    import json
    out_path = Path("data/processed/pipeline_results.json")
    safe_results = []
    for p in output.dream_predictions:
        safe_results.append({k: v for k, v in p.items() if k != "probs"})

    with open(out_path, "w") as f:
        json.dump({
            "sleep_timeline":    output.sleep_timeline,
            "rem_epochs":        output.rem_epochs,
            "dream_predictions": safe_results,
            "rem_percentage":    output.rem_percentage,
        }, f, indent=2)

    print(f"\n💾 Results saved to {out_path}")


if __name__ == "__main__":
    main()