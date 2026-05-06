"""
generate_synthetic.py
=====================
Generate realistic synthetic EEG data for testing the pipeline
without downloading any datasets.

The synthetic signals mimic real EEG by summing sinusoids at
brain-rhythm frequencies:
  - Delta (0.5–4 Hz):  dominant in N3 (deep) sleep
  - Theta (4–8 Hz):    dominant in N1, REM
  - Alpha (8–13 Hz):   dominant in relaxed wakefulness
  - Beta  (13–30 Hz):  dominant in active wakefulness
  - Gamma (30–40 Hz):  present in REM and wakefulness

Usage:
  python src/preprocessing/generate_synthetic.py
"""

import numpy as np
from pathlib import Path
import yaml
import os


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def make_eeg_signal(duration_s, fs, stage, n_channels=2, noise_level=5.0):
    """
    Synthesize one EEG epoch with stage-appropriate frequency content.
    
    Args:
        duration_s:  epoch duration in seconds
        fs:          sampling frequency (Hz)
        stage:       0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
        n_channels:  number of EEG channels
        noise_level: amplitude of Gaussian noise in µV
    
    Returns:
        signal: (n_channels, n_samples) in µV
    """
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    
    # Frequency-band amplitudes per sleep stage (in µV)
    # Columns: [delta, theta, alpha, beta, gamma]
    amp_table = {
        0: [5,  5,  20, 15, 5],    # Wake  - alpha dominant
        1: [10, 20, 10, 5,  3],    # N1    - theta dominant
        2: [20, 10, 5,  5,  2],    # N2    - delta rising, sleep spindles
        3: [60, 5,  3,  2,  1],    # N3    - delta dominant (slow-wave)
        4: [5,  25, 5,  10, 8],    # REM   - theta + gamma (dreaming!)
    }
    amps = amp_table[stage]
    
    # Representative frequencies for each band
    freqs = [2.0, 6.0, 10.0, 20.0, 35.0]
    
    signal = np.zeros((n_channels, len(t)))
    
    for ch in range(n_channels):
        for amp, freq in zip(amps, freqs):
            # Random phase per channel/band makes each channel unique
            phase = np.random.uniform(0, 2 * np.pi)
            signal[ch] += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Add realistic noise
        signal[ch] += np.random.randn(len(t)) * noise_level
        
        # Add occasional slow drift (common in real EEG)
        drift_freq = np.random.uniform(0.1, 0.5)
        signal[ch] += 3.0 * np.sin(2 * np.pi * drift_freq * t)
    
    return signal


def generate_sleep_edf_synthetic(config, n_subjects=5):
    """
    Generate synthetic Sleep-EDF style data.
    
    Creates a realistic hypnogram (sleep architecture) that mirrors
    how humans actually sleep: wake → N1 → N2 → N3 → REM cycles.
    """
    print("🔄 Generating synthetic Sleep-EDF data...")
    
    fs        = config["preprocessing"]["sample_rate"]
    epoch_dur = config["preprocessing"]["epoch_duration"]
    
    all_X, all_y = [], []
    
    # Simulate a realistic 8-hour sleep session
    # A typical night: ~4-5 REM cycles, each ~90 minutes
    sleep_hypnogram = (
        [0] * 5  +    # 5 min wake
        [1] * 5  +    # 5 min N1
        [2] * 10 +    # 10 min N2
        [3] * 20 +    # 20 min N3 (deep sleep)
        [2] * 5  +    # 5 min N2
        [4] * 15 +    # 15 min REM ← first dream period
        [2] * 10 +    # 10 min N2
        [3] * 15 +    # 15 min N3
        [2] * 5  +    # transition
        [4] * 20 +    # 20 min REM ← second dream period
        [1] * 5  +    # light sleep
        [4] * 25 +    # 25 min REM ← third dream period (longest)
        [0] * 5        # wake up
    )
    
    for subj in range(n_subjects):
        print(f"  Subject {subj+1}/{n_subjects}...")
        subj_X, subj_y = [], []
        
        for stage in sleep_hypnogram:
            # Each entry in hypnogram = 1 epoch (30 seconds)
            sig = make_eeg_signal(epoch_dur, fs, stage, n_channels=2)
            subj_X.append(sig)
            subj_y.append(stage)
        
        all_X.extend(subj_X)
        all_y.extend(subj_y)
    
    X = np.array(all_X)   # (n_epochs, 2, 3000)
    y = np.array(all_y)   # (n_epochs,)
    
    print(f"  Generated {X.shape[0]} epochs, shape {X.shape}")
    print(f"  Class counts: {np.bincount(y)}")
    
    return X, y


def generate_eeg_imagenet_synthetic(config, n_trials=500):
    """
    Generate synthetic EEG-ImageNet style data.
    
    Simulates EEG responses to visual stimuli. Different categories
    produce slightly different neural patterns (simplified).
    
    Visual categories: Face, Object, Animal, Scene, Text, Movement
    """
    print("\n🔄 Generating synthetic EEG-ImageNet data...")
    
    fs         = config["preprocessing"]["dream_sample_rate"]
    epoch_dur  = config["preprocessing"]["dream_epoch_duration"]
    n_channels = config["dream_model"]["n_channels"]
    n_classes  = config["dream_model"]["n_classes"]
    
    all_X, all_y = [], []
    
    for label in range(n_classes):
        # Each class has slightly different alpha suppression / gamma boost
        category_noise = np.random.uniform(0.8, 1.2)
        
        for _ in range(n_trials // n_classes):
            sig = make_eeg_signal(
                epoch_dur, fs,
                stage=4,              # REM-like pattern
                n_channels=n_channels,
                noise_level=5.0 * category_noise
            )
            # Add a category-specific evoked component (P300-like)
            t = np.linspace(0, epoch_dur, int(epoch_dur * fs))
            p300 = 15.0 * np.exp(-((t - 0.3) ** 2) / (2 * 0.05**2))
            sig[0] += p300 * (1 + 0.3 * label)   # slightly different peak per class
            
            all_X.append(sig)
            all_y.append(label)
    
    X = np.array(all_X)                # (n_trials, n_channels, T)
    y = np.array(all_y, dtype=np.int64)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    print(f"  Generated {X.shape[0]} trials, shape {X.shape}")
    print(f"  Class counts: {np.bincount(y)}")
    
    return X, y


def main():
    config = load_config()
    np.random.seed(42)
    
    out_dir = Path(config["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Sleep data
    X_sleep, y_sleep = generate_sleep_edf_synthetic(config, n_subjects=10)
    np.save(out_dir / "sleep_X.npy", X_sleep.astype(np.float32))
    np.save(out_dir / "sleep_y.npy", y_sleep.astype(np.int64))
    print(f"\n✅ Saved → {out_dir}/sleep_X.npy")
    
    # Dream data
    X_dream, y_dream = generate_eeg_imagenet_synthetic(config, n_trials=1200)
    np.save(out_dir / "dream_X.npy", X_dream.astype(np.float32))
    np.save(out_dir / "dream_y.npy", y_dream.astype(np.int64))
    print(f"✅ Saved → {out_dir}/dream_X.npy")
    
    print("\n🎉 Synthetic data generation complete!")
    print("   You can now run training without downloading any datasets.")


if __name__ == "__main__":
    main()