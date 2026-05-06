"""
preprocess.py
=============
EEG Preprocessing Pipeline

Steps performed:
  1. Load EDF files (Sleep-EDF) or HDF5 (EEG-ImageNet)
  2. Bandpass filter (1–40 Hz) to keep brain rhythms of interest
  3. Basic artifact removal (amplitude thresholding)
  4. Segment into fixed-length epochs
  5. Save as numpy arrays for fast loading during training

Usage:
  python preprocess.py --dataset sleep-edf
  python preprocess.py --dataset eeg-imagenet
"""

import os
import argparse
import numpy as np
from scipy.signal import butter, sosfiltfilt
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings("ignore")


# ─── Load config ────────────────────────────────────────────────────────────
def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─── Step 1: Bandpass Filter ─────────────────────────────────────────────────
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Apply a Butterworth bandpass filter to remove unwanted frequencies.
    
    Args:
        signal: numpy array of shape (n_channels, n_samples)
        lowcut:  lower frequency bound in Hz (e.g., 1.0)
        highcut: upper frequency bound in Hz (e.g., 40.0)
        fs:      sampling frequency in Hz
        order:   filter order (4 is a good default)
    
    Returns:
        Filtered signal of same shape
    """
    # Nyquist frequency (max representable frequency)
    nyq = 0.5 * fs
    
    # Normalise cutoffs to [0, 1]
    low  = lowcut  / nyq
    high = highcut / nyq
    
    # Design the filter using second-order sections (numerically stable)
    sos = butter(order, [low, high], btype='band', output='sos')
    
    # Apply zero-phase filtering (no time shift)
    filtered = sosfiltfilt(sos, signal, axis=-1)
    return filtered


# ─── Step 2: Artifact Removal ────────────────────────────────────────────────
def reject_artifacts(epochs, threshold_uv=100.0):
    """
    Remove epochs where any channel exceeds the amplitude threshold.
    
    This is a simple peak-to-peak check. More sophisticated methods
    (ICA, regression) exist but this works well as a first pass.
    
    Args:
        epochs:       numpy array of shape (n_epochs, n_channels, n_samples)
        threshold_uv: amplitude threshold in microvolts
    
    Returns:
        clean_epochs: filtered epochs
        good_mask:    boolean mask of which epochs were kept
    """
    # Peak-to-peak amplitude per epoch per channel
    ptp = np.ptp(epochs, axis=-1)          # (n_epochs, n_channels)
    max_ptp = ptp.max(axis=-1)             # (n_epochs,) - worst channel per epoch
    
    good_mask = max_ptp < threshold_uv
    n_rejected = (~good_mask).sum()
    
    if n_rejected > 0:
        print(f"  Artifact rejection: removed {n_rejected}/{len(epochs)} epochs "
              f"({100*n_rejected/len(epochs):.1f}%)")
    
    return epochs[good_mask], good_mask


# ─── Step 3: Epoch Segmentation ──────────────────────────────────────────────
def segment_into_epochs(signal, epoch_duration, fs):
    """
    Cut a continuous signal into fixed-length non-overlapping epochs.
    
    Args:
        signal:         (n_channels, n_samples)
        epoch_duration: length of each epoch in seconds
        fs:             sampling frequency
    
    Returns:
        epochs: (n_epochs, n_channels, epoch_samples)
    """
    n_channels, n_samples = signal.shape
    epoch_samples = int(epoch_duration * fs)        # samples per epoch
    n_epochs = n_samples // epoch_samples           # drop the last partial epoch
    
    # Reshape: (n_epochs, n_channels, epoch_samples)
    epochs = signal[:, :n_epochs * epoch_samples]
    epochs = epochs.reshape(n_channels, n_epochs, epoch_samples)
    epochs = epochs.transpose(1, 0, 2)             # (n_epochs, n_channels, epoch_samples)
    
    return epochs


# ─── Step 4: Normalisation ───────────────────────────────────────────────────
def normalize_epochs(epochs):
    """
    Z-score normalise each epoch independently.
    
    Each epoch is shifted to have mean=0 and std=1.
    This removes DC offsets and makes different recordings comparable.
    """
    mean = epochs.mean(axis=-1, keepdims=True)
    std  = epochs.std(axis=-1, keepdims=True) + 1e-8   # avoid div-by-zero
    return (epochs - mean) / std


# ─── Sleep-EDF Loader ────────────────────────────────────────────────────────
def load_sleep_edf(data_dir, config):
    """
    Load Sleep-EDF .edf files using MNE.
    
    Sleep-EDF files come in pairs:
      SC4001E0-PSG.edf    → raw EEG recording
      SC4001EC-Hypnogram.edf → sleep stage annotations
    
    Returns:
        X: (n_epochs, n_channels, epoch_samples)  - EEG data
        y: (n_epochs,)                              - sleep stage labels
           0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
    """
    try:
        import mne
    except ImportError:
        raise ImportError("Please install MNE: pip install mne")
    
    # Map annotation text → integer label
    stage_map = {
        "Sleep stage W":  0,   # Wake
        "Sleep stage 1":  1,   # N1
        "Sleep stage 2":  2,   # N2
        "Sleep stage 3":  3,   # N3
        "Sleep stage 4":  3,   # N3 (old classification)
        "Sleep stage R":  4,   # REM
    }
    
    fs        = config["preprocessing"]["sample_rate"]
    epoch_dur = config["preprocessing"]["epoch_duration"]
    low       = config["preprocessing"]["bandpass_low"]
    high      = config["preprocessing"]["bandpass_high"]
    thresh    = config["preprocessing"]["artifact_threshold"]
    
    data_dir = Path(data_dir)
    psg_files = sorted(data_dir.glob("*-PSG.edf"))
    
    if len(psg_files) == 0:
        raise FileNotFoundError(
            f"No *-PSG.edf files found in {data_dir}. "
            "Please download Sleep-EDF from https://physionet.org/content/sleep-edfx/1.0.0/"
        )
    
    all_X, all_y = [], []
    
    for psg_path in psg_files:
        # Find matching hypnogram
        subject_id = psg_path.name[:7]
        hyp_files  = list(data_dir.glob(f"{subject_id}*Hypnogram.edf"))
        if not hyp_files:
            print(f"  Skipping {psg_path.name}: no matching hypnogram found")
            continue
        
        print(f"  Loading {psg_path.name}...")
        
        # Load PSG recording (suppress MNE's verbose output)
        raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
        
        # Pick only EEG channels (Fpz-Cz and Pz-Oz are standard in Sleep-EDF)
        eeg_channels = [ch for ch in raw.ch_names if "EEG" in ch]
        if not eeg_channels:
            eeg_channels = raw.ch_names[:2]   # fallback: first 2 channels
        raw.pick_channels(eeg_channels)
        
        # Resample if needed
        if raw.info["sfreq"] != fs:
            raw.resample(fs, verbose=False)
        
        # Load annotations
        ann = mne.read_annotations(str(hyp_files[0]))
        raw.set_annotations(ann, verbose=False)
        
        # Create fixed-length epochs from annotations
        events, event_id = mne.events_from_annotations(
            raw, event_id=stage_map, chunk_duration=epoch_dur, verbose=False
        )
        
        epochs_mne = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=0, tmax=epoch_dur - 1/fs,
            baseline=None, preload=True, verbose=False
        )
        
        X = epochs_mne.get_data()   # (n_epochs, n_channels, n_times)
        y = epochs_mne.events[:, 2] # labels
        
        # Map back to 0-4 if MNE remapped them
        unique_events = np.unique(y)
        if unique_events.max() > 4:
            label_map = {v: i for i, v in enumerate(sorted(unique_events))}
            y = np.array([label_map[l] for l in y])
        
        # Filter
        X = np.array([bandpass_filter(ep, low, high, fs) for ep in X])
        
        # Artifact removal
        X, mask = reject_artifacts(X, thresh)
        y = y[mask]
        
        # Normalise
        X = normalize_epochs(X)
        
        all_X.append(X)
        all_y.append(y)
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"\n  Sleep-EDF loaded: {X.shape[0]} epochs, {X.shape[1]} channels")
    print(f"  Class distribution: {np.bincount(y)}")
    
    return X, y


# ─── EEG-ImageNet Loader ─────────────────────────────────────────────────────
def load_eeg_imagenet(data_dir, config):
    """
    Load EEG-ImageNet dataset.
    
    The dataset contains EEG recordings while subjects viewed 
    ImageNet images. We treat image categories as dream content classes.
    
    Expected file: eeg_signals_raw_with_mean_std.pth (PyTorch format)
    or a folder of .npy files.
    
    Returns:
        X: (n_epochs, n_channels, n_samples)
        y: (n_epochs,) - visual category labels
    """
    import torch
    
    data_dir = Path(data_dir)
    
    # Try loading the standard EEG-ImageNet format
    pth_file = data_dir / "eeg_signals_raw_with_mean_std.pth"
    
    if pth_file.exists():
        print(f"  Loading {pth_file.name}...")
        data = torch.load(str(pth_file), map_location="cpu")
        
        # dataset["dataset"] is a list of dicts with keys: eeg, label, image
        samples = data["dataset"]
        X = np.array([s["eeg"].numpy() for s in samples])   # (N, C, T)
        y = np.array([s["label"] for s in samples])
        
    else:
        # Look for numpy files
        npy_files = list(data_dir.glob("*.npy"))
        if npy_files:
            print(f"  Loading {len(npy_files)} .npy files...")
            X = np.load(str(data_dir / "eeg_data.npy"))
            y = np.load(str(data_dir / "labels.npy"))
        else:
            raise FileNotFoundError(
                f"No EEG-ImageNet data found in {data_dir}. "
                "Please run: python src/preprocessing/generate_synthetic.py"
            )
    
    fs    = config["preprocessing"]["dream_sample_rate"]
    low   = config["preprocessing"]["bandpass_low"]
    high  = config["preprocessing"]["bandpass_high"]
    thresh = config["preprocessing"]["artifact_threshold"]
    
    # Filter and clean
    print("  Applying bandpass filter...")
    X = np.array([bandpass_filter(ep, low, high, fs) for ep in X])
    X, mask = reject_artifacts(X, thresh)
    y = y[mask]
    X = normalize_epochs(X)
    
    print(f"\n  EEG-ImageNet loaded: {X.shape[0]} epochs, {X.shape[1]} channels")
    print(f"  Class distribution: {np.bincount(y.astype(int))}")
    
    return X, y.astype(np.int64)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EEG Preprocessing Pipeline")
    parser.add_argument("--dataset", choices=["sleep-edf", "eeg-imagenet"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    os.makedirs(config["paths"]["processed"], exist_ok=True)
    
    if args.dataset == "sleep-edf":
        print("\n🔄 Preprocessing Sleep-EDF dataset...")
        X, y = load_sleep_edf(config["paths"]["raw_sleep_edf"], config)
        
        # Save
        out = Path(config["paths"]["processed"])
        np.save(out / "sleep_X.npy", X)
        np.save(out / "sleep_y.npy", y)
        print(f"\n✅ Saved → {out}/sleep_X.npy  shape={X.shape}")
        print(f"✅ Saved → {out}/sleep_y.npy  shape={y.shape}")
    
    elif args.dataset == "eeg-imagenet":
        print("\n🔄 Preprocessing EEG-ImageNet dataset...")
        X, y = load_eeg_imagenet(config["paths"]["raw_eeg_imagenet"], config)
        
        out = Path(config["paths"]["processed"])
        np.save(out / "dream_X.npy", X)
        np.save(out / "dream_y.npy", y)
        print(f"\n✅ Saved → {out}/dream_X.npy  shape={X.shape}")
        print(f"✅ Saved → {out}/dream_y.npy  shape={y.shape}")


if __name__ == "__main__":
    main()