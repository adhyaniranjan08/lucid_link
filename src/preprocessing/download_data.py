"""
download_data.py
================
Automated Dataset Downloader for EEG Dream Detection System

Supports:
  - Sleep-EDF (Physionet) — sleep stage classification
  - EEG-ImageNet (GitHub)  — dream content classification

Usage:
  python src/preprocessing/download_data.py --dataset sleep-edf
  python src/preprocessing/download_data.py --dataset eeg-imagenet
  python src/preprocessing/download_data.py --dataset both
  python src/preprocessing/download_data.py --dataset sleep-edf --n_files 4
  python src/preprocessing/download_data.py --dataset sleep-edf --method mne
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
import zipfile
import tarfile
import shutil
import hashlib
import time
from pathlib import Path

import yaml

# ── Config loader ─────────────────────────────────────────────────────────────
def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Progress bar for downloads ────────────────────────────────────────────────
class DownloadProgress:
    """Simple terminal progress bar for urllib downloads."""

    def __init__(self, filename):
        self.filename  = filename
        self.last_pct  = -1
        self.start     = time.time()

    def __call__(self, block_num, block_size, total_size):
        if total_size <= 0:
            downloaded = block_num * block_size
            print(f"\r  Downloading {self.filename}: {downloaded/1024:.0f} KB...", end="")
            return

        downloaded = min(block_num * block_size, total_size)
        pct = int(100 * downloaded / total_size)

        if pct != self.last_pct:
            self.last_pct = pct
            bar_len  = 35
            filled   = int(bar_len * downloaded / total_size)
            bar      = "█" * filled + "░" * (bar_len - filled)
            elapsed  = time.time() - self.start
            speed_kb = (downloaded / 1024) / max(elapsed, 0.1)
            eta      = (total_size - downloaded) / max(speed_kb * 1024, 1)
            print(
                f"\r  [{bar}] {pct:>3}%  "
                f"{downloaded/1e6:.1f}/{total_size/1e6:.1f} MB  "
                f"{speed_kb:.0f} KB/s  ETA {eta:.0f}s   ",
                end="",
                flush=True,
            )
        if downloaded >= total_size:
            print()   # newline when done


def download_file(url, dest_path, label=None):
    """
    Download a single file with progress display.

    Args:
        url:       remote URL to download
        dest_path: local Path to save to
        label:     short name shown in the progress bar

    Returns:
        True on success, False on failure
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"  ⏭  Already exists: {dest_path.name} — skipping")
        return True

    filename = label or dest_path.name
    print(f"  ⬇  {filename}")
    print(f"     URL: {url}")

    try:
        urllib.request.urlretrieve(url, dest_path, DownloadProgress(filename))
        size_mb = dest_path.stat().st_size / 1e6
        print(f"  ✅ Saved → {dest_path}  ({size_mb:.1f} MB)")
        return True

    except urllib.error.HTTPError as e:
        print(f"\n  ❌ HTTP {e.code}: {e.reason}  — {url}")
        if dest_path.exists():
            dest_path.unlink()
        return False

    except urllib.error.URLError as e:
        print(f"\n  ❌ Network error: {e.reason}")
        if dest_path.exists():
            dest_path.unlink()
        return False

    except KeyboardInterrupt:
        print("\n  ⚠️  Interrupted by user")
        if dest_path.exists():
            dest_path.unlink()
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  SLEEP-EDF DOWNLOADER
# ══════════════════════════════════════════════════════════════════════════════

# 20 subjects from Sleep-EDF Cassette (SC) study.
# Each subject has 2 nights → 2 PSG files + 2 Hypnogram files.
# We use only night 1 (E0) to keep downloads manageable.
SLEEP_EDF_BASE = "https://physionet.org/files/sleep-edfx/1.0.0"

# (subject_id, psg_filename, hypnogram_filename, psg_sha256_first8)
SLEEP_EDF_FILES = [
    ("SC4001", "SC4001E0-PSG.edf",  "SC4001EC-Hypnogram.edf"),
    ("SC4002", "SC4002E0-PSG.edf",  "SC4002EC-Hypnogram.edf"),
    ("SC4003", "SC4003E0-PSG.edf",  "SC4003EC-Hypnogram.edf"),
    ("SC4004", "SC4004E0-PSG.edf",  "SC4004EC-Hypnogram.edf"),
    ("SC4005", "SC4005E0-PSG.edf",  "SC4005EC-Hypnogram.edf"),
    ("SC4006", "SC4006E0-PSG.edf",  "SC4006EC-Hypnogram.edf"),
    ("SC4007", "SC4007E0-PSG.edf",  "SC4007EC-Hypnogram.edf"),
    ("SC4008", "SC4008E0-PSG.edf",  "SC4008EC-Hypnogram.edf"),
    ("SC4011", "SC4011E0-PSG.edf",  "SC4011EC-Hypnogram.edf"),
    ("SC4012", "SC4012E0-PSG.edf",  "SC4012EC-Hypnogram.edf"),
    ("SC4013", "SC4013E0-PSG.edf",  "SC4013EC-Hypnogram.edf"),
    ("SC4014", "SC4014E0-PSG.edf",  "SC4014EC-Hypnogram.edf"),
    ("SC4021", "SC4021E0-PSG.edf",  "SC4021EC-Hypnogram.edf"),
    ("SC4022", "SC4022E0-PSG.edf",  "SC4022EC-Hypnogram.edf"),
    ("SC4031", "SC4031E0-PSG.edf",  "SC4031EC-Hypnogram.edf"),
    ("SC4032", "SC4032E0-PSG.edf",  "SC4032EC-Hypnogram.edf"),
    ("SC4041", "SC4041E0-PSG.edf",  "SC4041EC-Hypnogram.edf"),
    ("SC4042", "SC4042E0-PSG.edf",  "SC4042EC-Hypnogram.edf"),
    ("SC4051", "SC4051E0-PSG.edf",  "SC4051EC-Hypnogram.edf"),
    ("SC4052", "SC4052E0-PSG.edf",  "SC4052EC-Hypnogram.edf"),
]


def download_sleep_edf_physionet(out_dir, n_files=2):
    """
    Download Sleep-EDF directly from PhysioNet (no login required).

    Args:
        out_dir:  destination folder
        n_files:  number of subjects to download (1–20).
                  Each subject = ~50 MB. Start with 2 for a quick test.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = SLEEP_EDF_FILES[:n_files]
    total_files = len(subjects) * 2    # PSG + Hypnogram per subject

    print(f"\n📥 Downloading {n_files} Sleep-EDF subject(s) ({total_files} files)")
    print(f"   Destination: {out_dir.resolve()}")
    print(f"   Estimated size: ~{n_files * 50:.0f} MB\n")

    ok_count   = 0
    fail_count = 0

    for subj_id, psg_file, hyp_file in subjects:
        print(f"\n  👤 Subject {subj_id}")

        # PSG recording
        psg_url  = f"{SLEEP_EDF_BASE}/sleep-cassette/{psg_file}"
        psg_dest = out_dir / psg_file
        if download_file(psg_url, psg_dest, psg_file):
            ok_count += 1
        else:
            fail_count += 1

        # Hypnogram annotations
        hyp_url  = f"{SLEEP_EDF_BASE}/sleep-cassette/{hyp_file}"
        hyp_dest = out_dir / hyp_file
        if download_file(hyp_url, hyp_dest, hyp_file):
            ok_count += 1
        else:
            fail_count += 1

    print(f"\n{'═'*55}")
    print(f"  Sleep-EDF download complete:")
    print(f"  ✅ {ok_count} files downloaded successfully")
    if fail_count:
        print(f"  ❌ {fail_count} files failed")
        print(f"\n  💡 If downloads fail, try manual download:")
        print(f"     1. Go to https://physionet.org/content/sleep-edfx/1.0.0/")
        print(f"     2. Click 'Files' tab → sleep-cassette/")
        print(f"     3. Download any *-PSG.edf and matching *-Hypnogram.edf")
        print(f"     4. Place them in: {out_dir.resolve()}")
    print(f"{'═'*55}")

    return fail_count == 0


def download_sleep_edf_mne(out_dir, n_files=2):
    """
    Download Sleep-EDF using the MNE library (easier, handles auth automatically).

    MNE's built-in downloader fetches from PhysioNet with proper headers.

    Args:
        out_dir:  destination folder
        n_files:  number of subjects (passed to MNE as tetrode_recording count)
    """
    try:
        import mne
    except ImportError:
        print("❌ MNE not installed. Run: pip install mne")
        return False

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Downloading Sleep-EDF via MNE ({n_files} subject(s))...")
    print(f"   Destination: {out_dir.resolve()}\n")

    try:
        # MNE downloads to its own data dir; we'll symlink/copy to ours
        paths = mne.datasets.sleep_physionet.age.fetch_data(
            subjects=list(range(n_files)),
            recording=[1],           # night 1 only
            path=str(out_dir),
            update_path=False,
            verbose=True,
        )

        print(f"\n✅ MNE downloaded {len(paths)} file pairs to: {out_dir}")

        # Flatten nested dirs if MNE put them in subdirs
        for p in Path(out_dir).rglob("*.edf"):
            dest = out_dir / p.name
            if p != dest and not dest.exists():
                shutil.copy2(p, dest)
                print(f"  Copied {p.name} → {out_dir}/")

        return True

    except Exception as e:
        print(f"❌ MNE download failed: {e}")
        print("   Falling back to direct PhysioNet download...")
        return download_sleep_edf_physionet(out_dir, n_files)


# ══════════════════════════════════════════════════════════════════════════════
#  EEG-IMAGENET DOWNLOADER
# ══════════════════════════════════════════════════════════════════════════════

# The EEG-ImageNet dataset is hosted on GitHub (perceivelab/eeg_visual_classification)
# Primary file: preprocessed EEG signals (~500 MB for full dataset)
EEG_IMAGENET_URLS = [
    # Primary: GitHub releases page (preprocessed, ready to use)
    {
        "url": "https://github.com/perceivelab/eeg_visual_classification/releases/download/v1.0/eeg_signals_raw_with_mean_std.pth",
        "filename": "eeg_signals_raw_with_mean_std.pth",
        "size_mb": 490,
        "description": "Full EEG-ImageNet dataset (preprocessed)",
    },
    # Smaller split files if the full one fails
    {
        "url": "https://github.com/perceivelab/eeg_visual_classification/raw/main/data/block_splits_by_image_all.pth",
        "filename": "block_splits_by_image_all.pth",
        "size_mb": 1,
        "description": "Train/test split indices",
    },
]


def download_eeg_imagenet(out_dir):
    """
    Download the EEG-ImageNet dataset from GitHub.

    Falls back to generating a synthetic version if download fails.

    Args:
        out_dir: destination folder
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Downloading EEG-ImageNet dataset...")
    print(f"   Destination: {out_dir.resolve()}")
    print(f"   Note: Full dataset is ~490 MB. This may take a while.\n")

    any_ok = False
    for entry in EEG_IMAGENET_URLS:
        dest = out_dir / entry["filename"]
        print(f"\n  📦 {entry['description']} (~{entry['size_mb']} MB)")
        ok = download_file(entry["url"], dest, entry["filename"])
        if ok:
            any_ok = True

    if not any_ok:
        print(f"\n⚠️  All downloads failed.")
        print("   Manual download instructions:")
        print("   1. Go to: https://github.com/perceivelab/eeg_visual_classification")
        print("   2. Click 'Releases' → download eeg_signals_raw_with_mean_std.pth")
        print(f"   3. Place in: {out_dir.resolve()}")
        print("\n   Alternatively, generate synthetic data (works immediately):")
        print("   python src/preprocessing/generate_synthetic.py")
        return False

    print(f"\n✅ EEG-ImageNet download complete → {out_dir}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_sleep_edf(data_dir):
    """Check that Sleep-EDF files are present and readable."""
    data_dir = Path(data_dir)
    psg_files = list(data_dir.glob("*-PSG.edf"))
    hyp_files = list(data_dir.glob("*-Hypnogram.edf"))

    print(f"\n🔍 Verifying Sleep-EDF in {data_dir}:")
    print(f"   PSG files:        {len(psg_files)}")
    print(f"   Hypnogram files:  {len(hyp_files)}")

    if not psg_files:
        print("   ❌ No PSG files found!")
        return False

    # Try to open the first file with MNE (lightweight check)
    try:
        import mne
        raw = mne.io.read_raw_edf(str(psg_files[0]), preload=False, verbose=False)
        duration_h = raw.n_times / raw.info["sfreq"] / 3600
        print(f"   ✅ First file OK: {psg_files[0].name}")
        print(f"      Channels: {raw.ch_names}")
        print(f"      Duration: {duration_h:.1f} hours @ {raw.info['sfreq']} Hz")
        return True
    except ImportError:
        # MNE not installed — just check file sizes
        for f in psg_files[:2]:
            size_mb = f.stat().st_size / 1e6
            if size_mb < 1:
                print(f"   ⚠️  {f.name} looks too small ({size_mb:.1f} MB) — may be corrupt")
            else:
                print(f"   ✅ {f.name}: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"   ⚠️  Could not open EDF: {e}")
        return False


def verify_eeg_imagenet(data_dir):
    """Check that EEG-ImageNet files are present."""
    data_dir = Path(data_dir)
    pth_files = list(data_dir.glob("*.pth"))
    npy_files = list(data_dir.glob("*.npy"))

    print(f"\n🔍 Verifying EEG-ImageNet in {data_dir}:")
    print(f"   .pth files: {len(pth_files)}")
    print(f"   .npy files: {len(npy_files)}")

    if not pth_files and not npy_files:
        print("   ❌ No data files found!")
        return False

    # Try loading
    for f in pth_files[:1]:
        size_mb = f.stat().st_size / 1e6
        print(f"   ✅ {f.name}: {size_mb:.1f} MB")
        try:
            import torch
            data = torch.load(str(f), map_location="cpu")
            if isinstance(data, dict) and "dataset" in data:
                print(f"      Samples: {len(data['dataset'])}")
        except Exception:
            pass

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK-START: Generate synthetic if download fails
# ══════════════════════════════════════════════════════════════════════════════

def offer_synthetic_fallback(dataset):
    """If real data download failed, offer to generate synthetic data."""
    print(f"\n{'─'*55}")
    print("  💡 FALLBACK: Use synthetic data instead?")
    print("     Synthetic data mimics real EEG patterns and lets you")
    print("     train + test the full pipeline without any downloads.")
    print(f"{'─'*55}")

    try:
        answer = input("  Generate synthetic data now? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer in ("", "y", "yes"):
        print("\n  Running: python src/preprocessing/generate_synthetic.py")
        import subprocess
        result = subprocess.run(
            [sys.executable, "src/preprocessing/generate_synthetic.py"],
            capture_output=False
        )
        if result.returncode == 0:
            print("\n  ✅ Synthetic data ready! You can now run:")
            print("     python src/training/train_sleep.py")
            if dataset in ("eeg-imagenet", "both"):
                print("     python src/training/train_dream.py")
        else:
            print("  ❌ Synthetic generation failed. Check errors above.")
    else:
        print("\n  OK. You can generate it later with:")
        print("     python src/preprocessing/generate_synthetic.py")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download EEG datasets for the Dream Detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 2 Sleep-EDF subjects (quickest, ~100 MB)
  python src/preprocessing/download_data.py --dataset sleep-edf

  # Download more subjects for better training
  python src/preprocessing/download_data.py --dataset sleep-edf --n_files 10

  # Use MNE's built-in downloader (more reliable)
  python src/preprocessing/download_data.py --dataset sleep-edf --method mne

  # Download EEG-ImageNet (~490 MB)
  python src/preprocessing/download_data.py --dataset eeg-imagenet

  # Download both datasets
  python src/preprocessing/download_data.py --dataset both

  # Skip downloads entirely — use synthetic data
  python src/preprocessing/generate_synthetic.py
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=["sleep-edf", "eeg-imagenet", "both"],
        default="sleep-edf",
        help="Which dataset to download (default: sleep-edf)",
    )
    parser.add_argument(
        "--n_files",
        type=int,
        default=2,
        help="Number of Sleep-EDF subjects to download (1–20, default: 2). "
             "Each subject ≈ 50 MB.",
    )
    parser.add_argument(
        "--method",
        choices=["direct", "mne"],
        default="direct",
        help="Download method for Sleep-EDF: 'direct' (urllib) or 'mne' (MNE library).",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing files, do not download.",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"❌ Config not found: {args.config}")
        print("   Make sure you run this from the project root directory.")
        sys.exit(1)

    sleep_dir   = Path(config["paths"]["raw_sleep_edf"])
    imagenet_dir = Path(config["paths"]["raw_eeg_imagenet"])

    print("\n" + "═"*55)
    print("  🧠 EEG DREAM DETECTION — Dataset Downloader")
    print("═"*55)
    print(f"  Dataset:  {args.dataset}")
    if args.dataset in ("sleep-edf", "both"):
        print(f"  Subjects: {args.n_files}  (~{args.n_files * 50} MB)")
        print(f"  Method:   {args.method}")
    print("═"*55)

    # ── Verify-only mode ──────────────────────────────────────────────────────
    if args.verify_only:
        if args.dataset in ("sleep-edf", "both"):
            verify_sleep_edf(sleep_dir)
        if args.dataset in ("eeg-imagenet", "both"):
            verify_eeg_imagenet(imagenet_dir)
        return

    # ── Download Sleep-EDF ────────────────────────────────────────────────────
    sleep_ok = True
    if args.dataset in ("sleep-edf", "both"):
        if args.method == "mne":
            sleep_ok = download_sleep_edf_mne(sleep_dir, n_files=args.n_files)
        else:
            sleep_ok = download_sleep_edf_physionet(sleep_dir, n_files=args.n_files)

        if sleep_ok:
            verify_sleep_edf(sleep_dir)
            print(f"\n✅ Next step — preprocess the data:")
            print(f"   python src/preprocessing/preprocess.py --dataset sleep-edf")
        else:
            offer_synthetic_fallback("sleep-edf")

    # ── Download EEG-ImageNet ─────────────────────────────────────────────────
    imagenet_ok = True
    if args.dataset in ("eeg-imagenet", "both"):
        imagenet_ok = download_eeg_imagenet(imagenet_dir)

        if imagenet_ok:
            verify_eeg_imagenet(imagenet_dir)
            print(f"\n✅ Next step — preprocess the data:")
            print(f"   python src/preprocessing/preprocess.py --dataset eeg-imagenet")
        else:
            offer_synthetic_fallback("eeg-imagenet")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  📋 SUMMARY")
    print("═"*55)
    if args.dataset in ("sleep-edf", "both"):
        status = "✅ Ready" if sleep_ok else "❌ Failed (use synthetic)"
        print(f"  Sleep-EDF:     {status}")
        print(f"  Location:      {sleep_dir.resolve()}")
    if args.dataset in ("eeg-imagenet", "both"):
        status = "✅ Ready" if imagenet_ok else "❌ Failed (use synthetic)"
        print(f"  EEG-ImageNet:  {status}")
        print(f"  Location:      {imagenet_dir.resolve()}")

    print("\n  Full workflow:")
    print("  1. python src/preprocessing/download_data.py --dataset sleep-edf")
    print("  2. python src/preprocessing/preprocess.py --dataset sleep-edf")
    print("  3. python src/training/train_sleep.py")
    print("  4. python src/training/train_dream.py")
    print("  5. streamlit run src/dashboard/app.py")
    print("═"*55 + "\n")


if __name__ == "__main__":
    main()