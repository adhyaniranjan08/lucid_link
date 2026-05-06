# 🧠 EEG Dream Detection & Classification System

A complete end-to-end pipeline that detects REM sleep from EEG signals and classifies dream content using deep learning.

## 📁 Project Structure

```
eeg_dream_detection/
├── configs/                    # Configuration files
│   └── config.yaml
├── data/
│   ├── raw/                    # Downloaded datasets go here
│   └── processed/              # Preprocessed .npy files
├── models/
│   ├── saved/                  # Final trained models (.pt files)
│   └── checkpoints/            # Training checkpoints
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py       # Bandpass filter, artifact removal, epoching
│   ├── models/
│   │   ├── dense_sleep_net.py  # Sleep stage classifier (Awake/N1/N2/N3/REM)
│   │   └── eegnet.py           # Dream content classifier
│   ├── training/
│   │   ├── train_sleep.py      # Train DenseSleepNet
│   │   └── train_dream.py      # Train EEGNet
│   ├── inference/
│   │   └── pipeline.py         # Combined inference pipeline
│   └── dashboard/
│       └── app.py              # Streamlit dashboard
├── notebooks/
│   └── exploration.ipynb       # EDA notebook
├── tests/
│   └── test_pipeline.py        # Quick sanity-check tests
├── requirements.txt
└── README.md
```

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets

**Sleep-EDF (Sleep Stage Detection):**
```bash
# Option A: Use MNE (recommended for beginners)
python src/preprocessing/download_data.py --dataset sleep-edf

# Option B: Manual download
# Visit: https://physionet.org/content/sleep-edfx/1.0.0/
# Download any SC*-PSG.edf and SC*-Hypnogram.edf pair
# Place in data/raw/sleep_edf/
```

**EEG-ImageNet (Dream Classification):**
```bash
# Visit: https://github.com/perceivelab/eeg_visual_classification
# Download the EEG dataset and place in data/raw/eeg_imagenet/
# OR use the synthetic generator (no download needed):
python src/preprocessing/generate_synthetic.py
```

### 3. Preprocess data
```bash
python src/preprocessing/preprocess.py --dataset sleep-edf
python src/preprocessing/preprocess.py --dataset eeg-imagenet
```

### 4. Train models
```bash
# Train sleep stage classifier (~10-20 min on CPU)
python src/training/train_sleep.py

# Train dream classifier (~5-10 min on CPU)
python src/training/train_dream.py
```

### 5. Run inference
```bash
python src/inference/pipeline.py --input data/processed/sample_eeg.npy
```

### 6. Launch dashboard
```bash
streamlit run src/dashboard/app.py
```

## 🧪 Test with synthetic data (no download needed)
```bash
python tests/test_pipeline.py
```

## 🏗️ Model Architecture

| Model | Task | Input | Output |
|-------|------|-------|--------|
| DenseSleepNet | Sleep staging | EEG epoch (30s, 100Hz) | 5 classes |
| EEGNet | Dream content | EEG epoch (1s, 128Hz) | N categories |

## 💻 Hardware Requirements
- RAM: 8 GB minimum
- CPU: Any modern multi-core processor
- GPU: Optional (CUDA supported but not required)
- Storage: ~2 GB for datasets