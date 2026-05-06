"""
eegnet.py
=========
EEGNet: Dream Content Classification from EEG

Architecture Overview:
──────────────────────
EEGNet is a compact CNN designed specifically for EEG-based BCI tasks.
It was proposed by Lawhern et al. (2018) and is popular because:
  - Very few parameters (hundreds to thousands, not millions)
  - Generalises well across subjects and datasets
  - Interpretable: filters correspond to known EEG features

Layers:
  1. Temporal Convolution (Conv1)
     - Learns frequency-specific filters across time
     - Equivalent to bandpass filtering
  
  2. Depthwise Spatial Convolution (Conv2)
     - Learns spatial weights ACROSS EEG CHANNELS
     - Each filter is channel-specific (depthwise = no cross-channel mixing)
     - Equivalent to learning optimal electrode combinations
  
  3. Separable Convolution (Conv3)
     - Depthwise: per-channel temporal summary
     - Pointwise: mix information across channels
  
  4. Classification Head
     - Flatten → Dropout → Linear → n_classes

Reference:
  Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for
  EEG-based Brain-Computer Interfaces", J. Neural Engineering, 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EEGNet(nn.Module):
    """
    EEGNet for dream content classification.
    
    Input:  EEG epoch  → (batch, n_channels, temporal_length)
    Output: class logits → (batch, n_classes)
    
    Args:
        n_channels:      number of EEG electrodes (e.g. 14)
        n_classes:       number of output classes (e.g. 6 visual categories)
        temporal_length: number of time samples (e.g. 128 for 1s at 128Hz)
        F1:              number of temporal filters (default 8)
        D:               depth multiplier for spatial filters (default 2)
                         total spatial filters = F1 * D
        F2:              number of pointwise filters (default 16 = F1*D)
        kernel_length:   size of temporal convolution kernel (default 64 = 0.5s)
        dropout:         dropout rate (default 0.5 — EEGNet uses aggressive dropout)
    """
    def __init__(
        self,
        n_channels=14,
        n_classes=6,
        temporal_length=128,
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        dropout=0.5,
    ):
        super().__init__()
        
        self.n_channels      = n_channels
        self.n_classes       = n_classes
        self.temporal_length = temporal_length
        
        # ── Block 1: Temporal + Spatial Convolution ──────────────────────────
        # Step 1a: Temporal filter
        # Input: (batch, 1, n_channels, T)  ← treat EEG as 2D "image"
        # Output: (batch, F1, n_channels, T)
        self.conv1 = nn.Conv2d(
            in_channels  = 1,
            out_channels = F1,
            kernel_size  = (1, kernel_length),
            padding      = (0, kernel_length // 2),
            bias         = False,
        )
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Step 1b: Depthwise spatial filter (per temporal filter, learn channel weights)
        # kernel_size = (n_channels, 1) → convolves over ALL channels
        # groups = F1 → each temporal filter gets its own spatial filter (depthwise)
        self.conv2 = nn.Conv2d(
            in_channels  = F1,
            out_channels = F1 * D,           # D spatial filters per temporal filter
            kernel_size  = (n_channels, 1),  # full spatial extent
            groups       = F1,               # depthwise convolution
            bias         = False,
        )
        self.bn2      = nn.BatchNorm2d(F1 * D)
        self.act1     = nn.ELU()             # ELU is preferred over ReLU for EEG
        self.pool1    = nn.AvgPool2d((1, 4)) # temporal downsampling ×4
        self.dropout1 = nn.Dropout(dropout)
        
        # ── Block 2: Separable Convolution ───────────────────────────────────
        # Depthwise: each channel independently
        self.conv3 = nn.Conv2d(
            in_channels  = F1 * D,
            out_channels = F1 * D,           # same number of channels
            kernel_size  = (1, 16),
            padding      = (0, 8),
            groups       = F1 * D,           # fully depthwise
            bias         = False,
        )
        # Pointwise: mix channels
        self.conv4 = nn.Conv2d(
            in_channels  = F1 * D,
            out_channels = F2,
            kernel_size  = (1, 1),
            bias         = False,
        )
        self.bn3      = nn.BatchNorm2d(F2)
        self.act2     = nn.ELU()
        self.pool2    = nn.AvgPool2d((1, 8)) # temporal downsampling ×8
        self.dropout2 = nn.Dropout(dropout)
        
        # ── Classification Head ───────────────────────────────────────────────
        # Compute the size of the flattened feature vector
        # After pool1: T/4, after pool2: T/4/8 = T/32
        # Spatial dim collapses to 1 after conv2 (n_channels → 1)
        self._feature_size = self._compute_feature_size()
        
        self.classifier = nn.Sequential(
            nn.Linear(self._feature_size, n_classes),
        )
    
    def _compute_feature_size(self):
        """Dry-run a dummy input to compute the flattened size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.temporal_length)
            out = self._forward_features(dummy)
            return out.shape[1]
    
    def _forward_features(self, x):
        """Feature extraction part (everything before classifier)."""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.flatten(start_dim=1)
        return x
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, temporal_length)  ← standard EEG format
        
        Returns:
            logits: (batch, n_classes)
        """
        # Add singleton channel dim: (batch, 1, n_channels, T)
        x = x.unsqueeze(1)
        
        # Extract features
        x = self._forward_features(x)
        
        # Classify
        logits = self.classifier(x)
        return logits
    
    def predict(self, epoch):
        """
        Classify a single EEG epoch.
        
        Args:
            epoch: numpy array of shape (n_channels, temporal_length)
        
        Returns:
            category:      int - predicted class index
            probabilities: numpy array of shape (n_classes,)
            confidence:    float - max probability
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(epoch, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(x)
            probs  = F.softmax(logits[0], dim=-1).numpy()
        
        category   = int(probs.argmax())
        confidence = float(probs.max())
        return category, probs, confidence


# ─── Improved EEGNet with attention ──────────────────────────────────────────
class EEGNetWithAttention(EEGNet):
    """
    EEGNet enhanced with a channel attention mechanism.
    
    Before the spatial convolution, learn which EEG channels to
    pay more attention to. This is useful when some channels are
    more informative than others (e.g. occipital channels for visual).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Channel attention: learn scalar weight per EEG channel
        self.channel_attention = nn.Sequential(
            nn.Linear(self.n_channels, self.n_channels // 2),
            nn.ReLU(),
            nn.Linear(self.n_channels // 2, self.n_channels),
            nn.Sigmoid(),   # output in (0, 1)
        )
    
    def forward(self, x):
        # x: (batch, n_channels, T)
        
        # Compute channel attention weights using global temporal average
        avg_power = x.mean(dim=-1)                              # (batch, n_channels)
        attn = self.channel_attention(avg_power)                # (batch, n_channels)
        x = x * attn.unsqueeze(-1)                              # scale each channel
        
        # Rest of EEGNet
        x = x.unsqueeze(1)
        x = self._forward_features(x)
        return self.classifier(x)


# ─── Model Factory ────────────────────────────────────────────────────────────
def build_dream_model(config, use_attention=False):
    """Build EEGNet from config dict."""
    cfg = config["dream_model"]
    
    ModelClass = EEGNetWithAttention if use_attention else EEGNet
    
    return ModelClass(
        n_channels      = cfg["n_channels"],
        n_classes       = cfg["n_classes"],
        temporal_length = cfg["temporal_length"],
        F1              = cfg["F1"],
        D               = cfg["D"],
        F2              = cfg["F2"],
        kernel_length   = cfg["kernel_length"],
        dropout         = cfg["dropout"],
    )


if __name__ == "__main__":
    # Quick sanity check
    model = EEGNet(n_channels=14, n_classes=6, temporal_length=128)
    
    # Simulate a batch of 16 EEG epochs
    x = torch.randn(16, 14, 128)
    out = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")    # Expected: (16, 6)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")   # Typically ~2,000–5,000
    
    # Test predict method
    epoch = np.random.randn(14, 128).astype(np.float32)
    cat, probs, conf = model.predict(epoch)
    print(f"Predicted category: {cat}, Confidence: {conf:.2%}")