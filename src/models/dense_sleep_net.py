"""
dense_sleep_net.py
==================
DenseSleepNet: Sleep Stage Classification Model

Architecture Overview:
──────────────────────
Input EEG epoch (30s, 100Hz) → shape: (batch, channels, 3000 samples)

1. Feature Extractor (CNN block)
   - Captures local temporal patterns (spindles, K-complexes, delta waves)
   - Uses progressively larger kernels to capture patterns at multiple scales

2. Dense Connectivity Block
   - Each layer receives features from ALL previous layers (DenseNet-style)
   - Promotes feature reuse and gradient flow → better training

3. Transformer Encoder (Temporal Context)
   - Operates on a SEQUENCE of epochs (e.g. 21 consecutive epochs)
   - Captures long-range dependencies: knowing epoch N-5 was N3 helps classify N

4. Classification Head
   - 5 output classes: Wake (0), N1 (1), N2 (2), N3 (3), REM (4)

References:
  - DenseNet: Huang et al., "Densely Connected Convolutional Networks", CVPR 2017
  - SleepTransformer: Phan et al., "XSleepNet", 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── Building Block: Dense Layer ─────────────────────────────────────────────
class DenseLayer(nn.Module):
    """
    One layer in a Dense Block.
    
    Receives: concatenation of all previous feature maps
    Produces: new_features feature maps (growth_rate)
    
    Architecture: BN → ReLU → Conv → Dropout
    """
    def __init__(self, in_features, growth_rate, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        new_features = self.layers(x)
        # Dense connection: concatenate input and output along channel dim
        return torch.cat([x, new_features], dim=1)


# ─── Dense Block ─────────────────────────────────────────────────────────────
class DenseBlock(nn.Module):
    """
    Stack of DenseLayers with dense connectivity.
    
    After n_layers, the total number of channels is:
        in_channels + n_layers * growth_rate
    """
    def __init__(self, in_channels, growth_rate=16, n_layers=4, dropout=0.2):
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(n_layers):
            layers.append(DenseLayer(current_channels, growth_rate, dropout))
            current_channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = current_channels
    
    def forward(self, x):
        return self.block(x)


# ─── Transition Layer (downsampling between dense blocks) ─────────────────────
class TransitionLayer(nn.Module):
    """Reduce channels and temporal resolution between dense blocks."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2),   # halve temporal resolution
        )
    
    def forward(self, x):
        return self.layers(x)


# ─── CNN Epoch Encoder ────────────────────────────────────────────────────────
class EpochEncoder(nn.Module):
    """
    Encode a single 30-second EEG epoch into a fixed-size feature vector.
    
    Uses multi-scale convolutions to capture:
    - Fine-grained features (kernel=25 → 0.25s at 100Hz)
    - Medium features   (kernel=51 → 0.5s at 100Hz)
    - Coarse features   (kernel=101 → 1s at 100Hz)
    """
    def __init__(self, n_channels, d_model=64):
        super().__init__()
        
        # Multi-scale temporal convolutions
        self.conv_fine   = nn.Conv1d(n_channels, 16, kernel_size=25,  padding=12,  bias=False)
        self.conv_medium = nn.Conv1d(n_channels, 16, kernel_size=51,  padding=25,  bias=False)
        self.conv_coarse = nn.Conv1d(n_channels, 16, kernel_size=101, padding=50, bias=False)
        
        self.bn_init = nn.BatchNorm1d(48)   # 16 + 16 + 16 = 48 channels
        self.pool1   = nn.MaxPool1d(8)       # 3000 → 375 samples
        
        # Dense block for rich feature extraction
        self.dense1  = DenseBlock(48,  growth_rate=16, n_layers=4)   # 48 → 112
        self.trans1  = TransitionLayer(self.dense1.out_channels, 64)  # 112 → 64; 375 → 187
        
        self.dense2  = DenseBlock(64,  growth_rate=16, n_layers=3)   # 64 → 112
        self.trans2  = TransitionLayer(self.dense2.out_channels, 64)  # 112 → 64; 187 → 93
        
        # Global average pooling: collapse temporal dim → (batch, 64)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Project to d_model
        self.proj = nn.Linear(64, d_model)
    
    def forward(self, x):
        # x: (batch, n_channels, 3000)
        
        # Multi-scale feature extraction
        f = torch.cat([
            F.relu(self.conv_fine(x)),
            F.relu(self.conv_medium(x)),
            F.relu(self.conv_coarse(x)),
        ], dim=1)                           # (batch, 48, 3000)
        
        f = self.bn_init(f)
        f = self.pool1(f)                   # (batch, 48, 375)
        
        # Dense feature extraction
        f = self.dense1(f)                  # (batch, 112, 375)
        f = self.trans1(f)                  # (batch, 64,  187)
        f = self.dense2(f)                  # (batch, 112, 187)
        f = self.trans2(f)                  # (batch, 64,  93)
        
        f = self.gap(f).squeeze(-1)         # (batch, 64)
        f = self.proj(f)                    # (batch, d_model)
        
        return f


# ─── Positional Encoding ──────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Add position information to each epoch's embedding.
    
    Without this, the Transformer treats all epochs as interchangeable.
    With positional encoding, it knows epoch 5 comes AFTER epoch 4.
    
    Uses sinusoidal encoding (no learned parameters).
    """
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─── DenseSleepNet: Main Model ────────────────────────────────────────────────
class DenseSleepNet(nn.Module):
    """
    Complete sleep staging model.
    
    Input:  sequence of EEG epochs → (batch, seq_len, n_channels, epoch_samples)
    Output: sleep stage predictions → (batch, seq_len, n_classes)
    
    The model classifies EACH epoch in the sequence, using context from
    neighbouring epochs to resolve ambiguities at stage transitions.
    """
    def __init__(
        self,
        n_channels=2,
        n_classes=5,
        sequence_length=21,
        d_model=64,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.seq_len   = sequence_length
        
        # Per-epoch CNN encoder
        self.epoch_encoder = EpochEncoder(n_channels, d_model)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=sequence_length + 10, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,    # (batch, seq, features) ordering
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head (applied to every position)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_channels, epoch_samples)
        
        Returns:
            logits: (batch, seq_len, n_classes)
        """
        batch, seq_len, n_ch, n_t = x.shape
        
        # Encode each epoch independently
        # Reshape to (batch*seq_len, n_channels, n_t)
        x_flat = x.view(batch * seq_len, n_ch, n_t)
        features = self.epoch_encoder(x_flat)           # (batch*seq_len, d_model)
        features = features.view(batch, seq_len, -1)    # (batch, seq_len, d_model)
        
        # Add positional encoding
        features = self.pos_enc(features)
        
        # Transformer: attend across the sequence
        context = self.transformer(features)             # (batch, seq_len, d_model)
        
        # Classify each epoch
        logits = self.classifier(context)                # (batch, seq_len, n_classes)
        
        return logits
    
    def predict_single(self, epoch):
        """
        Convenience method: classify a single epoch with no sequence context.
        
        Args:
            epoch: (n_channels, epoch_samples) numpy array
        
        Returns:
            stage: int (0–4)
            probabilities: numpy array of shape (n_classes,)
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(epoch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # (1, 1, n_channels, epoch_samples)
            logits = self.forward(x)          # (1, 1, n_classes)
            probs  = F.softmax(logits[0, 0], dim=-1).numpy()
            stage  = int(probs.argmax())
        return stage, probs


# ─── Model Factory ────────────────────────────────────────────────────────────
def build_sleep_model(config):
    """Build DenseSleepNet from config dict."""
    cfg = config["sleep_model"]
    return DenseSleepNet(
        n_channels      = cfg["n_channels"],
        n_classes       = cfg["n_classes"],
        sequence_length = cfg["sequence_length"],
        d_model         = cfg["d_model"],
        n_heads         = cfg["n_heads"],
        n_layers        = cfg["n_layers"],
        dropout         = cfg["dropout"],
    )


if __name__ == "__main__":
    # Quick sanity check
    model = DenseSleepNet()
    
    # Simulate a batch of 4, sequence of 21 epochs, 2 channels, 3000 samples
    x = torch.randn(4, 21, 2, 3000)
    out = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")   # Expected: (4, 21, 5)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")