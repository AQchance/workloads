"""
Transformer V2: Overlap-Gated + Resource-Aware Attention.

Key improvements over V1:
  1. Dynamic batch masking (no fixed MAX_NODES padding)
  2. Overlap-gated attention: zero overlap → -inf attention weight
  3. Resource-aware attention bias from pair features
  4. Time-relative positional encoding
  5. Per-query MLP before cross-query attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResourceAttentionBias(nn.Module):
    """Compute attention bias from pre-computed pair features (8-dim)."""
    def __init__(self, pair_dim=8, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, pair_features):
        """pair_features: [B,N,N,8] — elapsed, is_before, overlap, conflict(5)"""
        return self.mlp(pair_features).squeeze(-1)  # [B,N,N]


class TimeRelativePE(nn.Module):
    """Time-relative positional encoding based on start time differences."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, start_times):
        """start_times: [B, N] — actual start timestamps in seconds."""
        B, N = start_times.shape
        # Relative time differences
        rel_time = start_times.unsqueeze(1) - start_times.unsqueeze(2)  # [B,N,N]
        # Encode as multiple frequencies
        div = torch.exp(torch.arange(0, self.d_model, 2, device=start_times.device).float()
                        * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(B, N, self.d_model, device=start_times.device)
        # Mean relative time per position
        mean_rel = rel_time.mean(dim=-1)  # [B,N]
        for k in range(self.d_model // 2):
            pe[:, :, 2 * k] = torch.sin(mean_rel * div[k])
            pe[:, :, 2 * k + 1] = torch.cos(mean_rel * div[k])
        return pe


class OverlapGatedTransformer(nn.Module):
    def __init__(self, node_dim=134, pair_dim=8, d_model=256, n_heads=4,
                 n_layers=3, dropout=0.2):
        super().__init__()

        # Per-query feature encoding
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, d_model), nn.ReLU(), nn.Dropout(dropout))

        # Time-relative position encoding
        self.time_pe = TimeRelativePE(d_model)

        # Resource-aware attention bias
        self.attn_bias = ResourceAttentionBias(pair_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1))

        # Resource-conditioned bias
        self.res_bias = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, node_features, pair_features, start_times, padding_mask, target_idx=0):
        """
        Args:
            node_features: [B, N, 134] — per-query features
            pair_features:  [B, N, N, 8] — per-pair (elapsed, is_before, overlap, 5-conflict)
            start_times:    [B, N] — actual start timestamps
            padding_mask:   [B, N] — True for padding positions
            target_idx:     which query is the prediction target (default 0)
        Returns:
            pred: [B] — slowdown ratio predictions
        """
        B, N, _ = node_features.shape

        # 1. Per-query encoding
        h = self.node_mlp(node_features)  # [B, N, d_model]

        # 2. Time-relative position encoding
        h = h + self.time_pe(start_times)

        # 3. Resource-aware attention bias from pair features
        bias = self.attn_bias(pair_features)  # [B,N,N]

        # 4. Overlap gating: zero overlap → very negative (not -inf for stability)
        NEG = -1e9
        overlap = pair_features[:, :, :, 2]
        bias = torch.where(overlap < 0.01, torch.full_like(bias, NEG), bias)

        # 5. Manual layer pass with additive bias
        n_heads = self.encoder.layers[0].self_attn.num_heads
        # Convert padding_mask to float for compatibility with attn_mask
        pad_mask = padding_mask  # keep as bool, works with recent PyTorch
        for layer in self.encoder.layers:
            attn_bias = bias.unsqueeze(1).expand(-1, n_heads, -1, -1).reshape(B * n_heads, N, N)
            h2 = layer.self_attn(h, h, h, attn_mask=attn_bias,
                                 key_padding_mask=padding_mask, need_weights=False)[0]
            h = h + layer.dropout1(h2)
            h = layer.norm1(h)
            h2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = h + layer.dropout2(h2)
            h = layer.norm2(h)

        # 6. Extract target query output
        h_target = h[:, target_idx, :]  # [B, d_model]

        # 7. Prediction
        base_pred = self.predictor(h_target).squeeze(-1)

        # 8. Resource bias
        res_self = node_features[:, target_idx, 128:133]
        res_mean = node_features[:, :, 128:133].mean(dim=1)
        res_bias = self.res_bias(torch.cat([res_self, res_mean], dim=-1)).squeeze(-1)

        return base_pred + res_bias
