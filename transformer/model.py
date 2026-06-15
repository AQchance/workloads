"""
Concurrent Query Transformer with Resource-Aware Attention.

Each concurrent query set is processed as a fully-connected graph:
  - Nodes: queries (plan_emb + resources + serial_lat)
  - Edges: pair features (elapsed, is_before, overlap, resource_conflict)
  - Attention bias: edge features → scalar bias injected into self-attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Time-based positional encoding using actual start timestamps."""
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, timestamps):
        """timestamps: [B, N] — actual start times in seconds (can be float)"""
        idx = timestamps.long().clamp(0, self.pe.size(0) - 1)
        return self.pe[idx]  # [B, N, d_model]


class ResourceAttentionBias(nn.Module):
    """Compute per-pair attention bias from resource profiles and timing."""
    def __init__(self, pair_dim=8, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, pair_features):
        """pair_features: [B, N, N, pair_dim] → bias: [B, N, N]"""
        return self.mlp(pair_features).squeeze(-1)


class ConcurrentQueryTransformer(nn.Module):
    def __init__(self, node_dim=134, pair_dim=8, d_model=256, n_heads=4,
                 n_layers=3, dropout=0.2):
        super().__init__()

        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, d_model), nn.ReLU(), nn.Dropout(dropout))

        # Time-based position encoding
        self.pos_enc = PositionalEncoding(d_model)

        # Pairwise attention bias
        self.attn_bias = ResourceAttentionBias(pair_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1))

        # Resource-conditioned bias (from ResourceFull GNN)
        self.res_bias = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, node_features, pair_features, start_times, res_profile,
                target_mask=None, src_key_padding_mask=None):
        """
        Args:
            node_features: [B, N, 134] — per-query features
            pair_features:  [B, N, N, 8] — per-pair interaction features
            start_times:    [B, N] — actual start timestamps
            res_profile:    [B, N, 5] — GNN resource predictions
            target_mask:    [B, N] — which query is the prediction target
            src_key_padding_mask: [B, N] — mask for padding (non-existent queries)
        Returns:
            pred: [B] — slowdown ratio predictions for target queries
        """
        B, N, _ = node_features.shape

        # Node embedding + positional encoding
        h = self.node_embed(node_features)  # [B, N, d_model]
        h = h + self.pos_enc(start_times)   # add time position

        # Attention bias from pair features
        attn_bias = self.attn_bias(pair_features)  # [B, N, N]
        # Expand to [B*n_heads, N, N] for multi-head attention
        n_heads = self.encoder.layers[0].self_attn.num_heads
        attn_bias = attn_bias.repeat_interleave(n_heads, dim=0)  # [B*H, N, N]

        # Manual self-attention with bias
        for layer in self.encoder.layers:
            h2 = layer.self_attn(h, h, h, attn_mask=attn_bias,
                                 key_padding_mask=src_key_padding_mask)[0]
            h = h + layer.dropout1(h2)
            h = layer.norm1(h)

            # Feed-forward
            h2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = h + layer.dropout2(h2)
            h = layer.norm2(h)

        # Extract target query predictions (target is always at index 0)
        h_target = h[:, 0, :]  # [B, d_model]

        # Prediction
        base_pred = self.predictor(h_target).squeeze(-1)  # [B]

        # Resource bias: target query resources || mean of all concurrent query resources
        res_self = res_profile[:, 0, :]    # [B, 5]
        res_mean = res_profile.mean(dim=1) # [B, 5]
        res_bias = self.res_bias(
            torch.cat([res_self, res_mean], dim=-1)).squeeze(-1)  # [B]

        return base_pred + res_bias
