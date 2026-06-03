"""
GNN model for query resource prediction.

Node features (20 dims): 13 base + 3 NDV + 4 distributed
Edge features (5 dims): 4 base + cross_engine flag

Architecture:
  Node Encoder (20-dim raw → 99-dim → 128-dim embeddings)
  → GATv2Conv × 3 with edge features + residual + LayerNorm
  → Hybrid Readout (max_pool + gated_attention + sum_pool) → 128-dim plan embedding
  → Global scalar skip (16-dim → 128-dim)
  → 4 independent prediction heads (memory, disk, network, CPU)
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool

from plan_parser import (
    N_OP_CLASSES, N_LOCATIONS, N_JOIN_TYPES, N_EXCHANGE_TYPES, N_ENGINE_TYPES,
)

# ─── Embedding dimensions ───
OP_CLASS_EMB_DIM = 32
LOCATION_EMB_DIM = 8
JOIN_TYPE_EMB_DIM = 8
EXCHANGE_TYPE_EMB_DIM = 8
ENGINE_TYPE_EMB_DIM = 8

N_CAT_EMB = OP_CLASS_EMB_DIM + LOCATION_EMB_DIM + JOIN_TYPE_EMB_DIM + EXCHANGE_TYPE_EMB_DIM + ENGINE_TYPE_EMB_DIM  # 64
N_SCALAR_PROJ = 32
N_NDV = 3
NODE_ENC_INPUT_DIM = N_CAT_EMB + N_SCALAR_PROJ + N_NDV  # 99

EDGE_LOC_PAIR_EMB_DIM = 8
EDGE_ENC_OUT_DIM = 1 + EDGE_LOC_PAIR_EMB_DIM + EXCHANGE_TYPE_EMB_DIM + 1 + 1  # 19

HIDDEN_DIM = 128
N_GAT_LAYERS = 3
N_HEADS = 4


class PlanGNN(nn.Module):
    """GAT-based GNN for execution plan resource prediction."""

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_GAT_LAYERS,
        n_heads: int = N_HEADS,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ─── Node feature embeddings ───
        self.op_class_emb = nn.Embedding(N_OP_CLASSES, OP_CLASS_EMB_DIM)
        self.location_emb = nn.Embedding(N_LOCATIONS, LOCATION_EMB_DIM)
        self.join_type_emb = nn.Embedding(N_JOIN_TYPES, JOIN_TYPE_EMB_DIM)
        self.exchange_type_emb = nn.Embedding(N_EXCHANGE_TYPES, EXCHANGE_TYPE_EMB_DIM)
        self.engine_type_emb = nn.Embedding(N_ENGINE_TYPES, ENGINE_TYPE_EMB_DIM)

        # ─── Edge feature embeddings ───
        self.edge_loc_emb = nn.Embedding(N_LOCATIONS * N_LOCATIONS, EDGE_LOC_PAIR_EMB_DIM)
        self.edge_exchange_emb = nn.Embedding(N_EXCHANGE_TYPES, EXCHANGE_TYPE_EMB_DIM)

        # ─── Node encoder ───
        # 9 base + 3 distributed + 2 exchange_bytes → 32-dim projection
        N_SCALAR = 14
        self.scalar_proj = nn.Sequential(
            nn.Linear(N_SCALAR, N_SCALAR_PROJ),
            nn.LeakyReLU(0.1),
        )
        # Input: 64 (cat_emb) + 32 (scalar_proj) + 3 (NDV) = 99
        self.node_encoder = nn.Sequential(
            nn.Linear(NODE_ENC_INPUT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ─── GAT layers with residual connections ───
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            concat = (i < n_layers - 1)
            if concat:
                out_dim = hidden_dim // n_heads
            else:
                out_dim = hidden_dim

            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=out_dim,
                    heads=n_heads,
                    edge_dim=EDGE_ENC_OUT_DIM,
                    concat=concat,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # ─── Gated attention readout ───
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.out_proj = nn.Identity()

        # ─── Global scalar skip connection ───
        # 9 base + n_nodes + 3 dist + 3 engine + 2 exch + 2 exch_bytes = 20
        N_GLOBAL = 20
        self.global_skip = nn.Sequential(
            nn.Linear(N_GLOBAL, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ─── Per-node memory auxiliary head ───
        self.node_mem_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ─── Query-level prediction heads (128 + 128 = 256-dim input) ───
        head_hidden = hidden_dim // 2
        self.mem_head = self._make_head(hidden_dim * 2, head_hidden, dropout)
        self.disk_head = self._make_head(hidden_dim * 2, head_hidden, dropout)
        self.net_head = self._make_head(hidden_dim * 2, head_hidden, dropout)
        self.cpu_head = self._make_head(hidden_dim * 2, head_hidden, dropout)

    @staticmethod
    def _make_head(in_dim: int, hidden: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Expand 20-dim raw features to 99-dim node embeddings.

        Feature layout (20 dims):
          0:  op_class_id (cat)
          1:  est_rows_log (scalar)
          2:  location_id (cat)
          3:  stream_count (scalar)
          4:  children_count (scalar)
          5:  depth_ratio (scalar)
          6:  n_equi_conds (scalar)
          7:  n_group_keys (scalar)
          8:  n_sort_keys (scalar)
          9:  has_filter (scalar)
          10: subtree_est_rows_log (scalar)
          11: join_type_id (cat)
          12: exchange_type_id (cat)
          13: join_mem_log (NDV, appended directly)
          14: agg_mem_log (NDV)
          15: sort_mem_log (NDV)
          16: engine_type_id (cat)
          17: table_skew_log (scalar)
          18: n_tiflash_instances (scalar)
          19: avg_column_correlation (scalar)
        """
        # ─── Categorical embeddings (64 dims) ───
        cat_emb = torch.cat([
            self.op_class_emb(x[:, 0].long()),          # 32
            self.location_emb(x[:, 2].long()),           # 8
            self.join_type_emb(x[:, 11].long()),         # 8
            self.exchange_type_emb(x[:, 12].long()),     # 8
            self.engine_type_emb(x[:, 16].long()),       # 8
        ], dim=-1)

        # ─── Scalar features (14 dims → 32) ───
        scalars = torch.cat([
            x[:, 1:2],   # est_rows_log
            x[:, 3:4],   # stream_count
            x[:, 4:5],   # children_count
            x[:, 5:6],   # depth_ratio
            x[:, 6:7],   # n_equi_conds
            x[:, 7:8],   # n_group_keys
            x[:, 8:9],   # n_sort_keys
            x[:, 9:10],  # has_filter
            x[:, 10:11], # subtree_est_rows_log
            x[:, 17:18], # table_skew_log
            x[:, 18:19], # n_tiflash_instances
            x[:, 19:20], # avg_column_correlation
            x[:, 20:21], # exch_row_width_log
            x[:, 21:22], # exch_est_bytes_log
        ], dim=-1)
        scalar_proj = self.scalar_proj(scalars)

        # ─── NDV features (3 dims) appended directly ───
        ndv_feats = x[:, 13:16]

        return torch.cat([cat_emb, scalar_proj, ndv_feats], dim=-1)  # 64 + 32 + 3 = 99

    def _encode_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Expand 5-dim raw edge features to 19-dim.

        Edge layout: [branch_ratio, loc_pair_id, exchange_type_id, is_build, cross_engine]
        """
        return torch.cat([
            edge_attr[:, 0:1],                              # branch_ratio
            self.edge_loc_emb(edge_attr[:, 1].long()),      # loc_pair → 8
            self.edge_exchange_emb(edge_attr[:, 2].long()), # exchange_type → 8
            edge_attr[:, 3:4],                              # is_build
            edge_attr[:, 4:5],                              # cross_engine
        ], dim=-1)  # 19

    def forward(self, data) -> Dict[str, torch.Tensor]:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # ─── Encode node and edge features ───
        x = self._encode_nodes(x)
        x = self.node_encoder(x)
        e = self._encode_edges(edge_attr)

        # ─── GAT layers with residual connections ───
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index, edge_attr=e)
            x = norm(x + x_new)

        # ─── Readout ───
        h_max = global_max_pool(x, batch)
        gate_logits = self.gate_mlp(x).squeeze(-1)
        h_gated_list = []
        for g in range(int(batch.max().item()) + 1):
            g_mask = (batch == g)
            g_x = x[g_mask]
            g_gate = F.softmax(gate_logits[g_mask], dim=0)
            h_gated_list.append((g_gate.unsqueeze(0) @ g_x).squeeze(0))
        h_gated = torch.stack(h_gated_list)
        h_sum = global_add_pool(x, batch)

        plan_emb = self.out_proj(h_max + h_gated + h_sum)

        # ─── Global scalar skip (16 dims → 128) ───
        # 9 base scalars
        node_scalars = torch.cat([
            data.x[:, 1:2], data.x[:, 3:4], data.x[:, 4:5], data.x[:, 5:6],
            data.x[:, 6:7], data.x[:, 7:8], data.x[:, 8:9], data.x[:, 9:10], data.x[:, 10:11],
        ], dim=-1)
        global_sum = global_add_pool(node_scalars, batch)

        # 3 distributed scalars
        dist_scalars = torch.cat([
            data.x[:, 17:18], data.x[:, 18:19], data.x[:, 19:20],
        ], dim=-1)
        dist_sum = global_add_pool(dist_scalars, batch)

        n_nodes = torch.bincount(batch + 1)[1:].float().unsqueeze(1)

        # 3 engine-type node counts
        engine_ids = data.x[:, 16].long()
        n_tidb = torch.zeros_like(n_nodes)
        n_tikv = torch.zeros_like(n_nodes)
        n_tiflash = torch.zeros_like(n_nodes)
        for g in range(int(batch.max().item()) + 1):
            g_mask = (batch == g)
            g_engines = engine_ids[g_mask]
            n_tidb[g] = (g_engines == 0).sum().float()
            n_tikv[g] = (g_engines == 1).sum().float()
            n_tiflash[g] = (g_engines == 2).sum().float()
        engine_counts = torch.cat([n_tidb, n_tikv, n_tiflash], dim=-1)

        # 2 Exchange-specific aggregates (row-count based)
        is_exch = (data.x[:, 0].long() == 3).float().unsqueeze(1)  # op_class == EXCHANGE
        exch_count = global_add_pool(is_exch, batch)
        exch_est_sum = global_add_pool(is_exch * data.x[:, 1:2], batch)  # log est_rows for Exchange

        # 2 Exchange bytes aggregates (width-aware, direct signal for network bytes)
        exch_row_width_sum = global_add_pool(is_exch * data.x[:, 20:21], batch)
        exch_est_bytes_sum = global_add_pool(is_exch * data.x[:, 21:22], batch)

        global_feat = self.global_skip(
            torch.cat([global_sum, n_nodes, dist_sum, engine_counts,
                        exch_count, exch_est_sum,
                        exch_row_width_sum, exch_est_bytes_sum], dim=-1))
        plan_emb_aug = torch.cat([plan_emb, global_feat], dim=-1)

        # ─── Predictions ───
        node_mem = self.node_mem_head(x)

        return {
            "mem": self.mem_head(plan_emb_aug),
            "disk": self.disk_head(plan_emb_aug),
            "net": self.net_head(plan_emb_aug),
            "cpu": self.cpu_head(plan_emb_aug),
            "plan_emb": plan_emb,
            "node_mem": node_mem,
        }
