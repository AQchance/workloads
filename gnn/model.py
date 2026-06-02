"""
GNN model for query resource prediction.

Architecture:
  Node Encoder (raw features → 128-dim embeddings)
  → GATv2Conv × 3 with edge features + residual + LayerNorm
  → Hybrid Readout (root || gated-attention || mean) → 128-dim plan embedding
  → 4 independent prediction heads (memory, disk, network, CPU)
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool, global_max_pool

from plan_parser import (
    N_OP_CLASSES, N_LOCATIONS, N_JOIN_TYPES, N_EXCHANGE_TYPES, N_ENGINE_TYPES,
)

# ─── Embedding dimensions ───
OP_CLASS_EMB_DIM = 32
LOCATION_EMB_DIM = 8
JOIN_TYPE_EMB_DIM = 8
EXCHANGE_TYPE_EMB_DIM = 8
ENGINE_TYPE_EMB_DIM = 8

# After embedding expansion: 64 (cat_emb) + 32 (scalar_proj) = 96
NODE_ENC_INPUT_DIM = (
    OP_CLASS_EMB_DIM + LOCATION_EMB_DIM + JOIN_TYPE_EMB_DIM
    + EXCHANGE_TYPE_EMB_DIM + ENGINE_TYPE_EMB_DIM + 32
)

# Edge embedding: 1 (ratio) + 8 (loc_pair_emb) + 8 (exchange_emb) + 1 (is_build) + 1 (cross_engine) = 19
EDGE_LOC_PAIR_EMB_DIM = 8
EDGE_ENC_OUT_DIM = 1 + EDGE_LOC_PAIR_EMB_DIM + EXCHANGE_TYPE_EMB_DIM + 1 + 1  # = 19

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
        # loc_pair: 3×3 = 9 possible parent-child location pairs
        self.edge_loc_emb = nn.Embedding(N_LOCATIONS * N_LOCATIONS, EDGE_LOC_PAIR_EMB_DIM)
        self.edge_exchange_emb = nn.Embedding(N_EXCHANGE_TYPES, EXCHANGE_TYPE_EMB_DIM)

        # ─── Node encoder ───
        # Scalar features need their own pathway: the 65-dim input has 56-dim embeddings
        # (op_class, location, join_type, exchange_type) but only 9 varying scalar dims.
        # Project scalars separately to give them comparable weight.
        N_SCALAR = 12  # 9 base + 3 distributed (skew_log, n_tiflash, col_corr)
        self.scalar_proj = nn.Sequential(
            nn.Linear(N_SCALAR, 32),
            nn.LeakyReLU(0.1),
        )
        # Total input: 56 (cat_emb) + 32 (scalar_proj) = 88 = NODE_ENC_INPUT_DIM
        # Use LeakyReLU to avoid killing gradient/variance from scalar features
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
            concat = (i < n_layers - 1)  # concat heads in early layers, average in last
            if concat:
                out_dim = hidden_dim // n_heads  # per-head, total concat = hidden_dim
            else:
                out_dim = hidden_dim  # average of n_heads each outputting hidden_dim = hidden_dim

            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,  # output of all layers is always hidden_dim
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

        # ─── Readout combiner: sum root + gated + mean → hidden_dim ───
        # Avoid projection layers that squash cross-graph variance;
        # summation preserves each component's signal.
        self.out_proj = nn.Identity()

        # ─── Global scalar skip connection ───
        # Aggregated scalar features capture graph-size-sensitive signals.
        # Distributed additions: per-engine node counts, skew/col_corr aggregates.
        #
        # Base: 9 scalar sums + n_nodes = 10
        # Distributed: 3 dist sums + 3 engine counts = 6
        N_GLOBAL = 16
        self.global_skip = nn.Sequential(
            nn.Linear(N_GLOBAL, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ─── Per-node memory auxiliary head ───
        # Predicts per-operator memory from node embeddings before readout.
        # Only trained on operators where EXPLAIN ANALYZE reports non-N/A memory.
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
        """Expand raw node features (13-dim) to embeddings + projected scalars.

        Categorical features → embedding lookup (56 dims).
        Scalar features → separate MLP projection (32 dims).
        Total: 88 dims → node_encoder → hidden_dim.
        """
        # ─── Categorical embeddings (64 dims total) ───
        op_class_id = x[:, 0].long()
        location_id = x[:, 2].long()
        join_type_id = x[:, 11].long()
        exchange_type_id = x[:, 12].long()
        engine_type_id = x[:, 13].long()

        cat_emb = torch.cat([
            self.op_class_emb(op_class_id),          # 32
            self.location_emb(location_id),           # 8
            self.join_type_emb(join_type_id),         # 8
            self.exchange_type_emb(exchange_type_id), # 8
            self.engine_type_emb(engine_type_id),     # 8
        ], dim=-1)  # total: 64

        # ─── Scalar features (12 dims) with separate projection ───
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
            x[:, 14:15], # table_skew_log (new)
            x[:, 15:16], # n_tiflash_instances (new)
            x[:, 16:17], # avg_column_correlation (new)
        ], dim=-1)

        scalar_proj = self.scalar_proj(scalars)  # 12 → 32

        return torch.cat([cat_emb, scalar_proj], dim=-1)  # 64 + 32 = 96

    def _encode_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Expand raw edge features (4-dim) to full edge features (18-dim)."""
        # edge_attr columns: [0:branch_ratio, 1:loc_pair, 2:exchange_type, 3:is_build]
        branch_ratio = edge_attr[:, 0:1]
        loc_pair_id = edge_attr[:, 1].long()
        exchange_type_id = edge_attr[:, 2].long()
        is_build = edge_attr[:, 3:4]

        return torch.cat([
            branch_ratio,                              # 1
            self.edge_loc_emb(loc_pair_id),            # 8
            self.edge_exchange_emb(exchange_type_id),  # 8
            is_build,                                  # 1
        ], dim=-1)

    def forward(self, data) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a batch of graphs.

        Args:
            data: PyG Batch object with x, edge_index, edge_attr, batch, root_idx

        Returns:
            Dict with keys: mem, disk, net, cpu (each [batch_size, 1])
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # ─── Encode node and edge features ───
        x = self._encode_nodes(x)
        x = self.node_encoder(x)          # → hidden_dim
        e = self._encode_edges(edge_attr) # → EDGE_ENC_OUT_DIM

        # ─── GAT layers with residual connections ───
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index, edge_attr=e)
            x = norm(x + x_new)  # Residual: all layers keep hidden_dim

        # ─── Readout ───
        # Global max pooling captures extreme node activations (structural signal)
        h_max = global_max_pool(x, batch)  # [batch_size, hidden_dim]

        # Gated attention pooling (per-graph weighted sum)
        gate_logits = self.gate_mlp(x).squeeze(-1)  # [total_nodes]
        h_gated_list = []
        for g in range(int(batch.max().item()) + 1):
            g_mask = (batch == g)
            g_x = x[g_mask]
            g_gate = F.softmax(gate_logits[g_mask], dim=0)
            h_gated_list.append((g_gate.unsqueeze(0) @ g_x).squeeze(0))
        h_gated = torch.stack(h_gated_list)  # [batch_size, hidden_dim]

        # Global sum pooling: graph-size-aware (18-node ≠ 10-node plan)
        h_sum = global_add_pool(x, batch)  # [batch_size, hidden_dim]

        # ─── Plan embedding ───
        plan_emb = self.out_proj(h_max + h_gated + h_sum)

        # ─── Global scalar skip: aggregate per-graph scalar features ───
        # Base 9 scalars
        node_scalars = torch.cat([
            data.x[:, 1:2],   # est_rows_log
            data.x[:, 3:4],   # stream_count
            data.x[:, 4:5],   # children_count
            data.x[:, 5:6],   # depth_ratio
            data.x[:, 6:7],   # n_equi
            data.x[:, 7:8],   # n_group
            data.x[:, 8:9],   # n_sort
            data.x[:, 9:10],  # has_filter
            data.x[:, 10:11], # subtree_est
        ], dim=-1)  # [total_nodes, 9]

        global_sum = global_add_pool(node_scalars, batch)  # [batch_size, 9]

        # ─── Distributed scalar aggregates ───
        # Per-graph sums of table-skew and column-correlation signals
        dist_scalars = torch.cat([
            data.x[:, 14:15],  # table_skew_log (nonzero only for SCAN nodes)
            data.x[:, 15:16],  # n_tiflash_instances
            data.x[:, 16:17],  # avg_column_correlation
        ], dim=-1)  # [total_nodes, 3]
        dist_sum = global_add_pool(dist_scalars, batch)  # [batch_size, 3]

        # ─── Per-graph engine-type node counts ───
        engine_type_ids = data.x[:, 13].long()  # [total_nodes]
        n_nodes_per_graph = torch.bincount(batch + 1)[1:].float().unsqueeze(1)
        n_tidb = torch.zeros_like(n_nodes_per_graph)
        n_tikv = torch.zeros_like(n_nodes_per_graph)
        n_tiflash = torch.zeros_like(n_nodes_per_graph)
        for g in range(int(batch.max().item()) + 1):
            g_mask = (batch == g)
            g_engines = engine_type_ids[g_mask]
            n_tidb[g] = (g_engines == 0).sum().float()
            n_tikv[g] = (g_engines == 1).sum().float()
            n_tiflash[g] = (g_engines == 2).sum().float()
        engine_counts = torch.cat([n_tidb, n_tikv, n_tiflash], dim=-1)  # [batch_size, 3]

        global_feat = self.global_skip(
            torch.cat([global_sum, n_nodes_per_graph, dist_sum, engine_counts], dim=-1)
        )  # [batch_size, 128]
        plan_emb_aug = torch.cat([plan_emb, global_feat], dim=-1)  # [batch_size, 256]

        # ─── Predictions ───
        # Node-level memory (from GNN embeddings, before readout)
        node_mem = self.node_mem_head(x)  # [total_nodes, 1]

        return {
            "mem": self.mem_head(plan_emb_aug),
            "disk": self.disk_head(plan_emb_aug),
            "net": self.net_head(plan_emb_aug),
            "cpu": self.cpu_head(plan_emb_aug),
            "plan_emb": plan_emb,
            "node_mem": node_mem,
        }
