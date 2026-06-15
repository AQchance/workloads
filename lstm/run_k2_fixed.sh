#!/bin/bash
# K=2 fixed trace: GNN vs GNN+Completion vs ICONQ
set -e
cd /home/anqian/Desktop/my_lab/workloads
source /home/anqian/code/python/workloads/venv/bin/activate

TRACE=collect_concurrent/trace_2_fixed.csv
PREFIX=gnn_features_k2_fixed

# Step 1: Extract GNN features
echo "=== Step 1: Extract GNN features ==="
python lstm/extract_gnn_features.py $TRACE lstm/${PREFIX}.json

# Step 2: Prepare unified data (GNN original + completion + ICONQ)
echo "=== Step 2: Prepare training data ==="
python lstm/prepare_unified.py 0.7 k2_fixed

echo "=== Done preparing. Train with: ==="
echo "  GNN original:   python lstm/train_bilstm.py --trace $TRACE --data-prefix _gnn_sr70_k2_fixed --epochs 150"
echo "  GNN completion: python lstm/train_bilstm.py --trace $TRACE --data-prefix _comp_sr70_k2_fixed --epochs 150"
echo "  ICONQ:          python lstm/train_iconq_v2.py --trace $TRACE --data-prefix _iconq_sr70_k2_fixed --epochs 150"
