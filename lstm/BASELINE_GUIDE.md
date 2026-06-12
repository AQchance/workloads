# Bi-LSTM 并发时延预测基线对比说明

## 文件总览

### 我们的方法（GNN plan_emb + 双头预测）

| 步骤 | 文件 | 作用 |
|------|------|------|
| 1. 提取特征 | `extract_gnn_features.py` | 用训练好的 GNN 对所有 trace 查询跑推理，产出 plan_emb(128d) + 5维资源画像 |
| 2. 准备数据 | `prepare_training_data.py` | 构造交互特征序列，预测目标为 slowdown_ratio。产出 train_data.npz / test_data.npz |
| 3. 训练 | `train_bilstm.py` | 双头 Bi-LSTM：ratio_head + delta_head + gate。最终方法 |

### ICONQ 基线（公平对比版）

| 步骤 | 文件 | 作用 |
|------|------|------|
| 1. 提取特征 | `extract_iconq_v2.py` | 提取 ICONQ 扁平向量（GNN预测时延 + 算子计数 + estRows + 表cardinality），47维 |
| 2. 准备数据 | `prepare_iconq_v2.py` | 构造交互特征，预测目标为绝对并发秒数（与 ICONQ 论文一致）|
| 3. 训练 | `train_iconq_v2.py` | 单头 Bi-LSTM，预测绝对 runtime 秒数 |

### 实验性/废弃文件

| 文件 | 说明 |
|------|------|
| `extract_iconq_features.py` | ICONQ v1：用真实 serial_lat（非预测值），不公平对比 |
| `prepare_iconq_data.py` | 包含 GNN 资源画像，非纯 ICONQ |
| `prepare_iconq_pure.py` | 预测 slowdown_ratio，与 ICONQ 原文不一致 |
| `train_iconq_baseline.py` | 单头 ratio 预测 |
| `prepare_gnn_abs.py` | GNN 预测绝对秒数（公平对比用）|
| `prepare_mixed.py` | 混合 K2+K4 数据 |
| `train_k4.py`, `train_k2_improved.py`, `train_k4_improved.py` | 早期实验脚本 |

---

## 运行我们的方法（GNN）

```bash
cd /home/anqian/Desktop/my_lab/workloads
source venv/bin/activate  # 或你的 venv 路径

# K=2
python lstm/extract_gnn_features.py collect_concurrent/trace_2.csv explain_plans lstm/gnn_features.json
python lstm/prepare_training_data.py --trace collect_concurrent/trace_2.csv --features lstm/gnn_features.json --output-prefix train_data
python lstm/train_bilstm.py --trace collect_concurrent/trace_2.csv --data-prefix "" --epochs 200

# K=4
python lstm/extract_gnn_features.py collect_concurrent/trace_4.csv explain_plans lstm/gnn_features_k4.json
python lstm/prepare_training_data.py --trace collect_concurrent/trace_4.csv --features lstm/gnn_features_k4.json --output-prefix train_data_k4
python lstm/train_bilstm.py --trace collect_concurrent/trace_4.csv --data-prefix _k4 --epochs 200
```

## 运行 ICONQ 基线

```bash
# K=2
python lstm/extract_iconq_v2.py collect_concurrent/trace_2.csv explain_plans lstm/gnn_features.json lstm/iconq_features_v2.json
python lstm/prepare_iconq_v2.py --trace collect_concurrent/trace_2.csv --features lstm/iconq_features_v2.json --output-prefix train_data_iconq_v2
python lstm/train_iconq_v2.py --trace collect_concurrent/trace_2.csv --data-prefix _iconq_v2 --epochs 200

# K=4
python lstm/extract_iconq_v2.py collect_concurrent/trace_4.csv explain_plans lstm/gnn_features_k4.json lstm/iconq_features_v2_k4.json
python lstm/prepare_iconq_v2.py --trace collect_concurrent/trace_4.csv --features lstm/iconq_features_v2_k4.json --output-prefix train_data_iconq_v2_k4
python lstm/train_iconq_v2.py --trace collect_concurrent/trace_4.csv --data-prefix _iconq_v2_k4 --epochs 200
```

## 对比关键

两个版本都在**预测绝对并发秒数**（统一评估标准，Q-error 算在同一目标上）。

| | ICONQ 基线 | 我们的方法 |
|------|-----------|---------|
| 查询编码 | 47维手工扁平向量 | 128维 plan_emb + 5维资源画像 |
| 资源竞争信号 | 无 | 5维 resource conflict |
| 预测头 | 单头（abs runtime） | 双头（ratio + delta）|
| 特征维度 | 96 | 275 |
| 模型参数量 | ~800K | ~900K |

## 最终结果

| 指标 | K=2 ICONQ | K=2 我们 | 提升 | K=4 ICONQ | K=4 我们 | 提升 |
|------|-----------|---------|------|-----------|---------|------|
| P50 | 1.65x | 1.25x | +24% | 1.75x | 1.56x | +11% |
| P90 | 3.39x | 3.23x | +5% | 4.55x | 3.39x | +25% |
| P95 | 4.24x | 4.83x | -14% | 6.73x | 4.91x | +27% |
