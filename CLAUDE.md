# GNN 查询资源预测项目

## 项目目标

用 GNN 从 TiDB 执行计划中预测查询的四维资源消耗：CPU、内存、磁盘 IO、网络 IO。最终产出是一个 128 维 plan_emb（资源感知的执行计划结构表示）+ 四个资源标量，用于下游并发调度（如 ICONQ 的 LSTM 输入）。

## 数据

- **1631 条 SQLStorm 查询**（TPC-H schema，768 种不同执行计划拓扑）
- EXPLAIN 计划：`/home/anqian/Desktop/my_lab/workloads/explain_plans/`（每查询一个 .txt，TAB 分隔）
- EXPLAIN ANALYZE：`/home/anqian/Desktop/my_lab/workloads/explain_analyze_results/`
- NDV 缓存：`/home/anqian/Desktop/my_lab/workloads/ndv_cache.json`
- 原始查询：`/home/anqian/Desktop/my_lab/workloads/SQLStorm/`

## 标签提取

- **CPU 资源**：`Σ(proc_max × threads × tasks)` TiFlash + `Σ(tot_proc)` TiKV。标签跨度 1s~45514s，Q-error 在原始空间爆炸，训练/评估在 `log(1+x)` 空间
- **内存**：`Σ(memory)` 只有 8.9% 算子有值（TiFlash 全 N/A），严重缺失。当前用 NDV 代理特征弥补（P50=1.55）
- **磁盘 IO**：行数版 `Σ(data_scanned_rows)`，字节版 `Σ(data_scanned_rows × 表行宽)`。字节版 R²=0.94 优于行数版 0.88
- **网络 IO**：`Σ(Exchange actRows)`。**不需要 GNN**，estRows 线性回归即可达到 P50=1.13（优于 GNN P50=1.38）

## 模型架构（最终版）

**文件**：`/home/anqian/Desktop/my_lab/workloads/gnn/train_ndv.py`（4 回归头版本）

- 节点特征：91 维（56 cat 嵌入 + 32 标量投影 + 3 NDV 代理）
- 3 层 GATv2Conv + 残差 + LayerNorm，4 注意力头，边特征参与注意力
- 混合读出：max_pool + gated_attention + sum_pool → 128 维 plan_emb
- 全局标量跳跃连接：9 标量 sums + n_nodes（10 维→128 维）+ plan_emb(128) = 256 维入预测头
- 4 个独立回归头，Huber 损失，log(1+x)→z-score 归一化

### 磁盘字节版

**文件**：`/home/anqian/Desktop/my_lab/workloads/gnn/train_disk_bytes.py`

## 最终结果（seed=42, 250 epochs）

| 维度 | P50 | P90 | P95 | R² | 备注 |
|------|-----|-----|-----|-----|------|
| 磁盘 IO（字节） | 1.26 | 1.90 | 2.80 | **0.94** | 字节版 |
| 磁盘 IO（行数） | 1.14 | 1.61 | 1.95 | 0.88 | 行数版 |
| CPU（墙钟延迟） | 1.54 | 3.19 | 4.54 | 0.72 | 和 Lamba 可对比 |
| 内存 | 1.52 | 15.78 | 71.48 | 0.79 | 中位尚可，尾部差 |
| 网络 IO | 1.13 | 1.43 | 1.70 | — | **estRows LR，不用 GNN** |

### 分类结果（CPU/内存，3 类 LOW/MED/HIGH）

| 维度 | 准确率 | 基线 |
|------|--------|------|
| CPU | 65% | 31% |
| 内存 | 89% | 33% |

## 执行命令

```bash
cd /home/anqian/Desktop/my_lab/workloads
source ../.venv/bin/activate

# 4 回归头（标准版）
python gnn/train_ndv.py --epochs 250 --seed 42

# 磁盘字节版
python gnn/train_disk_bytes.py --epochs 250 --seed 42

# 分类版（CPU/内存分类 + 磁盘回归 + 网络 LR）
python gnn/train_final.py --epochs 250 --seed 42
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `gnn/model.py` | GNN 模型定义（PlanGNN 类） |
| `gnn/train_ndv.py` | **主力**训练脚本，4 回归头，NDV 在节点编码器中 |
| `gnn/train_disk_bytes.py` | 磁盘字节版训练 |
| `gnn/train_final.py` | 分类版训练（CPU/内存分类） |
| `gnn/plan_parser.py` | 原始 TPC-H EXPLAIN 解析器 |
| `gnn/collect_ndv.py` | NDV 采集脚本 |
| `collect_explain_analyze.py` | 批量采集 EXPLAIN ANALYZE |
| `execute_queries.sh` | 查询执行脚本（含断点续传和 TiDB 重启） |
| `fix_sql_tidb.py` | SQLStorm 查询语法修复 |
| `ndv_cache.json` | 列级 NDV + avg_width 缓存 |
| `sqlstorm_sqls.txt` | 通过验证的查询编号列表 |

## 重要发现

1. **从零训练不稳定**：1631 条查询/768 种拓扑对 377K 参数不够。7 个 seed 中 6 个收敛但效果波动大，1 个完全坍塌。需要 warm-start 或更多数据（5000+ 条）

2. **网络 IO 用 estRows LR 即可**：P50=1.13，不需要 GNN

3. **内存标签严重缺失**：91% 算子（TiFlash）为 N/A。NDV 代理特征部分弥补（P50=1.52）。TiDB 无逐查询的 TiFlash 算子内存 API

4. **CPU 资源 ≠ 墙钟延迟**：CPU 资源 = Σ(proc_max × threads)，中位是墙钟的 100 倍（并行效应）。标签方差极大导致 Q-error 爆炸，R² 更适合做指标

5. **TPC-H 15 模板训练失败**：同模板内 50 条查询 EXPLAIN 特征完全相同，模型退化到预测模板均值。换成 SQLStorm 的 768 种拓扑后 R² 从 0 跳到 0.94

6. **多任务训练是必要的**：去掉网络头后 3 头版本不收敛。4 头互相提供正则化

## 待解决

- 内存尾部差（P95=71）：TiFlash 算子级内存数据缺失
- 从零训练不稳定：需采集更多查询（SQLStorm 有 15000+ 条可用）
- 日志空间 Q-error vs 原始空间 Q-error 的报告规范

## TiDB 连接

Docker 容器 `tidb1`：`mysql -h 172.19.0.11 -P 4000 -u root`，数据库 `tpch_sf40`
