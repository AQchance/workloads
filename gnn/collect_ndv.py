"""
Collect column-level NDV (number of distinct values) from TiDB statistics.

Usage: python collect_ndv.py
Output: ndv_cache.json — mapping from "table.column" to NDV value
"""

import json
import re
import os
import pymysql

DB_CONFIG = {
    "host": "172.19.0.11",
    "port": 4000,
    "user": "root",
    "password": "",
    "database": "tpch_sf40",
    "charset": "utf8mb4",
}


def collect_column_ndv() -> dict:
    """Query TiDB stats for all column NDVs. Returns {table.column: ndv}."""
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Get all TPC-H columns with their NDV
    cursor.execute("""
        SELECT
            CONCAT(c.table_name, '.', c.column_name) AS col,
            h.distinct_count AS ndv,
            ROUND(h.tot_col_size / GREATEST(m.count, 1), 0) AS avg_width
        FROM mysql.stats_histograms h
        JOIN information_schema.tables t
            ON h.table_id = t.tidb_table_id AND t.table_schema = 'tpch_sf40'
        JOIN information_schema.columns c
            ON t.table_name = c.table_name
            AND t.table_schema = c.table_schema
            AND CAST(h.hist_id AS CHAR) = CAST(c.ordinal_position AS CHAR)
        JOIN mysql.stats_meta m ON h.table_id = m.table_id
        WHERE t.table_schema = 'tpch_sf40'
            AND h.is_index = 0
    """)

    ndv_cache = {}
    for col, ndv, width in cursor.fetchall():
        ndv_cache[col] = {"ndv": int(ndv), "avg_width": int(width)}

    cursor.close()
    conn.close()

    print(f"Collected NDV for {len(ndv_cache)} columns")
    return ndv_cache


def parse_join_columns(op_info: str) -> list:
    """Extract column names from join condition in operator_info.

    Handles: inner join, equal:[eq(tpch_sf40.lineitem.l_orderkey, tpch_sf40.orders.o_orderkey)]
    Returns: ['lineitem.l_orderkey', 'orders.o_orderkey']
    """
    cols = re.findall(r'tpch_sf40\.(\w+)\.(\w+)', op_info)
    return [f"{t}.{c}" for t, c in cols]


def parse_group_columns(op_info: str) -> list:
    """Extract column names from group by clause.

    Handles: group by:tpch_sf40.nation.n_name, funcs:sum(...)
    Returns: ['nation.n_name']
    """
    if 'group by:' not in op_info:
        return []
    m = re.search(r'group by:(.*?)(?:, funcs:|$)', op_info)
    if not m:
        return []
    group_part = m.group(1)
    cols = re.findall(r'tpch_sf40\.(\w+)\.(\w+)', group_part)
    return [f"{t}.{c}" for t, c in cols]


def compute_node_ndv_features(plan_text: str, ndv_cache: dict) -> list:
    """
    Parse EXPLAIN plan and compute per-node NDV-based memory estimates.

    Returns: list of (est_hash_table_bytes, est_agg_mem, est_sort_mem) per node,
             aligned with plan node order. Returns 0 for non-memory-intensive nodes.
    """
    lines = plan_text.strip().split('\n')
    plan_lines = []
    for line in lines:
        if line.startswith('--') or not line.strip():
            continue
        if '\t' in line:
            plan_lines.append(line)

    results = []
    for line in plan_lines:
        stripped = line.lstrip(' │├└─')
        parts = stripped.split('\t')
        if len(parts) < 5:
            results.append((0.0, 0.0, 0.0))
            continue

        raw_id = parts[0].strip()
        est_rows_str = parts[1].strip() if len(parts) > 1 else "1.0"
        op_info = parts[4].strip() if len(parts) > 4 else ""

        try:
            est_rows = float(est_rows_str)
        except ValueError:
            est_rows = 1.0

        # Normalize operator name
        op_name = re.sub(r'^[│├└─\s]+', '', raw_id)
        op_name = re.sub(r'\(Build\)|\(Probe\)', '', op_name).strip()
        op_name = re.sub(r'_\d+$', '', op_name)

        is_join = op_name in ('HashJoin', 'IndexHashJoin', 'IndexJoin', 'MergeJoin')
        is_agg = op_name in ('HashAgg', 'StreamAgg')
        is_sort = op_name in ('Sort', 'TopN')
        is_build = '(Build)' in raw_id

        join_mem = 0.0
        agg_mem = 0.0
        sort_mem = 0.0

        if is_join:
            # Parse join columns
            cols = parse_join_columns(op_info)
            if cols:
                # Find NDV of each join key
                ndvs = []
                widths = []
                for col in cols:
                    info = ndv_cache.get(col, {"ndv": est_rows, "avg_width": 8})
                    ndvs.append(info["ndv"])
                    widths.append(info["avg_width"])

                # Build side determines hash table size
                # If this is the Build side, use its estRows; otherwise use the smaller NDV
                n_equi = len(re.findall(r'eq\(', op_info))
                min_ndv = min(ndvs) if ndvs else est_rows
                avg_width = sum(widths) / len(widths) if widths else 8

                # Hash table ≈ min(build_rows, build_ndv) × avg_width × (1 + 0.1 × n_equi)
                if is_build:
                    hash_entries = min(est_rows, min_ndv)
                else:
                    hash_entries = min(est_rows, min_ndv)  # Probe side estimate too

                join_mem = hash_entries * avg_width * (1.0 + 0.1 * n_equi)

        if is_agg:
            group_cols = parse_group_columns(op_info)
            if group_cols:
                ndvs = []
                widths = []
                for col in group_cols:
                    info = ndv_cache.get(col, {"ndv": est_rows, "avg_width": 8})
                    ndvs.append(info["ndv"])
                    widths.append(info["avg_width"])

                n_group = len(group_cols)
                # Agg hash table ≈ min(input_rows, product_of_ndvs) × total_width
                agg_ndv = min(ndvs) if ndvs else est_rows
                total_width = sum(widths) if widths else 8 * n_group
                agg_mem = min(est_rows, agg_ndv) * total_width

        if is_sort:
            n_sort = len(re.findall(r'(?:tpch_sf40\.\w+\.\w+|Column#\d+)', op_info))
            if n_sort == 0:
                n_sort = 1
            # Sort memory ≈ est_rows × n_sort × 8 bytes (rough per-row comparison cost)
            sort_mem = est_rows * n_sort * 8

        results.append((join_mem, agg_mem, sort_mem))

    return results


if __name__ == '__main__':
    # Collect NDV
    print("Collecting column NDV from TiDB...")
    ndv_cache = collect_column_ndv()

    # Quick stats
    ndvs = [v['ndv'] for v in ndv_cache.values()]
    print(f"NDV range: {min(ndvs)} ~ {max(ndvs)}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'ndv_cache.json')
    with open(output_path, 'w') as f:
        json.dump(ndv_cache, f, indent=2)
    print(f"Saved to {output_path}")

    # Test on a few plans
    plan_dir = os.path.join(os.path.dirname(__file__), '..', 'explain_plans')
    test_files = sorted(os.listdir(plan_dir))[:3]
    print(f"\nTesting on {len(test_files)} plans...")
    for fname in test_files:
        with open(os.path.join(plan_dir, fname)) as f:
            plan_text = f.read()
        features = compute_node_ndv_features(plan_text, ndv_cache)
        non_zero = [(i, f) for i, f in enumerate(features) if sum(f) > 0]
        max_mem = max((f[0] for f in features), default=0)
        print(f"  {fname}: {len(non_zero)}/{len(features)} nodes with memory signal, "
              f"max_hash_table={max_mem/1048576:.1f}MB")
