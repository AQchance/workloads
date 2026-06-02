"""
Extract resource labels from EXPLAIN ANALYZE results.

Parses the markdown-table EXPLAIN ANALYZE output and constructs four scalar
resource labels per query:
  - cpu_time_ms:       wall-clock execution time (root operator time)
  - memory_bytes:      sum of all operator memory allocations
  - disk_io_bytes:     total data scanned from storage (data_scanned_rows)
  - network_rows:      sum of actRows across EXCHANGE operators (proxy for network vol)
"""

import os
import re
from typing import Dict, List, Optional


def _parse_time_ms(exec_info: str) -> float:
    """Extract the time value (in milliseconds) from execution_info.

    Handles formats: "time:3.39s", "time:41.7ms", "time:4m16.5s", "time:4m51.1s"
    """
    if not exec_info:
        return 0.0

    # Try "XmY.Ys" format first (minutes + seconds)
    m = re.search(r'time:(\d+)m([\d.]+)s', exec_info)
    if m:
        minutes = float(m.group(1))
        seconds = float(m.group(2))
        return (minutes * 60.0 + seconds) * 1000.0

    # Try plain "X.XX unit" format
    m = re.search(r'time:([\d.]+)(s|ms|µs|us)', exec_info)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit == 's':
            return val * 1000.0
        elif unit == 'ms':
            return val
        elif unit in ('µs', 'us'):
            return val / 1000.0
    return 0.0


def _parse_memory_bytes(raw: str) -> float:
    """Parse memory string like '11.5 KB', '157.7 KB', '0 Bytes', 'N/A' to bytes."""
    raw = raw.strip()
    if raw.upper() == 'N/A' or raw == '':
        return 0.0
    m = re.match(r'([\d.]+)\s*(Bytes|KB|MB|GB|TB)', raw, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        multipliers = {"BYTES": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4}
        return val * multipliers.get(unit, 1)
    return 0.0


def _parse_data_scanned_rows(exec_info: str) -> float:
    """Extract data_scanned_rows from tiflash_scan sub-structure."""
    if not exec_info:
        return 0.0
    m = re.search(r'data_scanned_rows:(\d+)', exec_info)
    if m:
        return float(m.group(1))
    return 0.0


def _parse_tot_read_ms(exec_info: str) -> float:
    """Extract tot_read (disk read time) from tiflash_scan sub-structure."""
    if not exec_info:
        return 0.0
    m = re.search(r'tot_read:(\d+)ms', exec_info)
    if m:
        return float(m.group(1))
    return 0.0


def _parse_proc_max_ms(exec_info: str) -> float:
    """Extract proc max time in ms from tiflash_task sub-structure."""
    if not exec_info:
        return 0.0
    # Try "XmY.Ys" format
    m = re.search(r'proc max:(\d+)m([\d.]+)s', exec_info)
    if m:
        minutes = float(m.group(1))
        seconds = float(m.group(2))
        return (minutes * 60.0 + seconds) * 1000.0
    # Try plain format
    m = re.search(r'proc max:([\d.]+)(s|ms|µs|us)', exec_info)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit == 's':
            return val * 1000.0
        elif unit == 'ms':
            return val
        elif unit in ('µs', 'us'):
            return val / 1000.0
    return 0.0


def _parse_act_rows(raw: str) -> float:
    """Parse actRows from table cell."""
    try:
        return float(raw.strip())
    except (ValueError, AttributeError):
        return 0.0


def _normalize_op_name(raw_id: str) -> str:
    """Extract operator name from the id column (strip tree chars, ID number, annotations)."""
    cleaned = re.sub(r'^[│├└─\s]+', '', raw_id)
    cleaned = re.sub(r'\(Build\)|\(Probe\)', '', cleaned).strip()
    cleaned = re.sub(r'_\d+$', '', cleaned)
    return cleaned


def is_scan_operator(op_name: str) -> bool:
    """Check if operator is a scan type (reads from storage)."""
    scan_ops = {"TableFullScan", "TableRangeScan", "IndexRangeScan",
                "TableRowIDScan", "IndexLookUp", "IndexReader"}
    return op_name in scan_ops


def is_exchange_operator(op_name: str) -> bool:
    """Check if operator is an exchange type (network transfer)."""
    return op_name in {"ExchangeSender", "ExchangeReceiver"}


def parse_explain_analyze_plan(block_text: str) -> Optional[Dict]:
    """
    Parse a single EXPLAIN ANALYZE block into structured data.

    Args:
        block_text: Raw text of one block (from one "--- Round X | Line Y | Step Z ---")

    Returns:
        Dict with parsed table rows, or None if parsing fails.
    """
    lines = block_text.strip().split('\n')

    # Extract SQL text (line starting with "-- SQL:")
    sql_text = ""
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('-- SQL:'):
            sql_text = line.replace('-- SQL:', '').strip()
            start_idx = i + 1
            break

    # Find the table (starts with "| id |")
    table_start = -1
    for i in range(start_idx, len(lines)):
        if lines[i].strip().startswith('| id |'):
            table_start = i
            break

    if table_start < 0:
        return None

    # Parse table rows (skip header and separator lines)
    rows = []
    for i in range(table_start + 2, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('--- Round'):
            break

        # Each row: | cell1 | cell2 | ... |
        # Remove leading/trailing pipe, then split
        if line.startswith('|') and line.endswith('|'):
            cells = [c.strip() for c in line[1:-1].split('|')]
            if len(cells) >= 9:
                rows.append({
                    "id": cells[0].strip(),
                    "estRows": cells[1].strip(),
                    "actRows": cells[2].strip(),
                    "task": cells[3].strip(),
                    "access_object": cells[4].strip(),
                    "execution_info": cells[5].strip(),
                    "operator_info": cells[6].strip(),
                    "memory": cells[7].strip(),
                    "disk": cells[8].strip(),
                })

    if not rows:
        return None

    # Compute tree depth from indentation in id column
    for row in rows:
        stripped = row["id"].lstrip(' │├└─')
        depth = (len(row["id"]) - len(stripped)) // 2
        row["depth"] = depth
        row["op_name"] = _normalize_op_name(row["id"])

    return {
        "sql": sql_text,
        "rows": rows,
        "n_rows": len(rows),
    }


def extract_labels(plan_data: Dict) -> Dict[str, float]:
    """
    Extract four resource labels from a parsed EXPLAIN ANALYZE plan.

    Returns:
        Dict with keys: cpu_time_ms, memory_bytes, disk_io_rows, network_rows
    """
    rows = plan_data["rows"]

    # ─── CPU time: root operator's wall clock time ───
    # Root is the first row (no indentation)
    root_row = rows[0] if rows else {}
    cpu_time_ms = _parse_time_ms(root_row.get("execution_info", ""))

    # ─── Memory: sum of all operator memory allocations ───
    memory_bytes = 0.0
    for row in rows:
        memory_bytes += _parse_memory_bytes(row.get("memory", ""))

    # ─── Disk I/O: data_scanned_rows from SCAN operators ───
    disk_io_rows = 0.0
    for row in rows:
        if is_scan_operator(row.get("op_name", "")):
            disk_io_rows += _parse_data_scanned_rows(row.get("execution_info", ""))

    # ─── Network: actRows of EXCHANGE operators ───
    network_rows = 0.0
    for row in rows:
        if is_exchange_operator(row.get("op_name", "")):
            network_rows += _parse_act_rows(row.get("actRows", "0"))

    return {
        "cpu_time_ms": cpu_time_ms,
        "memory_bytes": memory_bytes,
        "disk_io_rows": disk_io_rows,       # rows scanned (can convert to bytes later)
        "network_rows": network_rows,        # rows exchanged (can convert to bytes later)
    }


def parse_explain_analyze_file(filepath: str) -> List[Dict]:
    """
    Parse a Q*_explain_analyze.txt file.

    Returns:
        List of dicts, each with keys: line_no, sql, labels, plan_data
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Split by block headers: "--- Round X | Line Y | Step Z ---"
    blocks = re.split(
        r'^--- Round \d+ \| Line (\d+) \| Step \d+ ---$',
        content, flags=re.MULTILINE
    )

    results = []
    for i in range(1, len(blocks), 2):
        if i + 1 >= len(blocks):
            break
        try:
            line_no = int(blocks[i].strip())
        except ValueError:
            continue
        block_text = blocks[i + 1].strip()
        if not block_text:
            continue

        plan_data = parse_explain_analyze_plan(block_text)
        if plan_data is None or plan_data["n_rows"] == 0:
            continue

        labels = extract_labels(plan_data)

        results.append({
            "line_no": line_no,
            "sql": plan_data["sql"],
            "labels": labels,
            "n_rows": plan_data["n_rows"],
        })

    return results


def extract_all_labels(analyze_dir: str) -> Dict[str, List[Dict]]:
    """
    Parse all Q*_explain_analyze.txt files in a directory.

    Returns:
        Dict mapping template name (e.g., "Q1") to list of label dicts.
    """
    all_labels = {}
    for fname in sorted(os.listdir(analyze_dir)):
        if not fname.endswith("_explain_analyze.txt"):
            continue
        template = fname.replace("_explain_analyze.txt", "")
        filepath = os.path.join(analyze_dir, fname)
        entries = parse_explain_analyze_file(filepath)
        if entries:
            all_labels[template] = entries
        print(f"  {fname}: {len(entries)} labels extracted")

    return all_labels


# ─── Standalone test ───
if __name__ == "__main__":
    import sys

    analyze_dir = os.path.join(os.path.dirname(__file__), "..", "explain_analyze_results")
    if not os.path.isdir(analyze_dir):
        print(f"ERROR: EXPLAIN ANALYZE directory not found: {analyze_dir}")
        sys.exit(1)

    print(f"Extracting labels from: {analyze_dir}\n")

    all_labels = extract_all_labels(analyze_dir)

    total = sum(len(v) for v in all_labels.values())
    print(f"\nTotal: {total} labeled queries across {len(all_labels)} templates")

    # Print summary
    print("\n" + "=" * 80)
    print("Per-template label summary (first plan):")
    print("=" * 80)
    print(f"{'Templ':>5s}  {'Line':>4s}  {'CPU(ms)':>10s}  {'Mem(B)':>10s}  {'Disk(rows)':>12s}  {'Net(rows)':>12s}")
    print("-" * 80)

    for template in sorted(all_labels.keys()):
        entry = all_labels[template][0]
        labels = entry["labels"]
        print(f"{template:>5s}  {entry['line_no']:>4d}  "
              f"{labels['cpu_time_ms']:>10.1f}  "
              f"{labels['memory_bytes']:>10.0f}  "
              f"{labels['disk_io_rows']:>12.0f}  "
              f"{labels['network_rows']:>12.0f}")

    # Distribution stats across all labels
    print("\n" + "=" * 80)
    print("Label distribution across all 750 queries:")
    print("=" * 80)
    all_cpu = []
    all_mem = []
    all_disk = []
    all_net = []
    for template, entries in all_labels.items():
        for e in entries:
            l = e["labels"]
            all_cpu.append(l["cpu_time_ms"])
            all_mem.append(l["memory_bytes"])
            all_disk.append(l["disk_io_rows"])
            all_net.append(l["network_rows"])

    for name, vals in [("CPU(ms)", all_cpu), ("Mem(B)", all_mem),
                        ("Disk(rows)", all_disk), ("Net(rows)", all_net)]:
        v = sorted(vals)
        n = len(v)
        print(f"  {name:>12s}:  min={min(v):>12.1f}  p50={v[n//2]:>12.1f}  "
              f"p95={v[int(n*0.95)]:>12.1f}  max={max(v):>12.1f}")
