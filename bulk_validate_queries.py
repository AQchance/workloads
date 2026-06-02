#!/usr/bin/env python3
"""
Bulk validate SQLStorm v1.0/tpch queries against TiDB.

For each pre-generated query:
  1. Fix PostgreSQL syntax for TiDB compatibility
  2. Run EXPLAIN to check TiDB accepts it
  3. If valid, save .sql and EXPLAIN plan to output dirs

Output:
  workloads/SQLStorm/<new_id>.sql        — validated SQL
  workloads/explain_plans/<new_id>.txt   — EXPLAIN plan
  workloads/new_sqlstorm_ids.txt         — mapping: new_id → original_source
"""

import os, sys, re, subprocess, argparse

# Paths
SQLSTORM_V1 = "/home/anqian/Desktop/my_lab/SQLStorm/v1.0/tpch/queries"
OUTPUT_SQL_DIR = "/home/anqian/Desktop/my_lab/workloads/SQLStorm"
OUTPUT_PLAN_DIR = "/home/anqian/Desktop/my_lab/workloads/explain_plans"
ID_MAP_FILE = "/home/anqian/Desktop/my_lab/workloads/new_sqlstorm_ids.txt"

MYSQL_CMD = ["mysql", "-h", "172.19.0.11", "-P", "4000", "-u", "root", "-D", "tpch_sf40", "--batch"]


def fix_for_tidb(sql: str) -> str:
    """Apply deterministic fixes for PostgreSQL→TiDB compatibility."""
    # 1. STRING_AGG(e, d ORDER BY c) → GROUP_CONCAT(e ORDER BY c SEPARATOR d)
    sql = re.sub(
        r'STRING_AGG\(\s*(.+?)\s*,\s*(.+?)\s+ORDER\s+BY\s+(.+?)\s*\)',
        r'GROUP_CONCAT(\1 ORDER BY \3 SEPARATOR \2)', sql, flags=re.IGNORECASE
    )

    # 2. Simple STRING_AGG(e, d) → GROUP_CONCAT(e SEPARATOR d)
    sql = re.sub(
        r'STRING_AGG\(\s*(.+?)\s*,\s*(.+?)\s*\)',
        r'GROUP_CONCAT(\1 SEPARATOR \2)', sql, flags=re.IGNORECASE
    )

    # 3. ILIKE → LOWER(x) LIKE LOWER(y)
    def replace_ilike(m):
        left = m.group(1).strip()
        right = m.group(2).strip()
        return f'LOWER({left}) LIKE LOWER({right})'
    sql = re.sub(r'(\S+(?:\s+IS\s+NOT\s+NULL|\s+IN\s*\([^)]+\))?)\s+ILIKE\s+(\'[^\']+\'|\S+(?:\s+IS\s+NOT\s+NULL)?)', replace_ilike, sql)

    # 4. a :: type → CAST(a AS type)
    sql = re.sub(r'(\S+)\s*::\s*(INTEGER|BIGINT|TEXT|VARCHAR\S*|NUMERIC\S*|FLOAT\S*|BOOLEAN|DATE|TIMESTAMP)', r'CAST(\1 AS \2)', sql)

    # 5. a || b → CONCAT(a, b) - handle nested cases
    for _ in range(10):
        new_sql = re.sub(r"(\S+)\s*\|\|\s*(\S+)", r"CONCAT(\1, \2)", sql)
        if new_sql == sql:
            break
        sql = new_sql

    # 6. FULL OUTER JOIN → skip (too complex, mark as skip)
    if re.search(r'FULL\s+OUTER\s+JOIN', sql, re.IGNORECASE):
        return None

    # 7. EXCEPT / INTERSECT → keep (TiDB 8.5 may support, otherwise will fail at EXPLAIN)
    # 8. Remove trailing semicolon
    sql = sql.strip().rstrip(';').strip()

    # 9. GENERATE_SERIES → skip (not supported)
    if re.search(r'GENERATE_SERIES', sql, re.IGNORECASE):
        return None

    return sql


def check_tidb_syntax(sql: str) -> bool:
    """Run EXPLAIN to check if TiDB accepts the syntax."""
    try:
        result = subprocess.run(
            MYSQL_CMD + ["-e", f"EXPLAIN FORMAT='verbose' {sql}"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return False
        if "ERROR" in result.stderr or "ERROR" in result.stdout:
            return False
        if "Unsupported" in result.stderr:
            return False
        return "id" in result.stdout and "estRows" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def get_explain_plan(sql: str) -> str:
    """Get EXPLAIN FORMAT='verbose' output."""
    try:
        result = subprocess.run(
            MYSQL_CMD + ["-e", f"EXPLAIN FORMAT='verbose' {sql}"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=5000, help='Max queries to validate')
    parser.add_argument('--start-id', type=int, default=1, help='Starting new query ID')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip queries already in OUTPUT_SQL_DIR')
    args = parser.parse_args()

    # Determine next ID
    existing = set()
    if args.skip_existing and os.path.isdir(OUTPUT_SQL_DIR):
        for f in os.listdir(OUTPUT_SQL_DIR):
            if f.endswith('.sql'):
                try:
                    existing.add(int(f.replace('.sql', '')))
                except ValueError:
                    pass

    next_id = max(existing) + 1 if existing else args.start_id
    print(f"Existing queries: {len(existing)}, starting from ID {next_id}")

    # Get source query files
    src_files = sorted([
        f for f in os.listdir(SQLSTORM_V1)
        if f.endswith('.sql') and f.replace('.sql', '').isdigit()
    ], key=lambda x: int(x.replace('.sql', '')))

    print(f"Source queries available: {len(src_files)}")

    os.makedirs(OUTPUT_SQL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PLAN_DIR, exist_ok=True)

    validated = 0
    skipped = 0
    failed = 0
    map_lines = []

    for fname in src_files:
        if validated >= args.limit:
            break

        src_id = int(fname.replace('.sql', ''))

        with open(os.path.join(SQLSTORM_V1, fname)) as f:
            raw_sql = f.read().strip()

        # Fix for TiDB
        fixed_sql = fix_for_tidb(raw_sql)
        if fixed_sql is None:
            skipped += 1
            continue

        # Validate syntax
        if not check_tidb_syntax(fixed_sql):
            failed += 1
            if failed % 100 == 0:
                print(f"  progress: {validated} valid, {failed} failed, {skipped} skipped")
            continue

        # Get EXPLAIN plan
        plan = get_explain_plan(fixed_sql)
        if not plan:
            failed += 1
            continue

        # Save
        sql_path = os.path.join(OUTPUT_SQL_DIR, f"{next_id}.sql")
        plan_path = os.path.join(OUTPUT_PLAN_DIR, f"{next_id}.txt")

        with open(sql_path, 'w') as f:
            f.write(fixed_sql.strip() + '\n')

        with open(plan_path, 'w') as f:
            f.write(f"-- Query: {next_id}\n")
            f.write(f"-- Source: SQLStorm v1.0 query {src_id}\n\n")
            f.write(plan)

        map_lines.append(f"{next_id}\t{src_id}")
        validated += 1
        next_id += 1

        if validated % 100 == 0:
            print(f"  progress: {validated} valid, {failed} failed, {skipped} skipped")

    # Save ID mapping
    with open(ID_MAP_FILE, 'w') as f:
        f.write("new_id\tsource_id\n")
        f.write('\n'.join(map_lines) + '\n')

    print(f"\nDone: {validated} validated, {failed} failed, {skipped} skipped")
    print(f"Next ID: {next_id}")
    print(f"ID map: {ID_MAP_FILE}")


if __name__ == '__main__':
    main()
