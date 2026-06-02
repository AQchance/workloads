#!/usr/bin/env python3
"""
Collect execution plans for all queries in generated_queries/*.sql.

Usage:
  python collect_plans.py
  python collect_plans.py --host 127.0.0.1 --port 4000 --user root

For each Q<N>.sql, runs EXPLAIN on every query and writes results to Q<N>_plan.txt,
annotated with the original line number.
"""

import argparse
import os
import re
import sys

import pymysql


def parse_sql_file(filepath: str) -> list[tuple[int, str]]:
    """Parse a .sql file, returning a list of (line_number, query_sql)."""
    entries = []
    with open(filepath, "r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            sql = raw_line.strip()
            if sql:
                entries.append((line_no, sql))
    return entries


def get_plan(cursor, sql: str) -> str:
    """Run EXPLAIN on a query and return the plan text."""
    clean_sql = sql.rstrip(";").strip()
    explain_sql = f"EXPLAIN {clean_sql}"
    cursor.execute(explain_sql)
    rows = cursor.fetchall()
    # pymysql returns tuples; join all columns with " | " for readability
    lines = []
    for row in rows:
        lines.append(" | ".join(str(col) for col in row))
    return "\n".join(lines)


def sanitize_filename(name: str) -> str:
    """Remove dangerous characters from filenames."""
    return re.sub(r"[^\w\-.]", "_", name)


def main():
    parser = argparse.ArgumentParser(
        description="Collect EXPLAIN plans for all queries in generated_queries/"
    )
    parser.add_argument("--input-dir", default=None,
                        help="Directory containing .sql files (default: script dir/generated_queries)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for plan output files (default: same as input-dir)")
    parser.add_argument("--dbname", default="tpch_sf40", help="Database name")
    parser.add_argument("--host", default="localhost", help="Database host")
    parser.add_argument("--port", type=int, default=4000, help="Database port")
    parser.add_argument("--user", default="root", help="Database user")
    parser.add_argument("--password", default="", help="Database password")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir or os.path.join(script_dir, "generated_queries")
    output_dir = args.output_dir or input_dir

    if not os.path.isdir(input_dir):
        print(f"ERROR: input directory not found: {input_dir}")
        sys.exit(1)

    sql_files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".sql") and not f.endswith("_plan.sql")
    )
    if not sql_files:
        print(f"No .sql files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(sql_files)} SQL file(s) in {input_dir}")
    print(f"Connecting to TiDB: host={args.host} port={args.port} db={args.dbname} user={args.user}")

    conn = pymysql.connect(
        database=args.dbname,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
    )
    cursor = conn.cursor()

    try:
        for sql_file in sql_files:
            filepath = os.path.join(input_dir, sql_file)
            base_name = os.path.splitext(sql_file)[0]
            output_name = f"{sanitize_filename(base_name)}_plan.txt"
            output_path = os.path.join(output_dir, output_name)

            print(f"\nProcessing {sql_file} -> {output_name}")

            entries = parse_sql_file(filepath)
            if not entries:
                print(f"  No valid queries found, skipping.")
                continue

            with open(output_path, "w") as out:
                for line_no, sql in entries:
                    try:
                        plan = get_plan(cursor, sql)
                    except Exception as e:
                        plan = f"ERROR: {e}"

                    out.write(f"--- Line {line_no} ---\n")
                    out.write(sql + "\n\n")
                    out.write("Execution Plan:\n")
                    out.write(plan + "\n")
                    out.write("\n\n")

                    shortened = sql[:80] + "..." if len(sql) > 80 else sql
                    status = "OK" if not plan.startswith("ERROR") else "FAIL"
                    print(f"  Line {line_no}: [{status}] {shortened}")

            print(f"  -> Wrote {len(entries)} plan(s) to {output_path}")

    finally:
        cursor.close()
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
