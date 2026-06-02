#!/usr/bin/env python3
"""
Collect EXPLAIN ANALYZE for all 750 queries using round-robin across query templates.

Round-robin strategy: pick query N from Q1, then Q2, Q4, ..., Q20, then query N+1
from each, avoiding consecutive execution of the same template to reduce cache bias.

Writes results incrementally (append) to Q*_explain_analyze.txt.
Supports resume via a progress file: if interrupted, re-running skips completed queries.
"""

import argparse
import os
import re
import sys
import json
import time

import pymysql

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 4000,
    "user": "root",
    "password": "",
    "database": "tpch_sf40",
    "connect_timeout": 30,
    "charset": "utf8mb4",
}


def parse_sql_file(filepath: str) -> list[tuple[int, str]]:
    """Return [(line_number, sql), ...] for non-empty lines."""
    entries = []
    with open(filepath, "r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            sql = raw_line.strip()
            if sql:
                entries.append((line_no, sql))
    return entries


def make_round_robin(file_entries: dict[str, list[tuple[int, str]]]) -> list[tuple[str, int, int, str]]:
    """
    Build round-robin schedule across files.

    Returns: [(file_basename, file_index, line_no, sql), ...]
    file_index is the per-file sequential index (0..49).
    """
    # Normalize all files to the same length
    lengths = {f: len(v) for f, v in file_entries.items()}
    assert len(set(lengths.values())) == 1, f"Files have different query counts: {lengths}"

    num_per_file = list(lengths.values())[0]
    file_order = sorted(file_entries.keys())

    schedule = []
    for round_idx in range(num_per_file):
        for fname in file_order:
            line_no, sql = file_entries[fname][round_idx]
            schedule.append((fname, round_idx, line_no, sql))

    return schedule


def load_progress(progress_path: str) -> set:
    """Load set of completed (filename, file_index) tuples."""
    if not os.path.exists(progress_path):
        return set()
    with open(progress_path, "r") as f:
        data = json.load(f)
    return {tuple(item) for item in data}


def save_progress(progress_path: str, completed: set):
    with open(progress_path, "w") as f:
        json.dump(sorted(list(completed)), f)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)


def run_explain_analyze(conn, sql: str) -> str:
    """Execute EXPLAIN ANALYZE and return tabular output as string."""
    clean_sql = sql.rstrip(";").strip()
    explain_sql = f"EXPLAIN ANALYZE {clean_sql}"

    with conn.cursor() as cur:
        cur.execute(explain_sql)
        rows = cur.fetchall()
        # rows is list of tuples; join into text
        if not rows:
            return "(empty result)"

        # Get column names for header
        col_names = [desc[0] for desc in cur.description]
        header = "| " + " | ".join(col_names) + " |"

        lines = [header]
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Collect EXPLAIN ANALYZE plans with round-robin scheduling"
    )
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--host", default=DB_CONFIG["host"])
    parser.add_argument("--port", type=int, default=DB_CONFIG["port"])
    parser.add_argument("--user", default=DB_CONFIG["user"])
    parser.add_argument("--password", default=DB_CONFIG["password"])
    parser.add_argument("--database", default=DB_CONFIG["database"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print schedule without executing")
    args = parser.parse_args()

    DB_CONFIG["host"] = args.host
    DB_CONFIG["port"] = args.port
    DB_CONFIG["user"] = args.user
    DB_CONFIG["password"] = args.password
    DB_CONFIG["database"] = args.database

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir or os.path.join(script_dir, "generated_queries")
    output_dir = args.output_dir or os.path.join(script_dir, "explain_analyze_results")
    os.makedirs(output_dir, exist_ok=True)

    progress_path = os.path.join(output_dir, "_progress.json")

    if not os.path.isdir(input_dir):
        print(f"ERROR: input directory not found: {input_dir}")
        sys.exit(1)

    # Parse all SQL files
    sql_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".sql"))
    if not sql_files:
        print(f"No .sql files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(sql_files)} SQL files in {input_dir}")

    file_entries = {}
    for f in sql_files:
        entries = parse_sql_file(os.path.join(input_dir, f))
        # Use basename without extension as key, e.g. "Q1"
        base = os.path.splitext(f)[0]
        file_entries[base] = entries
        print(f"  {f}: {len(entries)} queries")

    total = sum(len(v) for v in file_entries.values())
    print(f"Total: {total} queries")

    schedule = make_round_robin(file_entries)
    print(f"Round-robin schedule: {len(schedule)} steps")
    print(f"Output directory: {output_dir}")

    if args.dry_run:
        for i, (fname, fidx, line_no, _) in enumerate(schedule[:20]):
            print(f"  [{i:>3}] {fname}  file-idx={fidx}  line={line_no}")
        print("  ...")
        sys.exit(0)

    # Load progress for resume
    completed = load_progress(progress_path)
    if completed:
        print(f"Resuming: {len(completed)}/{total} already done, {total - len(completed)} remaining")

    conn = pymysql.connect(**DB_CONFIG)
    print(f"Connected to TiDB: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print("-" * 60)

    start_time = time.time()

    try:
        for step_idx, (fname, fidx, line_no, sql) in enumerate(schedule):
            key = (fname, fidx)
            if key in completed:
                continue

            output_file = os.path.join(output_dir, f"{sanitize_filename(fname)}_explain_analyze.txt")

            shortened = sql[:80] + "..." if len(sql) > 80 else sql
            sys.stdout.write(f"[{step_idx + 1:>4}/{total}] {fname} line {line_no} ... ")
            sys.stdout.flush()

            try:
                result = run_explain_analyze(conn, sql)
            except Exception as e:
                # Reconnect on error and retry once
                print(f"\n  Connection error: {e}, reconnecting...")
                try:
                    conn.close()
                except Exception:
                    pass
                time.sleep(2)
                conn = pymysql.connect(**DB_CONFIG)
                try:
                    result = run_explain_analyze(conn, sql)
                except Exception as e2:
                    result = f"ERROR: {e2}"

            with open(output_file, "a") as out:
                out.write(f"--- Round {fidx + 1} | Line {line_no} | Step {step_idx + 1} ---\n")
                out.write(f"-- SQL: {sql}\n\n")
                out.write(result + "\n\n\n")

            completed.add(key)
            save_progress(progress_path, completed)

            elapsed = time.time() - start_time
            rate = (step_idx + 1) / elapsed if elapsed > 0 else 0
            remaining = total - len(completed)
            eta = remaining / rate if rate > 0 else 0
            print(f"OK  ({len(completed)}/{total}, {rate:.1f} q/min, ETA {eta/60:.0f}m)")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted. Progress saved ({len(completed)}/{total} done).")
        print(f"Resume by re-running: python {os.path.basename(__file__)}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        save_progress(progress_path, completed)
        progress_pct = len(completed) / total * 100
        print(f"\nDone. {len(completed)}/{total} ({progress_pct:.1f}%) queries collected.")


if __name__ == "__main__":
    main()
