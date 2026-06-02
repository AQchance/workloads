#!/home/anqian/Desktop/my_lab/workloads/venv/bin/python3
"""
Execute 750 SQL queries against TiDB, respecting arrival times and concurrency limits.

Usage:
    python3 run_queries.py [--concurrency N] [--sql-file PATH] [--times-file PATH]
"""

import argparse
import asyncio
import sys
import time
import pymysql

# --- Database config ---
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 4000,
    "user": "root",
    "password": "",
    "database": "tpch_sf40",
    "connect_timeout": 30,
    "read_timeout": 300,
    "write_timeout": 300,
    "charset": "utf8mb4",
}

# --- File paths ---
SQL_FILE = "all_queries_shuffled.sql"
TIMES_FILE = "arrival_times.txt"

MAX_CONCURRENCY = 2


def load_data(sql_path, times_path):
    with open(sql_path) as f:
        queries = [line.strip() for line in f if line.strip()]
    times = []
    with open(times_path) as f:
        next(f)  # skip header
        for line in f:
            _, t = line.strip().split("\t")
            times.append(float(t))
    assert len(queries) == len(times), f"Query count mismatch: {len(queries)} vs {len(times)}"
    return queries, times


def execute_query(conn, sql):
    start = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql)
        # Fetch all rows to ensure the query fully completes
        _ = cur.fetchall()
    elapsed = time.perf_counter() - start
    return elapsed


async def worker(query_id, sql, arrival_time, semaphore, print_lock, stats):
    """Wait until arrival time, then execute with concurrency control."""
    # Wait until this query's scheduled arrival time
    now = time.time() - stats["t0"]
    delay = arrival_time - now
    if delay > 0:
        await asyncio.sleep(delay)

    async with semaphore:
        try:
            conn = pymysql.connect(**DB_CONFIG)
            try:
                elapsed = await asyncio.to_thread(execute_query, conn, sql)
                stats["completed"] += 1
                async with print_lock:
                    print(f"[{stats['completed']:>4}/{stats['total']}] query {query_id:>4}  OK   {elapsed:.3f}s")
            finally:
                conn.close()
        except Exception as e:
            stats["completed"] += 1
            async with print_lock:
                print(f"[{stats['completed']:>4}/{stats['total']}] query {query_id:>4}  ERROR  ({e})")


async def main():
    parser = argparse.ArgumentParser(description="Run SQL queries with arrival-time scheduling")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY, help="Max concurrent queries")
    parser.add_argument("--sql-file", type=str, default=SQL_FILE)
    parser.add_argument("--times-file", type=str, default=TIMES_FILE)
    parser.add_argument("--host", type=str, default=DB_CONFIG["host"])
    parser.add_argument("--port", type=int, default=DB_CONFIG["port"])
    parser.add_argument("--user", type=str, default=DB_CONFIG["user"])
    parser.add_argument("--password", type=str, default=DB_CONFIG["password"])
    parser.add_argument("--database", type=str, default=DB_CONFIG["database"])
    args = parser.parse_args()

    DB_CONFIG["host"] = args.host
    DB_CONFIG["port"] = args.port
    DB_CONFIG["user"] = args.user
    DB_CONFIG["password"] = args.password
    DB_CONFIG["database"] = args.database

    queries, times = load_data(args.sql_file, args.times_file)
    total = len(queries)

    semaphore = asyncio.Semaphore(args.concurrency)
    print_lock = asyncio.Lock()
    stats = {"t0": time.time(), "completed": 0, "total": total}

    print(f"Starting {total} queries, concurrency={args.concurrency}")
    print(f"Database: {DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print("-" * 60)

    tasks = [
        asyncio.create_task(worker(i + 1, queries[i], times[i], semaphore, print_lock, stats))
        for i in range(total)
    ]

    await asyncio.gather(*tasks)

    elapsed_total = time.time() - stats["t0"]
    print("-" * 60)
    print(f"Done. {total} queries in {elapsed_total:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
