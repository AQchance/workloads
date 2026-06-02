import sys
import time
import mysql.connector

# ============================================================
# Configuration — modify these to match your MySQL setup
# ============================================================
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 4000,
    "user": "root",
    "password": "",
    "database": "tpch_sf40",
}

QUERY_DIR = "generated_queries"

# Most expensive queries per type: { "Q1": [line_numbers], ... }
TEST_PLAN = {
    "Q1":  [33, 35, 50],
    "Q2":  [1,  14, 39],
    "Q4":  [9,  46, 25],
    "Q6":  [21, 44, 38],
    "Q7":  [17, 28, 47],
    "Q9":  [1,  2,  3],
    "Q11": [8,  10, 15],
    "Q12": [9,  34, 2],
    "Q13": [1,  10, 11],
    "Q14": [23, 41, 3],
    "Q16": [11, 12, 13],
    "Q17": [1,  14, 22],
    "Q18": [47, 39, 49],
    "Q19": [45, 26, 2],
    "Q20": [38, 44, 31],
}


def main():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    total = sum(len(lines) for lines in TEST_PLAN.values())
    print(f"Test plan: {len(TEST_PLAN)} query types, {total} queries total")
    print(f"Connected to {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print("=" * 60)

    idx = 0
    for qtype, line_nums in TEST_PLAN.items():
        fname = f"{qtype}.sql"
        with open(f"{QUERY_DIR}/{fname}") as f:
            queries = [line.strip() for line in f if line.strip()]

        for ln in line_nums:
            idx += 1
            sql = queries[ln - 1]
            print(f"[{idx}/{total}] {qtype} line {ln}: ", end="", flush=True)
            start = time.perf_counter()

            try:
                cursor.execute(sql)
                _ = cursor.fetchall()
                elapsed = time.perf_counter() - start
                print(f"OK  {elapsed:.3f}s")
            except Exception as e:
                elapsed = time.perf_counter() - start
                print(f"FAILED after {elapsed:.3f}s")
                print(f"Error: {e}")
                print(f"Query: {sql[:200]}...")
                cursor.close()
                conn.close()
                sys.exit(1)

    cursor.close()
    conn.close()
    print("=" * 60)
    print(f"All {total} queries executed successfully.")


if __name__ == "__main__":
    main()
