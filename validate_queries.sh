#!/bin/bash
# Validate all SQL queries against TiDB using EXPLAIN
# Usage: bash validate_queries.sh

SQL_DIR="/home/anqian/Desktop/my_lab/workloads/SQLStorm"
PASS_DIR="/home/anqian/Desktop/my_lab/workloads/SQLStorm_passed"
FAIL_DIR="/home/anqian/Desktop/my_lab/workloads/SQLStorm_failed"
MYSQL_CMD="docker exec tidb1 mysql -h 127.0.0.1 -P 4000 -u root -D tpch_sf40"

mkdir -p "$PASS_DIR" "$FAIL_DIR"

total=0
passed=0
failed=0

for f in "$SQL_DIR"/*.sql; do
    fname=$(basename "$f")
    total=$((total + 1))

    sql=$(cat "$f")
    result=$($MYSQL_CMD -e "EXPLAIN $sql" 2>&1)

    if echo "$result" | grep -q "ERROR"; then
        err_msg=$(echo "$result" | grep "ERROR" | head -1 | cut -c1-150)
        echo "[FAIL $total] $fname: $err_msg"
        cp "$f" "$FAIL_DIR/$fname"
        failed=$((failed + 1))
    else
        cp "$f" "$PASS_DIR/$fname"
        passed=$((passed + 1))
    fi

    # Progress every 100
    if [ $((total % 100)) -eq 0 ]; then
        echo "--- Progress: $total/3000, pass=$passed fail=$failed ---"
    fi
done

echo ""
echo "============================================"
echo "Validation complete: $passed passed, $failed failed out of $total"
echo "Passed: $PASS_DIR"
echo "Failed: $FAIL_DIR"
