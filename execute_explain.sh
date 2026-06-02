#!/bin/bash
# Execute EXPLAIN for all SQLStorm queries against TiDB.
#
# Unlike execute_queries.sh which runs EXPLAIN ANALYZE, this script runs
# plain EXPLAIN to collect estimated (not actual) execution plans.
#
# Output files:
#   explain_results.txt                — query_id, status (success/error/timeout/OOM)
#   explain_plans/<query_id>.txt       — EXPLAIN output per query

SQL_DIR="/home/anqian/Desktop/my_lab/workloads/SQLStorm"
RESULT_FILE="/home/anqian/Desktop/my_lab/workloads/explain_results.txt"
PLANS_DIR="/home/anqian/Desktop/my_lab/workloads/explain_plans"
TIMEOUT_SEC=300 # 5 minutes (EXPLAIN is much faster than EXPLAIN ANALYZE)

MYSQL_BASE="docker exec tidb1 mysql -h 127.0.0.1 -P 4000 -u root -D tpch_sf40"

mkdir -p "$PLANS_DIR"

# Build query list from files in SQL_DIR, sorted numerically
QUERY_LIST=$(ls "$SQL_DIR"/*.sql | xargs -n1 basename | sed 's/\.sql$//' | sort -n)
total=$(echo "$QUERY_LIST" | wc -l)

# Resume: collect already-done queries
declare -A DONE
if [ -f "$RESULT_FILE" ]; then
  while read -r qnum status; do
    DONE[$qnum]=$status
  done <"$RESULT_FILE"
  echo "Resuming: ${#DONE[@]} queries already completed"
fi

count=0
oom_flag=0
consecutive_errors=0

echo "Starting EXPLAIN for $total queries (timeout=${TIMEOUT_SEC}s)..."
echo "EXPLAIN output: $PLANS_DIR"
echo "Results: $RESULT_FILE"
echo "============================================"

for qnum in $QUERY_LIST; do
  count=$((count + 1))

  # Skip already done
  if [ -n "${DONE[$qnum]}" ]; then
    continue
  fi

  sql_file="$SQL_DIR/${qnum}.sql"
  sql=$(cat "$sql_file")

  # OOM cooldown: wait 20s after an OOM before next query
  if [ $oom_flag -eq 1 ]; then
    echo "[$count/$total] $qnum: cooling down after OOM (20s)..."
    sleep 20
    oom_flag=0
  fi

  # Execute EXPLAIN with timeout
  start_time=$(date +%s.%N)

  result=$(timeout $TIMEOUT_SEC docker exec tidb1 mysql -h 127.0.0.1 -P 4000 -u root -D tpch_sf40 --batch -e "EXPLAIN $sql" 2>&1)
  exit_code=$?

  end_time=$(date +%s.%N)
  elapsed=$(echo "$end_time - $start_time" | bc)

  if [ $exit_code -eq 124 ]; then
    status="timeout"
    echo "[$count/$total] $qnum: TIMEOUT (>${TIMEOUT_SEC}s)"
    consecutive_errors=0

  elif echo "$result" | grep -qiE "out of memory|memory exceeded|memory limit|alloc|OOM"; then
    status="OOM"
    oom_flag=1
    consecutive_errors=$((consecutive_errors + 1))
    echo "[$count/$total] $qnum: OOM"

  elif echo "$result" | grep -qiE "server has gone away|lost connection|Can't connect|ERROR 2003|ERROR 2006|ERROR 2013"; then
    status="OOM"
    oom_flag=1
    consecutive_errors=$((consecutive_errors + 1))
    echo "[$count/$total] $qnum: OOM (server crash)"

    if [ $consecutive_errors -ge 2 ]; then
      echo "[$count/$total] *** Consecutive failures detected, restarting TiDB ***"
      docker exec tidb1 bash /root/tidb_start.sh 2>&1 | tail -3
      sleep 30
      echo "[$count/$total] *** TiDB restart complete, resuming... ***"
      consecutive_errors=0
      oom_flag=0
    fi

  elif echo "$result" | grep -qiE "ERROR"; then
    status="error"
    consecutive_errors=$((consecutive_errors + 1))
    echo "[$count/$total] $qnum: ERROR - $(echo "$result" | grep -i ERROR | head -1 | cut -c1-100)"

    if [ $consecutive_errors -ge 2 ]; then
      echo "[$count/$total] *** Consecutive errors, restarting TiDB ***"
      docker exec tidb1 bash /root/tidb_start.sh 2>&1 | tail -3
      sleep 30
      echo "[$count/$total] *** TiDB restart complete, resuming... ***"
      consecutive_errors=0
      oom_flag=0
    fi

  elif [ $exit_code -ne 0 ]; then
    status="error"
    consecutive_errors=$((consecutive_errors + 1))
    echo "[$count/$total] $qnum: EXIT_CODE=$exit_code"

  else
    # Success — save EXPLAIN output with query_id as filename
    status="success"
    consecutive_errors=0

    {
      echo "-- Query: $qnum"
      echo "-- Execution time: ${elapsed}s"
      echo
      echo "$result"
    } >"$PLANS_DIR/${qnum}.txt"

    echo "[$count/$total] $qnum: OK (${elapsed}s)"
  fi

  echo "$qnum $status" >>"$RESULT_FILE"

  # Progress every 50
  if [ $((count % 50)) -eq 0 ]; then
    success_n=$(grep -c " success$" "$RESULT_FILE" 2>/dev/null || echo 0)
    timeout_n=$(grep -c " timeout$" "$RESULT_FILE" 2>/dev/null || echo 0)
    oom_n=$(grep -c " OOM$" "$RESULT_FILE" 2>/dev/null || echo 0)
    error_n=$(grep -c " error$" "$RESULT_FILE" 2>/dev/null || echo 0)
    echo "--- Progress: $count/$total | success=$success_n timeout=$timeout_n OOM=$oom_n error=$error_n ---"
  fi

done

# Final summary
success_n=$(grep -c " success$" "$RESULT_FILE" 2>/dev/null || echo 0)
timeout_n=$(grep -c " timeout$" "$RESULT_FILE" 2>/dev/null || echo 0)
oom_n=$(grep -c " OOM$" "$RESULT_FILE" 2>/dev/null || echo 0)
error_n=$(grep -c " error$" "$RESULT_FILE" 2>/dev/null || echo 0)

echo ""
echo "============================================"
echo "EXPLAIN collection complete!"
echo "  Success: $success_n"
echo "  Timeout: $timeout_n"
echo "  OOM:     $oom_n"
echo "  Error:   $error_n"
echo "  Total:   $total"
echo "Results:   $RESULT_FILE"
echo "Plans:     $PLANS_DIR"
