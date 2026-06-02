#!/bin/bash
# Execute EXPLAIN ANALYZE for all SQLStorm queries against TiDB.
#
# Output files:
#   sqlstorm_results.txt              — query_number, status (success/OOM/timeout/error)
#   explain_analyze/<qnum>.txt        — EXPLAIN ANALYZE output per query
#
# Status values:
#   success   = EXPLAIN ANALYZE completed
#   timeout   = exceeded 600s (10 min) limit
#   OOM       = out of memory / server crash
#   error     = other execution error
#
# DB crash recovery: if an OOM is followed by an error on the next query,
# the script restarts TiDB via "bash tidb_start.sh" inside the container.

SQL_DIR="/home/anqian/Desktop/my_lab/workloads/SQLStorm"
# Query list: dynamically build from SQLStorm directory
# This handles both old and new query IDs.
SQL_DIR="/home/anqian/Desktop/my_lab/workloads/SQLStorm"
RESULT_FILE="/home/anqian/Desktop/my_lab/workloads/sqlstorm_results.txt"
ANALYZE_DIR="/home/anqian/Desktop/my_lab/workloads/explain_analyze_results"
PLAN_DIR="/home/anqian/Desktop/my_lab/workloads/explain_plans"
MIN_QUERY=25247  # start from first new query ID
TIMEOUT_SEC=600 # 10 minutes

MYSQL_BASE="docker exec tidb1 mysql -h 127.0.0.1 -P 4000 -u root -D tpch_sf40"

mkdir -p "$ANALYZE_DIR"
mkdir -p "$PLAN_DIR"

# Resume: collect already-done query numbers
declare -A DONE
if [ -f "$RESULT_FILE" ]; then
  while read -r qnum status; do
    DONE[$qnum]=$status
  done <"$RESULT_FILE"
  echo "Resuming: ${#DONE[@]} queries already completed"
fi

# Build query list from all .sql files in SQL_DIR
QUERY_LIST=$(ls "$SQL_DIR"/*.sql 2>/dev/null | xargs -n1 basename | sed 's/\.sql//' | sort -n | awk -v min="$MIN_QUERY" '$1 >= min')
total=$(echo "$QUERY_LIST" | grep -c .)
count=${#DONE[@]}
oom_flag=0
consecutive_errors=0

echo "Starting EXPLAIN ANALYZE for $total new queries (timeout=${TIMEOUT_SEC}s)..."
echo "EXPLAIN ANALYZE output: $ANALYZE_DIR"
echo "Results: $RESULT_FILE"
echo "============================================"

# Write query list to temp file (avoid subshell pipe issues)
TEMP_LIST=$(mktemp)
echo "$QUERY_LIST" > "$TEMP_LIST"

while IFS= read -r qnum; do
  [ -z "$qnum" ] && continue

  count=$((count + 1))

  # Skip already done
  if [ -n "${DONE[$qnum]}" ]; then
    continue
  fi

  sql_file="$SQL_DIR/${qnum}.sql"
  if [ ! -f "$sql_file" ]; then
    echo "$qnum error" >>"$RESULT_FILE"
    continue
  fi

  sql=$(cat "$sql_file")

  # OOM cooldown: wait 20s after an OOM before next query
  if [ $oom_flag -eq 1 ]; then
    echo "[$count/$total] $qnum: cooling down after OOM (20s)..."
    sleep 20
    oom_flag=0
  fi

  # Execute EXPLAIN ANALYZE with timeout
  start_time=$(date +%s.%N)

  result=$(timeout $TIMEOUT_SEC docker exec tidb1 mysql -h 127.0.0.1 -P 4000 -u root -D tpch_sf40 --batch -e "EXPLAIN ANALYZE $sql" 2>&1)
  exit_code=$?

  end_time=$(date +%s.%N)
  elapsed=$(echo "$end_time - $start_time" | bc)

  if [ $exit_code -eq 124 ]; then
    # Timeout
    status="timeout"
    echo "[$count/$total] $qnum: TIMEOUT (>${TIMEOUT_SEC}s)"
    consecutive_errors=0

  elif echo "$result" | grep -qiE "out of memory|memory exceeded|memory limit|alloc|OOM"; then
    # OOM detected in error message
    status="OOM"
    oom_flag=1
    consecutive_errors=$((consecutive_errors + 1))
    echo "[$count/$total] $qnum: OOM"

  elif echo "$result" | grep -qiE "server has gone away|lost connection|Can't connect|ERROR 2003|ERROR 2006|ERROR 2013"; then
    # Server crash — treat as OOM
    status="OOM"
    oom_flag=1
    consecutive_errors=$((consecutive_errors + 1))
    echo "[$count/$total] $qnum: OOM (server crash)"

    # If we had a previous OOM, restart TiDB
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

    # If consecutive errors after OOM, restart TiDB
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
    # Success — save EXPLAIN ANALYZE output
    status="success"
    consecutive_errors=0

    # Save the EXPLAIN ANALYZE output
    {
      echo "-- Query: $qnum"
      echo "-- Execution time: ${elapsed}s"
      echo
      echo "$result"
    } >"$ANALYZE_DIR/${qnum}.txt"

    # Also collect EXPLAIN plan (VERBOSE format, for GNN model input)
    if [ ! -f "$PLAN_DIR/${qnum}.txt" ]; then
      explain_result=$(docker exec tidb1 mysql -h 127.0.0.1 -P 4000 -u root -D tpch_sf40 --batch -e "EXPLAIN FORMAT='verbose' $sql" 2>&1)
      if [ $? -eq 0 ] && ! echo "$explain_result" | grep -qi "ERROR"; then
        {
          echo "-- Query: $qnum"
          echo "-- Execution time: ${elapsed}s"
          echo
          echo "$explain_result"
        } >"$PLAN_DIR/${qnum}.txt"
      fi
    fi

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

done <"$TEMP_LIST"
rm -f "$TEMP_LIST"

# Final summary
success_n=$(grep -c " success$" "$RESULT_FILE" 2>/dev/null || echo 0)
timeout_n=$(grep -c " timeout$" "$RESULT_FILE" 2>/dev/null || echo 0)
oom_n=$(grep -c " OOM$" "$RESULT_FILE" 2>/dev/null || echo 0)
error_n=$(grep -c " error$" "$RESULT_FILE" 2>/dev/null || echo 0)

echo ""
echo "============================================"
echo "EXPLAIN ANALYZE collection complete!"
echo "  Success: $success_n"
echo "  Timeout: $timeout_n"
echo "  OOM:     $oom_n"
echo "  Error:   $error_n"
echo "  Total:   $total"
echo "Results:   $RESULT_FILE"
echo "Output:    $ANALYZE_DIR"
