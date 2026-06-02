#!/bin/bash
# Merge all SQL files from generated_queries/ into one shuffled SQL file.

INPUT_DIR="$(dirname "$0")/generated_queries"
OUTPUT_FILE="$(dirname "$0")/all_queries_shuffled.sql"

# Use Python with fixed seed for reproducibility
cat "$INPUT_DIR"/*.sql | python3 -c "
import random, sys
lines = sys.stdin.readlines()
random.Random(42).shuffle(lines)
sys.stdout.writelines(lines)
" > "$OUTPUT_FILE"

echo "Done: $(wc -l < "$OUTPUT_FILE") queries written to $OUTPUT_FILE"
