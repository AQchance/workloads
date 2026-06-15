"""
Shift start times in a trace CSV by a fixed offset for a range of rows.

Usage: python shift_trace.py <trace.csv> <start_row> <end_row> <offset_seconds>

Example: python shift_trace.py trace_2_fixed.csv 358 1549 8000
  → Adds 8000s to rows 358 through 1549 (inclusive, 1-indexed)
"""

import csv, sys


def shift_trace(filepath, start_row, end_row, offset):
    with open(filepath) as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        rows = [dict(x) for x in r]

    for i in range(start_row - 1, end_row):  # convert to 0-indexed
        rows[i]['start'] = round(float(rows[i]['start']) + offset, 1)

    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Shifted rows {start_row}-{end_row} by +{offset}s in {filepath}")


if __name__ == '__main__':
    shift_trace(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
