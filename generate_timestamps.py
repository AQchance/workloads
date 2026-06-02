#!/usr/bin/env python3
"""Generate 750 arrival timestamps with Poisson-distributed inter-arrival intervals (mean = 5s)."""
import random
import math

random.seed(42)
n = 750
mean_interval = 5.0  # seconds

intervals = [random.expovariate(1.0 / mean_interval) for _ in range(n)]
cumulative = [0.0] * n
for i in range(1, n):
    cumulative[i] = cumulative[i - 1] + intervals[i]

out = "/home/anqian/Desktop/my_lab/workloads/arrival_times.txt"
with open(out, "w") as f:
    f.write("query_order\tarrival_time_s\n")
    for i, t in enumerate(cumulative):
        f.write(f"{i + 1}\t{t:.3f}\n")

print(f"Done: {n} timestamps written to {out}")
print(f"Total time span: {cumulative[-1]:.1f}s ({cumulative[-1]/60:.1f} min)")
