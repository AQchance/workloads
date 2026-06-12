import random
with open('/home/anqian/Desktop/my_lab/workloads/collect_concurrent/selected_queries.txt') as f:
    queries = [line.strip() for line in f if line.strip()]
random.seed(42)
all_3000 = []
for r in range(3):
    q = queries[:]; random.shuffle(q); all_3000.extend(q)
out = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/query_order_k3.txt'
with open(out, 'w') as f: f.write('\n'.join(all_3000) + '\n')
print(f"Saved {len(all_3000)} queries to {out}")
