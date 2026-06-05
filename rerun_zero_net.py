"""Re-run zero-net queries then continue full collection."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_resources as cr

out_dir = '/home/anqian/Desktop/my_lab/workloads/cgroup_resources'
zero_qids = ['4076','4383','4676','4690','4722','4883','5076','5089','5123','5343','5389','5456','5459','5689','5765']

print(f"=== Phase 1: Re-running {len(zero_qids)} zero-net queries ===\n")
for i, qid in enumerate(zero_qids):
    print(f"[{i+1}/{len(zero_qids)}] Q{qid} ...", end=" ", flush=True)
    data = cr.collect_one_query(qid)
    if data is None:
        print("FAIL")
    elif data.get('status') == 'timeout':
        print(f"TIMEOUT ({data['elapsed_s']:.0f}s)")
        with open(os.path.join(out_dir, f'{qid}.json'), 'w') as f:
            json.dump(data, f)
    else:
        with open(os.path.join(out_dir, f'{qid}.json'), 'w') as f:
            json.dump(data, f)
        nd = sum(data['network_delta_bytes'].values())
        print(f"OK net={nd/1e9:.1f}GB")

print(f"\n=== Phase 2: Continuing full collection ===\n")
cr.main()
