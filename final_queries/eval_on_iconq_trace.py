"""
Evaluate trained TabPFN→BiLSTM on ICONQ scheduling trace.
Compares: predicted slowdown vs actual slowdown from the trace.
"""
import os, csv, json, math, numpy as np, torch, torch.nn as nn

DIR = '/home/anqian/code/python/workloads/final_queries'
RESOURCE_PATH = '/home/anqian/code/python/workloads/collect_concurrent/tabpfn_258_predictions_oof.json'
MODEL_PATH = os.path.join(DIR, 'bilstm_tabpfn.pt')
NORM_PATH = os.path.join(DIR, 'bilstm_tabpfn_norm.npz')
TRACE_PATH = '/home/anqian/code/python/workloads/collect_concurrent/iconq_k8_trace.csv'

DIMS = ['mem', 'disk', 'net', 'lat', 'cpures']
SEED = 42


def resource_conflict(t, c):
    t, c = np.array(t), np.array(c)
    return list(np.minimum(t, c) / np.maximum(np.abs(t) + np.abs(c) + 1e-8, 1e-8))


class BiLSTM(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
    def forward(self, X, lengths):
        x = self.emb(X)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.lstm(packed)
        return self.pred(torch.cat([hn[-2], hn[-1]], dim=-1)).squeeze(-1)


def main():
    print('=' * 55)
    print('Evaluating TabPFN→BiLSTM on ICONQ K8 Trace')
    print('=' * 55)

    # Load resources + model
    with open(RESOURCE_PATH) as f: cache = json.load(f)
    norm = np.load(NORM_PATH)
    X_mean, X_std = norm['X_mean'], norm['X_std']
    y_mean, y_std = norm['y_mean'], norm['y_std']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTM(input_dim=17).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f'  Model loaded, resources: {len(cache)} queries')

    # Load trace
    trace = []
    with open(TRACE_PATH) as f:
        for row in csv.DictReader(f):
            try:
                t0 = float(row['start']); rt = float(row['runtime'])
                tf = float(row['finish']) if row.get('finish', '').strip() else t0 + rt
            except ValueError:
                continue
            trace.append({
                'qid': row['qid'],
                'start': t0, 'finish': tf, 'runtime': rt,
                'status': row['status'],
                'old_slowdown': float(row.get('predicted_slowdown', 0) or 0),
            })
    print(f'  Trace: {len(trace)} events')

    # Build timeline for concurrency detection
    timeline = [(r['start'], r['finish'], r['qid']) for r in trace]
    qid_info = {q: (s, e) for s, e, q in timeline}

    results = []
    for i, r in enumerate(trace):
        if r['status'] != 'ok' or r['qid'] not in cache:
            continue
        si, ei, qi = r['start'], r['finish'], r['qid']
        rti = r['runtime']

        # Find concurrent peers
        concurrent_peers = []
        for j, (sj, ej, qj) in enumerate(timeline):
            if i != j and sj < ei and ej > si and qj in cache:
                concurrent_peers.append((sj, qj))

        if not concurrent_peers:
            continue

        # Build 17-dim sequence
        pred_i = cache[qi]
        serial_lat = pred_i.get('serial_lat_s', max(rti, 0.5))
        qv = [pred_i[d] for d in DIMS]
        tr_ = [pred_i[d] for d in DIMS]

        seq = []
        for sj, qj in concurrent_peers:
            pred_j = cache[qj]
            ovv = [pred_j[d] for d in DIMS]
            oc = [pred_j[d] for d in DIMS]
            c = resource_conflict(tr_, oc)
            seq.append(qv + ovv + [si - sj, 1.0 if sj < si else 0.0] + c)

        if not seq:
            continue

        # Normalize + predict
        X = np.stack(seq, dtype=np.float32)  # [L, 17]
        X = (X - X_mean) / X_std
        X_t = torch.FloatTensor(X).unsqueeze(0).to(device)  # [1, L, 17]
        L = torch.LongTensor([len(seq)]).to(device)

        with torch.no_grad():
            pred_z = model(X_t, L).cpu().item()

        pred_ratio = max(math.exp(pred_z * y_std + y_mean) - 1, 0.01)
        actual_ratio = rti / serial_lat
        qe = max(pred_ratio / actual_ratio, actual_ratio / pred_ratio)

        results.append({
            'qid': qi, 'n_peers': len(seq),
            'serial': serial_lat, 'runtime': rti,
            'actual_slowdown': actual_ratio,
            'pred_slowdown': pred_ratio,
            'qe': qe,
            'old_slowdown': r['old_slowdown'],
        })

    qes = sorted([r['qe'] for r in results]); n = len(qes)
    old_qes = sorted([
        max(r['old_slowdown'] / r['actual_slowdown'],
            r['actual_slowdown'] / max(r['old_slowdown'], 0.01))
        for r in results]); n2 = len(old_qes)

    print(f'\n  Valid samples: {n}')
    print(f'\n  {"Model":<20} {"P50":>7} {"P90":>7} {"P95":>7} {"P99":>7}')
    print(f'  {"-"*48}')
    print(f'  {"Old ICONQ Sched":<20} {old_qes[n2//2]:.2f}x {old_qes[int(n2*.9)]:.2f}x '
          f'{old_qes[int(n2*.95)]:.2f}x {old_qes[int(n2*.99)]:.2f}x')
    print(f'  {"TabPFN->BiLSTM":<20} {qes[n//2]:.2f}x {qes[int(n*.9)]:.2f}x '
          f'{qes[int(n*.95)]:.2f}x {qes[int(n*.99)]:.2f}x')

    # Per-query match
    tpf_win = sum(1 for r in results
                  if max(r['old_slowdown'] / r['actual_slowdown'],
                         r['actual_slowdown'] / max(r['old_slowdown'], 0.01)) > r['qe'])
    print(f'\n  TabPFN better than old: {tpf_win}/{n} ({tpf_win/n*100:.0f}%)')

    print(f'\n{"="*55}')


if __name__ == '__main__':
    main()
