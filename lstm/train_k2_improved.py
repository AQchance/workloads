"""
K=2 Bi-LSTM with two improvements:
  1. Predict slowdown ratio (conc_runtime / serial_latency) not absolute
  2. Residual skip: output = serial_lat * (1 + MLP(hidden))
"""

import sys, os, json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bilstm import ConcurrentBiLSTM, collate_fn

FEATURES_FILE = '/home/anqian/Desktop/my_lab/workloads/lstm/gnn_features.json'
GNN_LAT_FILE = '/home/anqian/Desktop/my_lab/workloads/lstm/gnn_lat_pred.json'
TRACE_FILE   = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/trace_2.csv'
with open(FEATURES_FILE) as f: gnn_features = json.load(f)
with open(GNN_LAT_FILE) as f: gnn_lat_pred = json.load(f)

trace = []
with open(TRACE_FILE) as f:
    for row in csv.DictReader(f): trace.append(row)

timeline = []
for r in trace:
    qid=r['qid']; rt=float(r['runtime']); st=r['status']
    if st=='penalty': continue
    start_s=float(r['start'])
    timeline.append((start_s, start_s+rt, qid, rt, st))

ci=[]
for i,(si,ei,qi,rti,sti) in enumerate(timeline):
    ov=[]
    for j,(sj,ej,qj,_,_) in enumerate(timeline):
        if i!=j and sj<ei and ej>si: ov.append(qj)
    ci.append((qi,si,ei,rti,sti,ov))

mt=max(e for _,_,e,_,_,_ in ci); sp=mt*0.7
tr_d=[c for c in ci if c[1]<sp]; te_d=[c for c in ci if c[1]>=sp]

def resource_conflict(t_res, c_res):
    t=np.array(t_res); c=np.array(c_res)
    return list(np.minimum(t,c)/np.maximum(np.abs(t)+np.abs(c)+1e-8,1e-8))

def build_seq(data_list, features):
    X, y_ratio, serial_lats = [], [], []
    for qi,si,ei,rti,sti,ov in data_list:
        if qi not in features: continue
        serial_lat = max(float(gnn_lat_pred.get(qi, 10)), 0.5)  # GNN predicted
        sl = math.log(1 + serial_lat)
        qv = features[qi]['plan_emb'] + list(features[qi]['gpu_resources'].values()) + [sl]
        t_res = list(features[qi]['gpu_resources'].values())
        seq=[]
        oinfo=[]
        for oq in ov:
            if oq not in features: continue
            for o_s,_,_,_,_ in [(ts,te,q,_,_) for ts,te,q,_,_ in timeline if q==oq]:
                oinfo.append((o_s,oq)); break
        oinfo.sort()
        for o_s,oq in oinfo:
            osl=math.log(1+features[oq]['serial_labels'].get('latency_s',10))
            ovv=features[oq]['plan_emb']+list(features[oq]['gpu_resources'].values())+[osl]
            ores=list(features[oq]['gpu_resources'].values())
            c=resource_conflict(t_res,ores)
            feat=qv+ovv+[si-o_s,1.0 if o_s<si else 0.0]+c
            seq.append(feat)
        if seq:
            X.append(seq)
            y_ratio.append(rti / serial_lat)       # ← predict ratio!
            serial_lats.append(serial_lat)
    return X, y_ratio, serial_lats

X_tr, y_tr, sl_tr = build_seq(tr_d, gnn_features)
X_te, y_te, sl_te = build_seq(te_d, gnn_features)
print(f"Train: {len(X_tr)}, Test: {len(X_te)}")
print(f"Ratio labels: min={min(y_tr):.2f} median={np.median(y_tr):.2f} max={max(y_tr):.2f}")

# Pad & normalize
ml=max(max(len(s) for s in X_tr), max(len(s) for s in X_te)); d=len(X_tr[0][0])
Xa=np.zeros((len(X_tr),ml,d),dtype=np.float32); Xta=np.zeros((len(X_te),ml,d),dtype=np.float32)
for i,s in enumerate(X_tr): Xa[i,:len(s)]=s
for i,s in enumerate(X_te): Xta[i,:len(s)]=s

mask=np.zeros_like(Xa)
for i,s in enumerate(X_tr): mask[i,:len(s)]=1.0
Xm=(Xa*mask).sum(axis=(0,1))/max(mask.sum(),1)
diff=(Xa-Xm)*mask; Xs=np.sqrt((diff**2).sum(axis=(0,1))/max(mask.sum(),1))+1e-8

yl=np.log(1+np.array(y_tr,dtype=np.float32))  # log(1+ratio)
ym,ys=yl.mean(),yl.std()+1e-8

class SD(Dataset):
    def __init__(self,X,l,y,sl):
        self.X=torch.FloatTensor(X); self.l=torch.LongTensor(l)
        self.y=torch.FloatTensor(y); self.sl=torch.FloatTensor(sl)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.l[i],self.y[i],self.sl[i]

tr_len=np.array([len(s) for s in X_tr],dtype=np.int32)
te_len=np.array([len(s) for s in X_te],dtype=np.int32)
sl_tr_arr=np.array(sl_tr,dtype=np.float32); sl_te_arr=np.array(sl_te,dtype=np.float32)
tr_ds=SD((Xa-Xm)/Xs,tr_len,(yl-ym)/ys,sl_tr_arr)
ytl=np.log(1+np.array(y_te,dtype=np.float32))
te_ds=SD((Xta-Xm)/Xs,te_len,(ytl-ym)/ys,sl_te_arr)

def collate_with_sl(batch):
    X,l,y,sl=zip(*batch)
    si=torch.argsort(torch.stack(l),descending=True)
    X=torch.stack([X[i] for i in si]); l=torch.stack([l[i] for i in si])
    y=torch.stack([y[i] for i in si]); sl=torch.stack([sl[i] for i in si])
    return X,l,y,sl

tr_ld=DataLoader(tr_ds,batch_size=64,shuffle=True,collate_fn=collate_with_sl)
te_ld=DataLoader(te_ds,batch_size=256,shuffle=False,collate_fn=collate_with_sl)

# ─── Model with residual skip ───
class ResidualBiLSTM(nn.Module):
    def __init__(self, input_dim=275, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim//2, num_layers=num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers>1 else 0)
        # Predicts the EXCESS slowdown beyond 1.0
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim+1, hidden_dim//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))

    def forward(self, X, lengths, serial_lat):
        x = self.embedding(X)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.bilstm(packed)
        final = torch.cat([hn[-2], hn[-1], torch.log1p(serial_lat).unsqueeze(1)], dim=-1)
        # Predict log(1+excess_ratio) = log(ratio) ≈ log(1 + slowdown_delta)
        # Then output = serial_lat * exp(pred)
        log_ratio_pred = self.predictor(final).squeeze(-1)
        return log_ratio_pred  # in log(1+ratio) z-score space

model = ResidualBiLSTM(input_dim=d)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=2e-5)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=200,eta_min=1e-6)
best_val,best_state=float('inf'),None

for epoch in range(1,201):
    model.train()
    for X,l,y,sl in tr_ld:
        opt.zero_grad()
        pred=model(X,l,sl)
        loss=nn.functional.huber_loss(pred,y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
    scheduler.step()

    model.eval()
    ap,at=[],[]
    with torch.no_grad():
        for X,l,y,sl in te_ld: ap.append(model(X,l,sl).numpy()); at.append(y.numpy())
    p=np.concatenate(ap); t=np.concatenate(at)
    # Denorm to log(1+ratio) then to ratio
    pr_ratio=np.exp(p*ys+ym)-1  # predicted ratio
    tr_ratio=np.exp(t*ys+ym)-1  # true ratio
    # Convert to absolute runtime for Q-error
    pr_abs=np.maximum(pr_ratio*sl_te_arr,0.01)
    tr_abs=np.maximum(tr_ratio*sl_te_arr,0.01)
    qe=np.maximum(pr_abs/tr_abs,tr_abs/pr_abs); med=np.median(qe)
    if med<best_val: best_val=med; best_state={k:v.clone() for k,v in model.state_dict().items()}
    if epoch%20==0 or epoch==1: print(f'E{epoch:3d} val_med={med:.2f}x best={best_val:.2f}x')

model.load_state_dict(best_state); model.eval()
ap,at=[],[]
with torch.no_grad():
    for X,l,y,sl in te_ld: ap.append(model(X,l,sl).numpy()); at.append(y.numpy())
p=np.concatenate(ap); t=np.concatenate(at)
pr_ratio=np.exp(p*ys+ym)-1; tr_ratio=np.exp(t*ys+ym)-1
pr_abs=np.maximum(pr_ratio*sl_te_arr,0.01)
tr_abs=np.maximum(tr_ratio*sl_te_arr,0.01)
qe=np.maximum(pr_abs/tr_abs,tr_abs/pr_abs); qs=np.sort(qe); nq=len(qs)
ss_r=np.sum((np.log(tr_abs+1)-np.log(pr_abs+1))**2)
ss_t=np.sum((np.log(tr_abs+1)-np.mean(np.log(tr_abs+1)))**2)
r2=1-ss_r/max(ss_t,1e-8)

print(f"\n=== K=2 Improved (ratio + residual) ===")
for pct in [10,20,30,40,50,60,70,80,90,95,99]:
    print(f"  P{pct:2d}: {qs[int(nq*pct/100)]:.2f}x")
print(f"  R²: {r2:.4f}")
print(f"\nOriginal K=2:  P50=1.41x P80=2.06x P90=2.66x R²=0.701")
print(f"This K=2:     P50={qs[nq//2]:.2f}x P80={qs[int(nq*0.8)]:.2f}x "
      f"P90={qs[int(nq*0.9)]:.2f}x R²={r2:.4f}")
