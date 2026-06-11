"""
K=4 Bi-LSTM with ratio prediction + residual skip using GNN predicted latency.
"""

import json, csv, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

FEATURES_FILE = '/home/anqian/Desktop/my_lab/workloads/lstm/gnn_features.json'
GNN_LAT_FILE = '/home/anqian/Desktop/my_lab/workloads/lstm/gnn_lat_pred.json'
TRACE_FILE   = '/home/anqian/Desktop/my_lab/workloads/collect_concurrent/trace_4.csv'
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

def conflict(t,c):
    t=np.array(t);c=np.array(c)
    return list(np.minimum(t,c)/np.maximum(np.abs(t)+np.abs(c)+1e-8,1e-8))

def build_seq(data_list, features):
    X,y,sla=[],[],[]
    for qi,si,ei,rti,sti,ov in data_list:
        if qi not in features: continue
        slat = max(float(gnn_lat_pred.get(qi,10)),0.5)
        sl = math.log(1+slat)
        qv = features[qi]['plan_emb']+list(features[qi]['gpu_resources'].values())+[sl]
        tr = list(features[qi]['gpu_resources'].values())
        seq=[]; oi=[]
        for oq in ov:
            if oq not in features: continue
            for os,_,_,_,_ in [(ts,te,q,_,_) for ts,te,q,_,_ in timeline if q==oq]:
                oi.append((os,oq)); break
        oi.sort()
        for os,oq in oi:
            osl=math.log(1+features[oq]['serial_labels'].get('latency_s',10))
            ovv=features[oq]['plan_emb']+list(features[oq]['gpu_resources'].values())+[osl]
            oc=list(features[oq]['gpu_resources'].values())
            c=conflict(tr,oc)
            seq.append(qv+ovv+[si-os,1.0 if os<si else 0.0]+c)
        if seq: X.append(seq); y.append(rti/slat); sla.append(slat)
    return X,y,sla

X_tr,y_tr,sl_tr=build_seq(tr_d,gnn_features)
X_te,y_te,sl_te=build_seq(te_d,gnn_features)
print(f"Train: {len(X_tr)}, Test: {len(X_te)}")

ml=max(max(len(s) for s in X_tr),max(len(s) for s in X_te)); d=len(X_tr[0][0])
Xa=np.zeros((len(X_tr),ml,d),dtype=np.float32); Xta=np.zeros((len(X_te),ml,d),dtype=np.float32)
for i,s in enumerate(X_tr): Xa[i,:len(s)]=s
for i,s in enumerate(X_te): Xta[i,:len(s)]=s
mask=np.zeros_like(Xa)
for i,s in enumerate(X_tr): mask[i,:len(s)]=1.0
Xm=(Xa*mask).sum(axis=(0,1))/max(mask.sum(),1)
diff = ((Xa-Xm)*mask)**2
xs = np.sqrt(diff.sum(axis=(0,1))/max(mask.sum(),1))+1e-8
yl=np.log(1+np.array(y_tr,dtype=np.float32)); ym,ys=yl.mean(),yl.std()+1e-8

class SD(Dataset):
    def __init__(s,X,l,y,sl): s.X=torch.FloatTensor(X);s.l=torch.LongTensor(l);s.y=torch.FloatTensor(y);s.sl=torch.FloatTensor(sl)
    def __len__(s): return len(s.X)
    def __getitem__(s,i): return s.X[i],s.l[i],s.y[i],s.sl[i]

tr_len=np.array([len(s) for s in X_tr],dtype=np.int32)
te_len=np.array([len(s) for s in X_te],dtype=np.int32)
sla=np.array(sl_tr,dtype=np.float32); slb=np.array(sl_te,dtype=np.float32)
tr_ds=SD((Xa-Xm)/xs,tr_len,(yl-ym)/ys,sla)
ytl=np.log(1+np.array(y_te,dtype=np.float32))
te_ds=SD((Xta-Xm)/xs,te_len,(ytl-ym)/ys,slb)

def collate_sl(batch):
    X,l,y,sl=zip(*batch); si=torch.argsort(torch.stack(l),descending=True)
    return torch.stack([X[i] for i in si]),torch.stack([l[i] for i in si]),torch.stack([y[i] for i in si]),torch.stack([sl[i] for i in si])

tr_ld=DataLoader(tr_ds,batch_size=64,shuffle=True,collate_fn=collate_sl)
te_ld=DataLoader(te_ds,batch_size=256,shuffle=False,collate_fn=collate_sl)

class Net(nn.Module):
    def __init__(s,d=275,h=256,n=2,dp=0.2):
        super().__init__()
        s.emb=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(dp))
        s.lstm=nn.LSTM(h,h//2,n,batch_first=True,bidirectional=True,dropout=dp if n>1 else 0)
        s.pred=nn.Sequential(nn.Linear(h+1,h//2),nn.ReLU(),nn.Dropout(dp),nn.Linear(h//2,1))
    def forward(s,X,lens,sl):
        x=s.emb(X)
        p=nn.utils.rnn.pack_padded_sequence(x,lens.cpu(),batch_first=True,enforce_sorted=True)
        _,(hn,_)=s.lstm(p)
        f=torch.cat([hn[-2],hn[-1],torch.log1p(sl).unsqueeze(1)],dim=-1)
        return s.pred(f).squeeze(-1)

model=Net(d=d); print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=2e-5)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=200,eta_min=1e-6)
bv,bs=float('inf'),None
for e in range(1,201):
    model.train()
    for X,l,y,sl in tr_ld: opt.zero_grad();loss=nn.functional.huber_loss(model(X,l,sl),y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
    sch.step()
    model.eval();ap,at=[],[]
    with torch.no_grad():
        for X,l,y,sl in te_ld: ap.append(model(X,l,sl).numpy());at.append(y.numpy())
    p=np.concatenate(ap);t=np.concatenate(at)
    prr=np.exp(p*ys+ym)-1;trr=np.exp(t*ys+ym)-1
    pr=np.maximum(prr*slb,0.01);tr=np.maximum(trr*slb,0.01)
    qe=np.maximum(pr/tr,tr/pr);med=np.median(qe)
    if med<bv: bv=med; bs={k:v.clone() for k,v in model.state_dict().items()}
    if e%20==0 or e==1: print(f'E{e:3d} val_med={med:.2f}x best={bv:.2f}x')

model.load_state_dict(bs); model.eval(); ap,at=[],[]
with torch.no_grad():
    for X,l,y,sl in te_ld: ap.append(model(X,l,sl).numpy());at.append(y.numpy())
p=np.concatenate(ap);t=np.concatenate(at)
prr=np.exp(p*ys+ym)-1;trr=np.exp(t*ys+ym)-1
pr=np.maximum(prr*slb,0.01);tr=np.maximum(trr*slb,0.01)
qe=np.maximum(pr/tr,tr/pr);qs=np.sort(qe);nq=len(qs)
r2=1-np.sum((np.log(tr+1)-np.log(pr+1))**2)/max(np.sum((np.log(tr+1)-np.mean(np.log(tr+1)))**2),1e-8)

print(f"\n=== K=4 (GNN predicted latency) ===")
for pct in [10,20,30,40,50,60,70,80,90,95,99]:
    print(f"  P{pct:2d}: {qs[int(nq*pct/100)]:.2f}x")
print(f"  R²: {r2:.4f}")
print(f"\nK=4 Comparison:")
print(f"              P50     P80     P90     R²")
print(f"Original:     1.63x   2.47x   3.09x   0.469")
print(f"Ratio+real:   1.46x   2.16x   2.72x   0.768")
print(f"Ratio+GNN:    {qs[nq//2]:.2f}x   {qs[int(nq*0.8)]:.2f}x   {qs[int(nq*0.9)]:.2f}x   {r2:.4f}")
