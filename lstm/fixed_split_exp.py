"""
Fixed-split experiment: fixes query split (seed=42), varies only model training seed.
Answers: is GNN genuinely better than ICONQ, or just lucky with data split?
Runs 5 model seeds on the SAME train/test queries.
"""
import os, sys, json, csv, math, re, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader

ROOT = '/home/anqian/Desktop/my_lab/workloads'
OP19 = ['TableFullScan', 'TableRangeScan', 'IndexRangeScan', 'TableRowIDScan',
        'IndexLookUp', 'IndexReader', 'HashJoin', 'MergeJoin', 'IndexJoin', 'IndexHashJoin',
        'HashAgg', 'StreamAgg', 'Sort', 'TopN', 'Window',
        'ExchangeSender', 'ExchangeReceiver', 'Projection', 'Selection']
TABLES = ['lineitem', 'orders', 'partsupp', 'part', 'supplier', 'customer', 'nation', 'region']

SPLIT_SEED = 42
MODEL_SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 250

gnf = {}
for fn in ['lstm/gnn_features_k2_fixed.json','lstm/gnn_features_k3_fixed.json','lstm/gnn_features_k4_fixed.json']:
    with open(os.path.join(ROOT, fn)) as f: gnf.update(json.load(f))

timeline = []
for tf in ['collect_concurrent/trace_2_mixed.csv','collect_concurrent/trace_3_fixed_mixed.csv','collect_concurrent/trace_4_fixed_mixed.csv']:
    with open(os.path.join(ROOT, tf)) as f:
        for row in csv.DictReader(f):
            rt=float(row['runtime']); actual=60.0 if row['status']=='penalty' else rt
            timeline.append((float(row['start']),float(row['start'])+actual,row['qid'],actual,row['status']))
qid_info = {q:(s,e) for s,e,q,_,_ in timeline}

# Fixed query split (seed=SPLIT_SEED)
unique_qids = sorted(set(q for _,_,q,_,_ in timeline if q in gnf))
np.random.seed(SPLIT_SEED); np.random.shuffle(unique_qids)
n_train = int(len(unique_qids)*0.7)
train_qids = set(unique_qids[:n_train]); test_qids = set(unique_qids[n_train:])

ci = []
for i,(si,ei,qi,rti,sti) in enumerate(timeline):
    ov = [qj for j,(sj,ej,qj,_,_) in enumerate(timeline) if i!=j and sj<ei and ej>si]
    ci.append((qi,si,ei,rti,sti,ov))
train_d = [c for c in ci if c[0] in train_qids]; test_d = [c for c in ci if c[0] in test_qids]

# ICONQ features
def build_iconq_features():
    iconq = {}
    for qid in gnf:
        pf = os.path.join(ROOT,'explain_plans',f'{qid}.txt')
        if not os.path.exists(pf): continue
        with open(pf) as f: plan=f.read()
        oc={o:0 for o in OP19}; oe={o:0.0 for o in OP19}; tc={t:0.0 for t in TABLES}
        for line in plan.split('\n'):
            if '\t' not in line or line.startswith('--'): continue
            parts=line.lstrip(' │├└─').split('\t')
            if len(parts)<5: continue
            op=re.sub(r'^[│├└─\s]+','',parts[0].strip())
            op=re.sub(r'\(Build\)|\(Probe\)','',op).strip(); op=re.sub(r'_\d+$','',op)
            try: est=float(parts[1].strip())
            except: est=1.0
            if op in oc: oc[op]+=1; oe[op]+=est
            oi=parts[4].strip() if len(parts)>4 else ''
            for t in TABLES:
                if t in oi.lower(): tc[t]=max(tc[t],est)
        feat=[math.log(1+max(abs(float(gnf[qid]['gpu_resources'].get('lat',1))),0.5))]
        for o in OP19: feat.append(float(oc[o])); feat.append(math.log(1+oe[o]))
        for t in TABLES: feat.append(math.log(1+tc[t]))
        iconq[qid]=feat
    return iconq

iconq_feats = build_iconq_features()

def rc(t,c):
    t=np.array(t);c=np.array(c)
    return list(np.minimum(t,c)/np.maximum(np.abs(t)+np.abs(c)+1e-8,1e-8))

def build_iconq_data(dl):
    X,ya=[],[]
    for qi,si,ei,rti,sti,ov in dl:
        if qi not in iconq_feats or sti=='penalty': continue
        qv=iconq_feats[qi]
        oi=[(qid_info[oq][0],oq) for oq in ov if oq in iconq_feats and oq in qid_info]
        if not oi: continue; oi.sort()
        seq=[qv+iconq_feats[oq]+[si-osv,1.0 if osv<si else 0.0] for osv,oq in oi]
        if seq: X.append(seq); ya.append(rti)
    return X,ya

def build_gnn_data(dl):
    X,yr=[],[]
    for qi,si,ei,rti,sti,ov in dl:
        if qi not in gnf or sti=='penalty': continue
        sl_=max(gnf[qi]['serial_labels'].get('latency_s',1),0.5)
        qv=gnf[qi]['plan_emb']+list(gnf[qi]['gpu_resources'].values())+[math.log(1+sl_)]
        tr_=list(gnf[qi]['gpu_resources'].values()); seq=[]
        for osv,oq in sorted([(qid_info[oq][0],oq) for oq in ov if oq in gnf and oq in qid_info]):
            oslv=math.log(1+gnf[oq]['serial_labels'].get('latency_s',10))
            ovv=gnf[oq]['plan_emb']+list(gnf[oq]['gpu_resources'].values())+[oslv]
            c=rc(tr_,list(gnf[oq]['gpu_resources'].values()))
            seq.append(qv+ovv+[si-osv,1.0 if osv<si else 0.0]+c)
        if seq: X.append(seq); yr.append(rti/sl_)
    return X,yr

# Prepare data once
X_tr_ic,y_tr_ic = build_iconq_data(train_d); X_te_ic,y_te_ic = build_iconq_data(test_d)
ml_ic = max(max(len(s) for s in X_tr_ic),max(len(s) for s in X_te_ic))
Xa_ic=np.zeros((len(X_tr_ic),ml_ic,len(X_tr_ic[0][0])),dtype=np.float32)
for i,s in enumerate(X_tr_ic): Xa_ic[i,:len(s)]=s
Xta_ic=np.zeros((len(X_te_ic),ml_ic,len(X_tr_ic[0][0])),dtype=np.float32)
for i,s in enumerate(X_te_ic): Xta_ic[i,:len(s)]=s
mask_ic=np.zeros_like(Xa_ic)
for i,s in enumerate(X_tr_ic): mask_ic[i,:len(s)]=1.0
Xm_ic=(Xa_ic*mask_ic).sum(axis=(0,1))/max(mask_ic.sum(),1)
Xs_ic=np.sqrt(((Xa_ic-Xm_ic)**2*mask_ic).sum(axis=(0,1))/max(mask_ic.sum(),1))+1e-8
yl_ic=np.log(1+np.array(y_tr_ic,dtype=np.float32))
ym_ic,ys_ic=float(yl_ic.mean()),float(yl_ic.std())+1e-8
Xa_ic_n=(Xa_ic-Xm_ic)/Xs_ic; Xta_ic_n=(Xta_ic-Xm_ic)/Xs_ic
yl_ic_n=(yl_ic-ym_ic)/ys_ic
yl_te_ic=np.log(1+np.array(y_te_ic,dtype=np.float32))
yl_te_ic_n=(yl_te_ic-ym_ic)/ys_ic

X_tr_gnn,y_tr_gnn = build_gnn_data(train_d); X_te_gnn,y_te_gnn = build_gnn_data(test_d)
ml_gnn = max(max(len(s) for s in X_tr_gnn),max(len(s) for s in X_te_gnn))
d_gnn=len(X_tr_gnn[0][0])
Xa_gnn=np.zeros((len(X_tr_gnn),ml_gnn,d_gnn),dtype=np.float32)
for i,s in enumerate(X_tr_gnn): Xa_gnn[i,:len(s)]=s
Xta_gnn=np.zeros((len(X_te_gnn),ml_gnn,d_gnn),dtype=np.float32)
for i,s in enumerate(X_te_gnn): Xta_gnn[i,:len(s)]=s
mask_gnn=np.zeros_like(Xa_gnn)
for i,s in enumerate(X_tr_gnn): mask_gnn[i,:len(s)]=1.0
Xm_gnn=(Xa_gnn*mask_gnn).sum(axis=(0,1))/max(mask_gnn.sum(),1)
Xs_gnn=np.sqrt(((Xa_gnn-Xm_gnn)**2*mask_gnn).sum(axis=(0,1))/max(mask_gnn.sum(),1))+1e-8
yl_gnn=np.log(1+np.array(y_tr_gnn,dtype=np.float32))
ym_gnn,ys_gnn=float(yl_gnn.mean()),float(yl_gnn.std())+1e-8
Xa_gnn_n=(Xa_gnn-Xm_gnn)/Xs_gnn; Xta_gnn_n=(Xta_gnn-Xm_gnn)/Xs_gnn
yl_gnn_n=(yl_gnn-ym_gnn)/ys_gnn

print(f'Fixed split (seed={SPLIT_SEED}): {len(train_qids)} train / {len(test_qids)} test queries')
print(f'ICONQ: {len(X_tr_ic)} train / {len(X_te_ic)} test, dim={len(X_tr_ic[0][0])}')
print(f'GNN:   {len(X_tr_gnn)} train / {len(X_te_gnn)} test, dim={d_gnn}')

# Datasets
class IC_DS:
    def __init__(self,X,l,y): self.X=torch.FloatTensor(X); self.l=torch.LongTensor(l); self.y=torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.l[i],self.y[i]

class GNN_DS:
    def __init__(self,X,l,y): self.X=torch.FloatTensor(X); self.l=torch.LongTensor(l); self.y=torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.l[i],self.y[i]

def cfn(b):
    X,l,y=zip(*b); si=torch.argsort(torch.stack(l),descending=True)
    return torch.stack([X[i] for i in si]),torch.stack([l[i] for i in si]),torch.stack([y[i] for i in si])

train_ic=IC_DS(Xa_ic_n, np.array([len(s) for s in X_tr_ic],dtype=np.int32), yl_ic_n)
test_ic=IC_DS(Xta_ic_n, np.array([len(s) for s in X_te_ic],dtype=np.int32), yl_te_ic_n)
train_gnn=GNN_DS(Xa_gnn_n, np.array([len(s) for s in X_tr_gnn],dtype=np.int32), yl_gnn_n)
test_gnn=GNN_DS(Xta_gnn_n, np.array([len(s) for s in X_te_gnn],dtype=np.int32), yl_gnn_n)

# Models
class ICONQBiLSTM(nn.Module):
    def __init__(self,idim,hs=256,nl=2,do=0.1):
        super().__init__()
        self.bn=nn.BatchNorm1d(idim); self.emb=nn.Sequential(nn.Linear(idim,128),nn.Linear(128,128))
        self.lstm=nn.LSTM(128,hs,nl,dropout=do,batch_first=True,bidirectional=True)
        self.out=nn.Sequential(nn.Linear(hs*2,hs//2),nn.Linear(hs//2,1))
        for m in [self.emb,self.lstm,self.out]:
            for n,p in m.named_parameters():
                if 'weight' in n: nn.init.xavier_uniform_(p.data)
                elif 'bias' in n: nn.init.constant_(p.data,0.0)
    def forward(self,x,le):
        if x.shape[1]>1: x=torch.transpose(x,1,2); x=self.bn(x); x=torch.transpose(x,1,2)
        x=self.emb(x); p=nn.utils.rnn.pack_padded_sequence(x,le.cpu(),batch_first=True,enforce_sorted=False)
        o,_=self.lstm(p); o,_=nn.utils.rnn.pad_packed_sequence(o,batch_first=True)
        return self.out(o[torch.arange(len(le)),le-1]).squeeze(-1)

class GNNResFull(nn.Module):
    def __init__(self,idim=275,hd=256,nl=3,do=0.2):
        super().__init__()
        self.res_gate=nn.Sequential(nn.Linear(10,hd//2),nn.ReLU(),nn.Linear(hd//2,idim),nn.Sigmoid())
        self.emb=nn.Sequential(nn.Linear(idim,hd),nn.ReLU(),nn.Dropout(do))
        self.lstm=nn.LSTM(hd,hd//2,nl,batch_first=True,bidirectional=True,dropout=do if nl>1 else 0)
        self.pred=nn.Sequential(nn.Linear(hd,hd//2),nn.ReLU(),nn.Dropout(do),nn.Linear(hd//2,1))
        self.res_bias=nn.Sequential(nn.Linear(10,hd//4),nn.ReLU(),nn.Linear(hd//4,1))
    def forward(self,X,le):
        rp=torch.cat([X[:,:,128:133],X[:,:,262:267]],dim=-1)
        X=self.emb(X*self.res_gate(rp))
        p=nn.utils.rnn.pack_padded_sequence(X,le.cpu(),batch_first=True,enforce_sorted=True)
        _,(hn,_)=self.lstm(p); f=torch.cat([hn[-2],hn[-1]],dim=-1)
        return self.pred(f).squeeze(-1)+self.res_bias(rp.mean(dim=1)).squeeze(-1)

# Train & eval
device=torch.device('cuda')
print(f'\n{"="*70}')
print(f'FIXED SPLIT (seed={SPLIT_SEED}), 5 MODEL SEEDS')
print(f'{"="*70}')
print(f'{"Seed":>6} {"ICONQ P50":>10} {"ICONQ P90":>10} {"ICONQ P95":>10} {"GNN P50":>10} {"GNN P90":>10} {"GNN P95":>10} {"Δ_P50":>7}')
print('-'*78)

results = []
for mseed in MODEL_SEEDS:
    # ICONQ
    torch.manual_seed(mseed); np.random.seed(mseed)
    tldr=DataLoader(train_ic,batch_size=128,shuffle=True,collate_fn=cfn)
    eldr=DataLoader(test_ic,batch_size=256,shuffle=False,collate_fn=cfn)
    m=ICONQBiLSTM(idim=len(X_tr_ic[0][0])).to(device)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=2e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-6)
    bv,bs=float('inf'),None
    for ep in range(1,EPOCHS+1):
        m.train()
        for X,l,y in tldr: X,y=X.to(device),y.to(device); opt.zero_grad(); loss=nn.functional.huber_loss(m(X,l),y); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),2.0); opt.step()
        sch.step()
        if ep%50==0 or ep==1:
            m.eval();ap,at=[],[]
            with torch.no_grad():
                for X,l,y in eldr: X=X.to(device); ap.append(m(X,l).cpu().numpy()); at.append(y.numpy())
            pz=np.concatenate(ap); tz=np.concatenate(at)
            ps=np.maximum(np.exp(pz*ys_ic+ym_ic)-1,0.01); ts=np.maximum(np.exp(tz*ys_ic+ym_ic)-1,0.01)
            med=np.median(np.maximum(ps/ts,ts/ps))
            if med<bv: bv=med; bs={k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval();ap,at=[],[]
    with torch.no_grad():
        for X,l,y in eldr: X=X.to(device); ap.append(m(X,l).cpu().numpy()); at.append(y.numpy())
    pz=np.concatenate(ap); tz=np.concatenate(at)
    ps=np.maximum(np.exp(pz*ys_ic+ym_ic)-1,0.01); ts=np.maximum(np.exp(tz*ys_ic+ym_ic)-1,0.01)
    qe_ic=np.sort(np.maximum(ps/ts,ts/ps)); nic=len(qe_ic)
    ic_p50=qe_ic[nic//2]; ic_p90=qe_ic[int(nic*0.9)]; ic_p95=qe_ic[int(nic*0.95)]

    # GNN
    torch.manual_seed(mseed); np.random.seed(mseed)
    tldr2=DataLoader(train_gnn,batch_size=128,shuffle=True,collate_fn=cfn)
    eldr2=DataLoader(test_gnn,batch_size=256,shuffle=False,collate_fn=cfn)
    m2=GNNResFull().to(device)
    opt2=torch.optim.AdamW(m2.parameters(),lr=1e-3,weight_decay=2e-5)
    sch2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=EPOCHS,eta_min=1e-6)
    bv2,bs2=float('inf'),None
    for ep in range(1,EPOCHS+1):
        m2.train()
        for X,l,y in tldr2: X,y=X.to(device),y.to(device); opt2.zero_grad(); loss=nn.functional.huber_loss(m2(X,l),y); loss.backward(); torch.nn.utils.clip_grad_norm_(m2.parameters(),2.0); opt2.step()
        sch2.step()
        if ep%50==0 or ep==1:
            m2.eval();ap,at=[],[]
            with torch.no_grad():
                for X,l,y in eldr2: X=X.to(device); ap.append(m2(X,l).cpu().numpy()); at.append(y.numpy())
            pz=np.concatenate(ap); tz=np.concatenate(at)
            pr=np.maximum(np.exp(pz*ys_gnn+ym_gnn)-1,0.01); tr=np.maximum(np.exp(tz*ys_gnn+ym_gnn)-1,0.01)
            med=np.median(np.maximum(pr/tr,tr/pr))
            if med<bv2: bv2=med; bs2={k:v.clone() for k,v in m2.state_dict().items()}
    m2.load_state_dict(bs2); m2.eval();ap,at=[],[]
    with torch.no_grad():
        for X,l,y in eldr2: X=X.to(device); ap.append(m2(X,l).cpu().numpy()); at.append(y.numpy())
    pz=np.concatenate(ap); tz=np.concatenate(at)
    pr=np.maximum(np.exp(pz*ys_gnn+ym_gnn)-1,0.01); tr=np.maximum(np.exp(tz*ys_gnn+ym_gnn)-1,0.01)
    qe_gnn=np.sort(np.maximum(pr/tr,tr/pr)); ngnn=len(qe_gnn)
    gn_p50=qe_gnn[ngnn//2]; gn_p90=qe_gnn[int(ngnn*0.9)]; gn_p95=qe_gnn[int(ngnn*0.95)]

    delta=(ic_p50-gn_p50)/ic_p50*100
    results.append((mseed,ic_p50,ic_p90,ic_p95,gn_p50,gn_p90,gn_p95,delta))
    print(f'{mseed:>6} {ic_p50:>8.2f}x {ic_p90:>8.2f}x {ic_p95:>8.2f}x {gn_p50:>10.2f}x {gn_p90:>10.2f}x {gn_p95:>10.2f}x {delta:>+6.1f}%')

print('\n' + '-'*78)
ic_p50s=[r[1] for r in results]; gn_p50s=[r[4] for r in results]
print(f'{"Mean":>6} {np.mean(ic_p50s):>8.2f}x {np.mean([r[2] for r in results]):>8.2f}x {np.mean([r[3] for r in results]):>8.2f}x {np.mean(gn_p50s):>10.2f}x {np.mean([r[5] for r in results]):>10.2f}x {np.mean([r[6] for r in results]):>10.2f}x {(np.mean(ic_p50s)-np.mean(gn_p50s))/np.mean(ic_p50s)*100:>+6.1f}%')

# Key stat
wins = sum(1 for r in results if r[4] < r[1])
print(f'\nGNN beats ICONQ on P50: {wins}/{len(results)} seeds')
print(f'ICONQ P50 range: {min(ic_p50s):.2f}x - {max(ic_p50s):.2f}x')
print(f'GNN P50 range:   {min(gn_p50s):.2f}x - {max(gn_p50s):.2f}x')
