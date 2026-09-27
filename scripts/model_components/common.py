"""Data preparation and evaluation utilities for the point-token model."""
from pathlib import Path
import os, sys, json, hashlib, time
import numpy as np
import torch
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from formal_protocol import run_superline_token_baselines_pointnet as R
S, P = R.S, R.P
BASE = ROOT / 'experiments/SuperlineToken_Baselines_Remaining_CenteredXYZ_20260812/pointmlp/best_pointmlp_superline_tokens.pt'
torch.set_num_threads(4)
THRESHOLDS = np.arange(0.05, 0.951, 0.05)

def atomic_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def seed_for(record):
    return 42 + int(hashlib.sha256((record['subset']+'/'+record['sample_id']).encode()).hexdigest()[:7],16)

def metric_record(record, prob, threshold):
    cp=record['projection'] @ prob
    pred=cp>=threshold; gt=record['candidate_gt']
    tp=int(np.sum(pred & gt)); fp=int(np.sum(pred & ~gt))
    fn=record['raw_positive']-tp
    met=P.binary_metrics_from_counts(tp,fp,fn,record['raw_n']-tp-fp-fn)
    return dict(subset=record['subset'],sample_id=record['sample_id'],**met)

def summarize(records, probs, threshold):
    rows=[metric_record(r,p,threshold) for r,p in zip(records,probs)]
    subsets={s:float(np.mean([r['iou'] for r in rows if r['subset']==s])) for s in sorted(set(r['subset'] for r in rows))}
    return {'mean':float(np.mean(list(subsets.values()))),'subsets':subsets,'per_case':rows,'threshold':float(threshold)}

def choose_threshold(records,probs):
    # Exactly the formal uniform-overlap projection and full raw ROI metric.
    cps=[r['projection']@p for r,p in zip(records,probs)]
    scores=[]
    for t in THRESHOLDS:
        ious=[]
        for r,cp in zip(records,cps):
            pred=cp>=t; tp=np.sum(pred&r['candidate_gt']); fp=np.sum(pred&~r['candidate_gt'])
            ious.append(float(tp/max(r['raw_positive']+fp,1)))
        scores.append(float(np.mean(ious)))
    best=int(np.argmax(scores))
    return float(THRESHOLDS[best]),scores

def descriptor(ds):
    names=S.M.DOWNSAMPLED_FEATURE_NAMES
    ids=[names.index(n) for n in S.FINAL_DESCRIPTOR_FEATURE_NAMES]
    d=np.asarray(ds['features'][:,ids],np.float32).copy()
    # Compress scale-sensitive positive quantities; invariant ratios remain explicit.
    for j,n in enumerate(S.FINAL_DESCRIPTOR_FEATURE_NAMES):
        if n not in ('bin_linearness','frag_linearness','token_order_norm','log_frag_num_points'):
            d[:,j]=np.log1p(np.maximum(d[:,j],0))
    return np.nan_to_num(d)

def load_data(smoke=False):
    import pickle
    with open(HERE/'cache'/('smoke_dataset.pkl' if smoke else 'dataset.pkl'),'rb') as f: data=pickle.load(f)
    return data

def ensure_extra_features(data):
    """Existing LCC and spacing-normalized geometry; fit synthetic TRAIN only."""
    for records in data.values():
        for r in records:
            if 'extra' in r:continue
            path=P.TOKEN_CACHE_ROOT/r['subset']/(r['sample_id']+'_downsampled.npz')
            with np.load(path,allow_pickle=False) as z:
                f=z['features'];spacing=np.maximum(f[:,14],1e-6)
                r['soft_y']=z['label_ratio'].astype(np.float32)
                r['extra']=np.stack([z['lcc'],np.log1p(f[:,1]/spacing),np.log1p(f[:,2]/spacing),
                    np.log1p(f[:,9]/spacing),np.log1p(f[:,8]/spacing),np.log1p(f[:,0]*spacing/np.maximum(f[:,1],1e-6))],-1).astype(np.float32)
    train=np.concatenate([r['extra'] for r in data['train']]);mean=train.mean(0);std=np.maximum(train.std(0),.01)
    for records in data.values():
        for r in records:r['extra']=np.clip((r['extra']-mean)/std,-8,8).astype(np.float32)
    return {'mean':mean,'std':std}

