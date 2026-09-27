"""Point-token encoder and residual head components."""
import argparse, copy, math, traceback
from datetime import datetime
from common import *
import torch.nn as nn
import torch.nn.functional as F

def sigmoid(x): return torch.sigmoid(torch.as_tensor(x)).numpy()
def configured_base(cfg): return Path(cfg.get('base_checkpoint',BASE))

class PatchEncoder(nn.Module):
    def __init__(self,invariant=False,multiscale=False):
        super().__init__();self.invariant=invariant;self.multiscale=multiscale
        self.net=nn.Sequential(nn.Linear(3,32),nn.LayerNorm(32),nn.GELU(),nn.Linear(32,48),nn.GELU())
        self.out=nn.Sequential(nn.Linear(288 if multiscale else 96,64),nn.LayerNorm(64),nn.GELU())
    def forward(self,p,mask):
        if self.invariant:
            valid=mask[...,None].to(p.dtype)
            center=(p*valid).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)
            rel=(p-center)*valid
            covariance=rel.transpose(1,2)@rel/valid.sum(1,keepdim=True).clamp_min(1)
            # Eigenvectors only transform observed coordinates; no gradients through
            # degenerate eigensystems are needed because patch inputs are fixed data.
            with torch.no_grad():axis=torch.linalg.eigh(covariance)[1][:,:,-1]
            axial=(rel*axis[:,None]).sum(-1).abs()
            radius=rel.square().sum(-1).sqrt()
            transverse=(radius.square()-axial.square()).clamp_min(0).sqrt()
            p=torch.stack([axial,transverse,radius],-1)
        h=self.net(p/6.0)
        masks=[mask]
        if self.multiscale:
            radius=p.square().sum(-1).sqrt()
            split=(radius*mask).sum(1,keepdim=True)/mask.sum(1,keepdim=True).clamp_min(1)
            masks.extend([mask&(radius<=split),mask&(radius>split)])
        pools=[]
        for m in masks:
            avg=(h*m[...,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
            maximum=h.masked_fill(~m[...,None],-1e4).amax(1)
            maximum=torch.where(m.any(1,keepdim=True),maximum,torch.zeros_like(maximum))
            pools.extend([avg,maximum])
        return self.out(torch.cat(pools,-1))

class ResidualHead(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.dids=list(range(16)) if not cfg.get('invariant') else [4,9,11,12,14,15]
        extra=len(self.dids)+1+(3 if cfg.get('rgb') else 0)
        self.extra_ids=([0] if cfg.get('extra_features')=='lcc' else [1,2,3,4,5] if cfg.get('extra_features')=='geometry' else list(range(6))) if cfg.get('extra_features') else []
        extra+=len(self.extra_ids)
        if cfg.get('patch'): self.patch_encoder=PatchEncoder(cfg.get('rotation_invariant',False),cfg.get('multiscale_patch',False)); extra+=64
        if cfg.get('dual_patch'): self.invariant_patch_encoder=PatchEncoder(True); extra+=64
        if cfg.get('features'):
            self.feat_encoder=nn.Sequential(nn.Linear(192,48),nn.LayerNorm(48),nn.GELU(),nn.Dropout(.15)); extra+=48
        dropout=cfg.get('head_dropout',.1)
        self.embed=nn.Sequential(nn.Linear(extra,96),nn.LayerNorm(96),nn.GELU(),nn.Dropout(dropout))
        if cfg.get('context'):
            self.nbr_embed=nn.Sequential(nn.Linear(len(self.dids)+1+len(self.extra_ids),96),nn.LayerNorm(96),nn.GELU())
            self.q=nn.Linear(96,48);self.k=nn.Linear(96,48);self.v=nn.Linear(96,96)
            self.context=nn.Sequential(nn.Linear(192,96),nn.LayerNorm(96),nn.GELU(),nn.Dropout(dropout))
            if cfg.get('edge_attention'):
                self.edge_net=nn.Sequential(nn.Linear(3,32),nn.GELU(),nn.Linear(32,1))
        self.out=nn.Linear(96,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        if cfg.get('expert'):
            self.gate=nn.Linear(96,1);nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-1.)
    def forward(self,b):
        base=b['logit'].clamp(-10,10)
        evidence=torch.zeros_like(base) if self.cfg.get('independent_geometry') else base
        parts=[b['d'][:,self.dids],evidence[:,None]/5.]
        if self.extra_ids:parts.append(b['extra'][:,self.extra_ids])
        if self.cfg.get('rgb'):parts.append(b['rgb'])
        if self.cfg.get('patch'):parts.append(self.patch_encoder(b['patch'],b['mask']))
        if self.cfg.get('dual_patch'):parts.append(self.invariant_patch_encoder(b['patch'],b['mask']))
        if self.cfg.get('features'):parts.append(self.feat_encoder(b['feat']))
        h=self.embed(torch.cat(parts,-1))
        if self.cfg.get('context'):
            nl=torch.zeros_like(b['nl']) if self.cfg.get('independent_geometry') else b['nl'].clamp(-10,10)
            nparts=[b['nd'][:,:,self.dids],nl[...,None]/5.]
            if self.extra_ids:nparts.append(b['nextra'][:,:,self.extra_ids])
            nh=self.nbr_embed(torch.cat(nparts,-1))
            score=(self.q(h)[:,None]*self.k(nh)).sum(-1)/math.sqrt(48)
            if self.cfg.get('edge_attention'):score=score+self.edge_net(b['edge']).squeeze(-1)
            score=score.masked_fill(~b['nm'],-1e4)
            context=(score.softmax(-1)[...,None]*self.v(nh)).sum(1)
            h=self.context(torch.cat([h,context],-1))
        correction=self.out(h).squeeze(-1)
        self.aux_logit=correction
        if self.cfg.get('geometry_aux'):
            alpha=self.cfg.get('geometry_weight',.5)
            if self.cfg.get('source_preserving_gate'):
                # Let the auxiliary expert act mainly where the frozen source
                # model is uncertain.  This gate uses predictions only.
                confidence=(torch.sigmoid(base)-.5).abs()*2
                amin=self.cfg.get('gate_min',.05);amax=self.cfg.get('gate_max',alpha)
                alpha=amin+(amax-amin)*(1-confidence).pow(self.cfg.get('gate_power',2.))
            if self.cfg.get('uncertainty_gate'):
                alpha=alpha+(1-alpha)*.7*(1-(torch.sigmoid(base)-.5).abs()*2)
            if self.cfg.get('aux_logit_limit'):
                correction=correction.clamp(-self.cfg['aux_logit_limit'],self.cfg['aux_logit_limit'])
            if self.cfg.get('probability_fusion'):
                p=(1-alpha)*torch.sigmoid(base)+alpha*torch.sigmoid(correction)
                return torch.logit(p.clamp(1e-5,1-1e-5))
            limit=self.cfg.get('base_logit_cap',10.)
            return (1-alpha)*base.clamp(-limit,limit)+alpha*correction
        if self.cfg.get('expert'):
            gate=.8*torch.sigmoid(self.gate(h).squeeze(-1))
            return (1-gate)*base+gate*correction
        return base+correction

def neighbors(record,cfg):
    window=cfg.get('order_window',8)
    if window==8:return record['nbr'],record['nbr_mask']
    key='window_'+str(window)
    if key not in record:
        n=len(record['y']);idx=np.tile(np.arange(n,dtype=np.int32)[:,None],(1,2*window+1));mask=np.zeros(idx.shape,bool)
        for fid in np.unique(record['fragment_id']):
            group=np.flatnonzero(record['fragment_id']==fid)
            ordered=group[np.argsort(record['d'][group,12],kind='stable')]
            for i,c in enumerate(ordered):
                ns=ordered[max(0,i-window):i+window+1];idx[c,:len(ns)]=ns;mask[c,:len(ns)]=True
        record[key]=(idx,mask)
    return record[key]

def batch(records,indices,cfg,augment=False):
    batches={k:[] for k in ['d','logit','rgb','patch','mask','feat','nd','nl','nm','y','edge','extra','nextra','soft_y']}
    for r,idx in zip(records,indices):
        center_d=r['d'][idx].copy()
        d_scale=d_shift=None
        if augment and cfg.get('descriptor_style_aug'):
            d_scale=np.random.uniform(cfg.get('descriptor_scale_low',.85),cfg.get('descriptor_scale_high',1.15),
                (1,center_d.shape[-1])).astype(np.float32)
            d_shift=np.random.normal(0,cfg.get('descriptor_shift_std',.08),(1,center_d.shape[-1])).astype(np.float32)
            center_d=center_d*d_scale+d_shift+np.random.normal(0,cfg.get('descriptor_noise_std',.04),center_d.shape).astype(np.float32)
        batches['d'].append(center_d); batches['logit'].append(r['logit'][idx]);batches['y'].append(r['y'][idx])
        extra_scale=extra_shift=extra_keep=None
        if cfg.get('extra_features'):
            center_extra=r['extra'][idx].copy()
            if augment and cfg.get('extra_style_aug'):
                extra_scale=np.random.uniform(cfg.get('extra_scale_low',.75),cfg.get('extra_scale_high',1.25),
                    (1,center_extra.shape[-1])).astype(np.float32)
                extra_shift=np.random.normal(0,cfg.get('extra_shift_std',.20),(1,center_extra.shape[-1])).astype(np.float32)
                extra_keep=(np.random.rand(1,center_extra.shape[-1])>=cfg.get('extra_feature_drop',.1)).astype(np.float32)
                center_extra=(center_extra*extra_scale+extra_shift+
                    np.random.normal(0,cfg.get('extra_noise_std',.06),center_extra.shape))*extra_keep
            batches['extra'].append(center_extra.astype(np.float32))
        if cfg.get('soft_target'):batches['soft_y'].append(r['soft_y'][idx])
        if cfg.get('rgb'):
            rgb=r['x'][idx,3:6].copy()
            if augment and cfg.get('augment'):
                if np.random.rand()<.3:rgb=np.repeat(rgb.mean(-1,keepdims=True),3,-1)
                rgb=np.clip(rgb*np.random.uniform(.8,1.2,(1,3))+np.random.uniform(-.05,.05,(1,3)),0,1)
            batches['rgb'].append(rgb)
        if cfg.get('patch'):
            p=r['patch'][idx].astype(np.float32); mask=r['patch_mask'][idx].copy()
            if augment and cfg.get('augment'):
                p=p*np.random.uniform(.85,1.15);p+=np.random.normal(0,.025,p.shape)
                mask=mask&(np.random.rand(*mask.shape)>.08);mask[:,0]=True
            batches['patch'].append(p);batches['mask'].append(mask)
        if cfg.get('features'):batches['feat'].append(r['feat'][idx])
        if cfg.get('context'):
            ni,nm=neighbors(r,cfg);nb=ni[idx];neighbor_d=r['d'][nb].copy()
            if d_scale is not None:
                neighbor_d=(neighbor_d*d_scale[:,None,:]+d_shift[:,None,:]+
                    np.random.normal(0,cfg.get('descriptor_noise_std',.04),neighbor_d.shape).astype(np.float32))
            batches['nd'].append(neighbor_d);batches['nl'].append(r['logit'][nb]);batches['nm'].append(nm[idx])
            if cfg.get('extra_features'):
                neighbor_extra=r['extra'][nb].copy()
                if extra_scale is not None:
                    neighbor_extra=(neighbor_extra*extra_scale[:,None,:]+extra_shift[:,None,:]+
                        np.random.normal(0,cfg.get('extra_noise_std',.06),neighbor_extra.shape))*extra_keep[:,None,:]
                batches['nextra'].append(neighbor_extra.astype(np.float32))
            if cfg.get('edge_attention'):
                dist=np.linalg.norm(r['x'][nb,:3]-r['x'][idx,None,:3],axis=-1)
                scale=(dist*nm[idx]).sum(-1,keepdims=True)/np.maximum(nm[idx].sum(-1,keepdims=True),1)
                dist=dist/np.maximum(scale,1e-4)
                order=r['d'][nb,12]-r['d'][idx,None,12]
                batches['edge'].append(np.stack([dist,order,np.abs(order)],-1))
    result={k:torch.from_numpy(np.concatenate(v)).to('cuda',dtype=torch.bool if k in ['mask','nm'] else torch.float32) for k,v in batches.items() if v}
    if augment and cfg.get('logit_noise'):
        result['logit']=result['logit']*torch.empty_like(result['logit']).uniform_(.5,1.)+torch.randn_like(result['logit'])*1.5
        if 'nl' in result:result['nl']=result['nl']*.75+torch.randn_like(result['nl'])*1.0
    return result

@torch.no_grad()
def infer_head(model,records,cfg,augment=False,seed=None):
    model.eval(); probs=[]
    np_state=np.random.get_state() if seed is not None else None
    if seed is not None:np.random.seed(seed)
    try:
        for r in records:
            pieces=[]
            for start in range(0,len(r['y']),1024):
                b=batch([r],[np.arange(start,min(start+1024,len(r['y'])))],cfg,augment=augment)
                pieces.append(torch.sigmoid(model(b)).cpu().numpy())
            probs.append(np.concatenate(pieces))
    finally:
        if np_state is not None:np.random.set_state(np_state)
    return probs

