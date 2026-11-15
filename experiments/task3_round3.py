"""Personalized user regressions and residual stacking for positive-only Task 3."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse, json, pickle, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz, load_npz
from catboost import CatBoostRegressor
from task3_optimize import dataset, clip, mse
from task3_round2 import current_features

KINDS=['joint','raw','learned','learnedwide','pearson']
def log(*a): print(*a,flush=True)

def user_ridge(data, prior, out, full=False):
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(full=full, **{f:hashlib.sha256((data/f).read_bytes()).hexdigest() for f in ['tasks_3_4_train.csv','task_3_test.csv']})
    guard=out/'manifest.json'
    if guard.exists(): assert json.loads(guard.read_text())==manifest
    guard.write_text(json.dumps(manifest,indent=2))
    if (out/'user_features.npz').exists(): return
    p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full)
    with open(prior/('full' if full else 'validation')/'temporal_bias.pkl','rb') as h: bias=pickle.load(h)
    r=y-bias.training_prediction; bq=bias.predict(qu.astype(np.int64),qi.astype(np.int64),q.timestamp.to_numpy(float))
    shape=(len(users),len(items));X=coo_matrix((np.ones(len(y),np.float32),(u,i)),shape=shape).tocsr(); R=coo_matrix((r,(u,i)),shape=shape).tocsr(); T=coo_matrix((t,(u,i)),shape=shape).tocsr()
    XT=X.T.tocsr(); RT=R.T.tocsr(); TC=T.T.tocsr(); pop=np.asarray(X.sum(1)).ravel(); norm=np.sqrt(np.asarray(R.multiply(R).sum(1)).ravel())
    settings=[('rating',5),('rating',25),('rating',100),('presence',25),('presence',100)]
    result={f'user_ridge_{k}{reg}_{tau}':bq.copy() for k,reg in settings for tau in [0,365]};result['user_ridge_support']=np.zeros(len(q));groups={}
    for z,j in enumerate(qu):
        if j>=0 and qi[z]>=0: groups.setdefault(j,[]).append(z)
    keys=np.array(list(groups));rows=[[] for _ in settings];cols=[[] for _ in settings];vals=[[] for _ in settings];qt=q.timestamp.to_numpy(float);start=time.time()
    for offset in range(0,len(keys),32):
        batch=keys[offset:offset+32];co=(X[batch]@XT).toarray();dot=(R[batch]@RT).toarray()
        for z,j in enumerate(batch):
            co[z,j]=0;dot[z,j]=0;sim=co[z]/np.sqrt(np.maximum(pop[j]*pop,1));corr=np.abs(dot[z])/np.maximum(norm[j]*norm,1e-8)*co[z]/(co[z]+5)
            take=min(32,len(sim));nb1=np.argpartition(sim,-take)[-take:];nb2=np.argpartition(corr,-take)[-take:];nb=np.unique(np.r_[nb1[sim[nb1]>0],nb2[corr[nb2]>0]]);nb=nb[nb!=j]
            if not len(nb):continue
            lo,hi=R.indptr[j:j+2];books=R.indices[lo:hi];yy=R.data[lo:hi];queries=np.array(groups[j]);rr=RT[qi[queries]][:,nb].toarray();xx=XT[qi[queries]][:,nb].toarray();times=TC[qi[queries]][:,nb].toarray();tw=np.exp(-np.abs(times-qt[queries,None])/86400/365)*((times>=0)&(times<=4102444800));result['user_ridge_support'][queries]=np.log1p(xx.sum(1))
            for kind in ['rating','presence']:
                A=(RT if kind=='rating' else XT)[books][:,nb].toarray();dual=A.shape[0]<A.shape[1];gram=A@A.T if dual else A.T@A;eig,V=np.linalg.eigh(gram);rhs=V.T@yy if dual else V.T@(A.T@yy)
                for k,(label,reg) in enumerate(settings):
                    if label!=kind:continue
                    w=((A.T@V if dual else V)@(rhs/(np.maximum(eig,0)+reg))).astype(np.float32);design=rr if kind=='rating' else xx
                    result[f'user_ridge_{kind}{reg}_0'][queries]+=design@w;result[f'user_ridge_{kind}{reg}_365'][queries]+=(design*tw)@w;rows[k].extend(nb);cols[k].extend([j]*len(nb));vals[k].extend(w)
        if offset%2048==0:log('User ridge',offset,len(keys),'seconds',round(time.time()-start))
    for k,(kind,reg) in enumerate(settings):
        W=coo_matrix((np.array(vals[k],np.float32),(rows[k],cols[k])),shape=(len(users),len(users))).tocsr();assert np.count_nonzero(W.diagonal())==0;save_npz(out/f'user_{kind}{reg}.npz',W)
    temp=out/'user_features.partial.npz';np.savez_compressed(temp,**{k:clip(v) if k!='user_ridge_support' else v for k,v in result.items()});temp.replace(out/'user_features.npz')

def design(data,legacy,prior,round2,out,full=False,include_user=True):
    F=current_features(data,legacy,prior,full);values={}
    for kind in KINDS:
        z=np.load(round2/('full' if full else 'validation')/(kind+'_features.npz'));values.update({k:z[k] for k in z.files})
    if include_user:
        z=np.load(out/('full' if full else 'validation')/'user_features.npz');values.update({k:z[k] for k in z.files})
    names=['prior_'+str(k) for k in range(F.shape[1])]+list(values)
    X=np.c_[F,np.column_stack(list(values.values()))].astype(np.float32)
    # Strong, label-free starting point: the learned book-rating neighborhood.
    anchor=np.mean([values['learned_rating25_365'],values['wide_learned_rating25_365']],axis=0)
    return X,anchor,names

def extended_design(data,legacy,prior,round2,out,full=False):
    X,anchor,names=design(data,legacy,prior,round2,out,full)
    z=np.load(out/('full' if full else 'validation')/'ordinal_features.npz')
    return np.c_[X,np.column_stack([z[k] for k in z.files])].astype(np.float32),anchor,names+list(z.files)

def ordinal_ridge(data,prior,round2,out,full=False):
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(full=full, **{f:hashlib.sha256((data/f).read_bytes()).hexdigest() for f in ['tasks_3_4_train.csv','task_3_test.csv']})
    guard=out/'manifest.json'
    if guard.exists():assert json.loads(guard.read_text())==manifest
    guard.write_text(json.dumps(manifest,indent=2))
    if (out/'ordinal_features.npz').exists():return
    p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full)
    with open(prior/('full' if full else 'validation')/'temporal_bias.pkl','rb') as h:bias=pickle.load(h)
    r=y-bias.training_prediction;bq=bias.predict(qu.astype(np.int64),qi.astype(np.int64),q.timestamp.to_numpy(float));shape=(len(users),len(items))
    R=coo_matrix((r,(u,i)),shape=shape).tocsr();Y=coo_matrix((y,(u,i)),shape=shape).tocsr();T=coo_matrix((t,(u,i)),shape=shape).tocsr();C=R.tocsc();W=load_npz(round2/('full' if full else 'validation')/'learned_rating25.npz').tocsc()
    settings=[(kind,reg) for kind in ['joint','ordinal'] for reg in [10,50]];result={f'book_{kind}{reg}_{tau}':bq.copy() for kind,reg in settings for tau in [0,365]};groups={};qt=q.timestamp.to_numpy(float)
    for z,j in enumerate(qi):
        if j>=0 and qu[z]>=0:groups.setdefault(j,[]).append(z)
    rows=[[] for _ in settings];cols=[[] for _ in settings];vals=[[] for _ in settings];start=time.time()
    for z,(j,queries) in enumerate(groups.items()):
        lo,hi=W.indptr[j:j+2];nb=W.indices[lo:hi]
        if not len(nb):continue
        lo,hi=C.indptr[j:j+2];us=C.indices[lo:hi];yy=C.data[lo:hi];queries=np.array(queries);ay=Y[us][:,nb].toarray();ar=R[us][:,nb].toarray();qy=Y[qu[queries]][:,nb].toarray();qr=R[qu[queries]][:,nb].toarray();times=T[qu[queries]][:,nb].toarray();tw=np.exp(-np.abs(times-qt[queries,None])/86400/365)*((times>=0)&(times<=4102444800))
        for kind in ['joint','ordinal']:
            if kind=='joint': A=np.c_[ay>0,ar];B=np.c_[qy>0,qr];BW=B*np.tile(tw,(1,2))
            else:A=np.concatenate([ay==v for v in range(1,6)],axis=1).astype(float);B=np.concatenate([qy==v for v in range(1,6)],axis=1).astype(float);BW=B*np.tile(tw,(1,5))
            dual=A.shape[0]<A.shape[1];gram=A@A.T if dual else A.T@A;eig,V=np.linalg.eigh(gram);rhs=V.T@yy if dual else V.T@(A.T@yy)
            for k,(label,reg) in enumerate(settings):
                if label!=kind:continue
                w=((A.T@V if dual else V)@(rhs/(np.maximum(eig,0)+reg))).astype(np.float32);result[f'book_{kind}{reg}_0'][queries]+=B@w;result[f'book_{kind}{reg}_365'][queries]+=BW@w
                rows[k].extend(np.concatenate([nb+block*len(items) for block in range(2 if kind=='joint' else 5)]));cols[k].extend([j]*len(w));vals[k].extend(w)
        if z%2048==0:log('Ordinal ridge',z,len(groups),'seconds',round(time.time()-start))
    for k,(kind,reg) in enumerate(settings):save_npz(out/f'book_{kind}{reg}.npz',coo_matrix((vals[k],(rows[k],cols[k])),shape=(len(items)*(2 if kind=='joint' else 5),len(items))).tocsr())
    temp=out/'ordinal_features.partial.npz';np.savez_compressed(temp,**{k:clip(v) for k,v in result.items()});temp.replace(out/'ordinal_features.npz')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--legacy',type=Path,required=True);ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--round2',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--worker',action='store_true');ap.add_argument('--ordinal',action='store_true');ap.add_argument('--full',action='store_true');args=ap.parse_args()
    if args.ordinal:return ordinal_ridge(args.data_dir,args.prior,args.round2,args.out/('full' if args.full else 'validation'),args.full)
    if args.worker:return user_ridge(args.data_dir,args.prior,args.out/('full' if args.full else 'validation'),args.full)
    raise SystemExit('Use --worker to build personalized user-neighborhood features.')
if __name__=='__main__':main()
