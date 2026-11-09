from pathlib import Path
ARTIFACTS=Path(__file__).resolve().parent/"artifacts"
ARTIFACTS.mkdir(exist_ok=True)
import os,time,json
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix
root=str(Path(__file__).resolve().parents[1]/'Assignment 1/assignment1_2026')+'/'
p=pd.read_csv(root+'tasks_1_2_train.csv');p=p[p.rating>0].reset_index(drop=True)
order=np.load(str(ARTIFACTS / 'task1_split.npz'))['order'];n=len(p);a=int(.9*n);b=int(.95*n)
u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);ratings=p.rating.to_numpy();U=len(users);I=len(items)
idx=order[:a];uu=u[idx];ii=i[idx];rr=ratings[idx].astype(float);mu=rr.mean();bu=np.zeros(U);bi=np.zeros(I)
cu=np.bincount(uu,minlength=U);ci=np.bincount(ii,minlength=I)
for _ in range(20):
 bu=np.bincount(uu,weights=rr-mu-bi[ii],minlength=U)/(cu+10)
 bi=np.bincount(ii,weights=rr-mu-bu[uu],minlength=I)/(ci+10)
res=rr-mu-bu[uu]-bi[ii]
X=coo_matrix((np.ones(len(idx),dtype=np.float32),(uu,ii)),shape=(U,I)).tocsr();G=X.T.tocsr();R=coo_matrix((res,(uu,ii)),shape=(U,I)).tocsr()
queries=order[a:];qu=u[queries];qi=i[queries];y=ratings[queries];base=mu+bu[qu]+bi[qi]
groups={}
for q,key in enumerate(qi):groups.setdefault(key,[]).append(q)
keys=np.array(list(groups));ks=[3,5,10,20,40,80];powers=[1,2,3,5]
preds={str(k)+'_'+str(power):base.copy() for k in ks for power in powers}
for offset in range(0,len(keys),64):
 batch=keys[offset:offset+64];overlap=(G[batch]@X).toarray()
 for row,key in enumerate(batch):
  sims=overlap[row]/np.sqrt(np.maximum(ci[key]*ci,1));sims[key]=0
  for q in groups[key]:
   lo,hi=R.indptr[qu[q]:qu[q]+2];history=R.indices[lo:hi];err=R.data[lo:hi]
   w=sims[history];idx=np.argsort(w)[::-1][:80];w=w[idx];err=err[idx]
   for power in powers:
    ww=w**power;num=np.cumsum(ww*err);den=np.cumsum(ww)
    for k in ks:
     j=min(k,len(w))-1
     if j>=0:preds[str(k)+'_'+str(power)][q]+=num[j]/(den[j]+1e-12)
 if offset%10000<64:print('Tuning items',offset,'/',len(keys),flush=True)
results={}
for key,pred in preds.items():
 pred=np.clip(pred,1,5);preds[key]=pred
 results[key]={'selection_MSE':float(np.mean((pred[:b-a]-y[:b-a])**2)),'evaluation_MSE':float(np.mean((pred[b-a:]-y[b-a:])**2))}
print(json.dumps(sorted(results.items(),key=lambda x:x[1]['selection_MSE'])[:8],indent=2),flush=True)
np.savez_compressed(str(ARTIFACTS / 'task1_neighbors_tune_predictions.npz'),**{'v_'+k:v[:b-a] for k,v in preds.items()},**{'t_'+k:v[b-a:] for k,v in preds.items()})
with open(str(ARTIFACTS / 'task1_neighbors_tune_results.json'),'w') as f:json.dump(results,f,indent=2)
