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
queries=order[a:];qu=u[queries];qi=i[queries];y=ratings[queries];base=mu+bu[qu]+bi[qi];C=R.tocsc()
groups={}
for q,key in enumerate(qi):groups.setdefault(key,[]).append(q)
keys=np.array(list(groups));penalties=np.array([1.,5.,20.,100.]);preds=np.tile(base[:,None],(1,len(penalties)))
start=time.time()
for offset in range(0,len(keys),64):
 batch=keys[offset:offset+64];overlap=(G[batch]@X).toarray()
 for row,key in enumerate(batch):
  sim=overlap[row]/np.sqrt(np.maximum(ci[key]*ci,1));sim[key]=0
  idx=np.argpartition(sim,-100)[-100:];idx=idx[sim[idx]>0]
  if not len(idx):continue
  lo,hi=C.indptr[key:key+2];us=C.indices[lo:hi];target=C.data[lo:hi]
  design=R[us][:,idx].toarray();nrows,ncols=design.shape
  if nrows<ncols:
   eig,Q=np.linalg.eigh(design@design.T)
   W=(design.T@Q)@((Q.T@target)[:,None]/(np.maximum(eig,0)[:,None]+penalties))
  else:
   eig,Q=np.linalg.eigh(design.T@design)
   W=Q@((Q.T@(design.T@target))[:,None]/(np.maximum(eig,0)[:,None]+penalties))
  qs=groups[key];testdesign=R[qu[qs]][:,idx].toarray();preds[qs]+=testdesign@W
 if offset%10000<64:print('Learned neighbors',offset,'/',len(keys),'seconds',round(time.time()-start),flush=True)
results={};out={}
for col,penalty in enumerate(penalties):
 pred=np.clip(preds[:,col],1,5);name=str(penalty)
 results[name]={'selection_MSE':float(np.mean((pred[:b-a]-y[:b-a])**2)),'evaluation_MSE':float(np.mean((pred[b-a:]-y[b-a:])**2))}
 out['v_'+name]=pred[:b-a];out['t_'+name]=pred[b-a:]
print(json.dumps(results,indent=2),flush=True)
np.savez_compressed(str(ARTIFACTS / 'task1_learned_neighbors_predictions.npz'),**out)
with open(str(ARTIFACTS / 'task1_learned_neighbors_results.json'),'w') as f:json.dump(results,f,indent=2)
