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
for q,item in enumerate(qi):groups.setdefault(item,[]).append(q)
keys=np.array(list(groups));preds={str(s):base.copy() for s in [0,10,50]};start=time.time()
for offset in range(0,len(keys),64):
 batch=keys[offset:offset+64];overlap=(G[batch]@X).toarray()
 for row,item in enumerate(batch):
  co=overlap[row]
  cosine=co/np.sqrt(np.maximum(ci[item]*ci,1))
  cosine[item]=0
  for q in groups[item]:
   user=qu[q];lo,hi=R.indptr[user:user+2];history=R.indices[lo:hi];errors=R.data[lo:hi]
   for shrink in [0,10,50]:
    weights=cosine[history]*co[history]/(co[history]+shrink+1e-8)
    # Top 40 among books rated by this user.
    if len(weights)>40:
     k=np.argpartition(weights,-40)[-40:];w=weights[k];e=errors[k]
    else:w=weights;e=errors
    denom=w.sum();preds[str(shrink)][q]+=float(w@e)/(denom+1e-8)
 if offset%6400==0:print('Neighborhood items',offset,'/',len(keys),'seconds',round(time.time()-start),flush=True)
results={}
for key,pred in preds.items():
 pred=np.clip(pred,1,5);v=pred[:b-a];t=pred[b-a:]
 results[key]={'selection_MSE':float(np.mean((v-y[:b-a])**2)),'evaluation_MSE':float(np.mean((t-y[b-a:])**2))}
 print(key,results[key],flush=True)
np.savez_compressed(str(ARTIFACTS / 'task1_neighbors_predictions.npz'),**{'v_'+k:v[:b-a] for k,v in preds.items()},**{'t_'+k:v[b-a:] for k,v in preds.items()})
with open(str(ARTIFACTS / 'task1_neighbors_results.json'),'w') as f:json.dump(results,f,indent=2)
