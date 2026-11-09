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
preds={}
for axis in ['item','user']:
 B=X.T.tocsr() if axis=='item' else X
 E=R.T.tocsr() if axis=='item' else R
 partner=B.T.tocsr();epartner=E.T.tocsr()
 counts=np.diff(B.indptr);norm=np.sqrt(np.asarray(E.power(2).sum(axis=1)).ravel())
 target=qi if axis=='item' else qu
 groups={}
 for q,key in enumerate(target):groups.setdefault(key,[]).append(q)
 keys=np.array(list(groups));names=['jaccard','residual'] if axis=='item' else ['binary','jaccard','residual']
 for method in names:preds[axis+'_'+method]=base.copy()
 other=R if axis=='item' else R.T.tocsr()
 start=time.time()
 for offset in range(0,len(keys),64):
  batch=keys[offset:offset+64];co=(B[batch]@partner).toarray();dot=(E[batch]@epartner).toarray()
  for row,key in enumerate(batch):
   sims={'binary':co[row]/np.sqrt(np.maximum(counts[key]*counts,1)), 'jaccard':co[row]/np.maximum(counts[key]+counts-co[row],1),'residual':np.maximum(0,dot[row]/np.maximum(norm[key]*norm,1e-8))}
   for method in names:sims[method][key]=0
   for q in groups[key]:
    o=qu[q] if axis=='item' else qi[q]
    lo,hi=other.indptr[o:o+2];idx=other.indices[lo:hi];errors=other.data[lo:hi]
    for method in names:
     weights=sims[method][idx]
     if len(weights)>40:
      k=np.argpartition(weights,-40)[-40:];w=weights[k];e=errors[k]
     else:w=weights;e=errors
     preds[axis+'_'+method][q]+=float(w@e)/(w.sum()+1e-8)
  if offset%10000<64:print(axis,offset,'/',len(keys),'seconds',round(time.time()-start),flush=True)
results={}
for key,pred in preds.items():
 pred=np.clip(pred,1,5);preds[key]=pred
 results[key]={'selection_MSE':float(np.mean((pred[:b-a]-y[:b-a])**2)),'evaluation_MSE':float(np.mean((pred[b-a:]-y[b-a:])**2))}
 print(key,results[key],flush=True)
np.savez_compressed(str(ARTIFACTS / 'task1_neighbors2_predictions.npz'),**{'v_'+k:v[:b-a] for k,v in preds.items()},**{'t_'+k:v[b-a:] for k,v in preds.items()})
with open(str(ARTIFACTS / 'task1_neighbors2_results.json'),'w') as f:json.dump(results,f,indent=2)
