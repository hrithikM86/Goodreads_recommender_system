from task3_paths import DATA,ART
import os,json,pickle,time
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.sparse import coo_matrix
from task3_temporal_model import TemporalBias
z=np.load(str(ART/'split.npz'));u,i,t,y,order=[z[k] for k in ['users','items','time','y','order']];U=int(u.max()+1);I=int(i.max()+1);a=int(.9*len(u));b=int(.95*len(u));tr=order[:a];query=order[a:]
static=TemporalBias({'user':5,'item':10}).fit(u[tr],i[tr],t[tr],y[tr],U)
with open(str(ART/'best_bias.pkl'),'rb') as f:temporal=pickle.load(f)
bases={'static':static.predict(u[query],i[query],t[query]),'temporal':temporal.predict(u[query],i[query],t[query])}
R={name:coo_matrix((y[tr]-m.training_prediction,(u[tr],i[tr])),shape=(U,I)).tocsr() for name,m in [('static',static),('temporal',temporal)]}
X=coo_matrix((np.ones(len(tr),dtype=np.float32),(u[tr],i[tr])),shape=(U,I)).tocsr();G=X.T.tocsr();pop=np.asarray(X.sum(axis=0)).ravel()
Times=coo_matrix((t[tr],(u[tr],i[tr])),shape=(U,I)).tocsr()
assert all(np.array_equal(m.indices,Times.indices) and np.array_equal(m.indptr,Times.indptr) for m in R.values())
configs=[(20,1,0),(20,2,0),(20,1,365),(20,2,365),(20,1,90),(20,2,90)]
preds={name+'_'+str(k)+'_'+str(power)+'_'+str(tau):base.copy() for name,base in bases.items() for k,power,tau in configs}
groups={}
for q,key in enumerate(i[query]):groups.setdefault(key,[]).append(q)
keys=np.array(list(groups));start=time.time()
for offset in range(0,len(keys),64):
 batch=keys[offset:offset+64];co=(G[batch]@X).toarray()
 for row,key in enumerate(batch):
  cosine=co[row]/np.sqrt(np.maximum(pop[key]*pop,1));cosine[key]=0
  for q in groups[key]:
   user=u[query[q]];lo,hi=X.indptr[user:user+2];hist=X.indices[lo:hi];sim=cosine[hist];distance=abs(Times.data[lo:hi]-t[query[q]])/86400
   for k,power,tau in configs:
    w=sim**power
    if tau:w=w*np.exp(-distance/tau)
    if len(w)>k:idx=np.argpartition(w,-k)[-k:]
    else:idx=np.arange(len(w))
    weights=w[idx];den=weights.sum()+1e-12
    for name in R:preds[name+'_'+str(k)+'_'+str(power)+'_'+str(tau)][q]+=(weights@R[name].data[lo:hi][idx])/den
 if offset%5000<64:print('Items',offset,'/',len(keys),'seconds',round(time.time()-start),flush=True)
results={};out={}
for name,p in preds.items():
 p=np.clip(p,1,5);v=p[:b-a];q=p[b-a:]
 results[name]={'selection_MSE':float(np.mean((v-y[order[a:b]])**2)),'evaluation_MSE':float(np.mean((q-y[order[b:]])**2))};out['v_'+name]=v;out['t_'+name]=q
print(json.dumps(sorted(results.items(),key=lambda x:x[1]['selection_MSE']),indent=2),flush=True)
with open(str(ART/'neighbors_results.json'),'w') as f:json.dump(results,f,indent=2)
np.savez_compressed(str(ART/'neighbors_predictions.npz'),**out)
