from task3_paths import DATA,ART
import os,json,pickle,time
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from task3_temporal_model import TemporalBias
z=np.load(str(ART/'split.npz'));u,i,t,y,order=[z[k] for k in ['users','items','time','y','order']];a=int(.9*len(u));b=int(.95*len(u));tr=order[:a];query=order[a:];U=int(u.max()+1)
models={'static':TemporalBias({'user':5,'item':10}).fit(u[tr],i[tr],t[tr],y[tr],U)}
with open(str(ART/'best_bias.pkl'),'rb') as f:models['temporal']=pickle.load(f)
config=[(tau,reg) for tau in [7,30,90,365] for reg in [1,5]]
pred={name+'_'+str(tau)+'_'+str(reg):m.predict(u[query],i[query],t[query]) for name,m in models.items() for tau,reg in config}
trainorder=np.argsort(u[tr]);tu=u[tr][trainorder];starts=np.searchsorted(tu,np.arange(U+1));groups={}
for q,user in enumerate(u[query]):groups.setdefault(user,[]).append(q)
for user,qs in groups.items():
 rows=trainorder[starts[user]:starts[user+1]]
 if not len(rows):continue
 dist=abs(t[query[qs]][:,None]-t[tr[rows]][None,:])/86400
 for tau,reg in config:
  w=np.exp(-dist/tau);den=w.sum(axis=1)+reg
  for name,m in models.items():
   residual=y[tr[rows]]-m.training_prediction[rows]
   pred[name+'_'+str(tau)+'_'+str(reg)][qs]+=w@residual/den
results={};out={}
for name,q in pred.items():
 q=np.clip(q,1,5);v=q[:b-a];e=q[b-a:]
 results[name]={'selection_MSE':float(np.mean((v-y[order[a:b]])**2)),'evaluation_MSE':float(np.mean((e-y[order[b:]])**2))};out['v_'+name]=v;out['t_'+name]=e
print(json.dumps(sorted(results.items(),key=lambda x:x[1]['selection_MSE'])[:8],indent=2),flush=True)
with open(str(ART/'kernel_results.json'),'w') as f:json.dump(results,f,indent=2)
np.savez_compressed(str(ART/'kernel_predictions.npz'),**out)
