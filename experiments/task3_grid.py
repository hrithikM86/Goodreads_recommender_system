from task3_paths import DATA,ART
import os,json,pickle,time
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from task3_temporal_model import TemporalBias
z=np.load(str(ART/'split.npz'));u,i,t,y,order=[z[k] for k in ['users','items','time','y','order']];a=int(.9*len(u));b=int(.95*len(u));tr=order[:a];va=order[a:b];te=order[b:];U=int(u.max()+1)
results={};out={}
configs=[]
for regu in [5,15]:
 for drift in [2,10,40]:
  configs.append(('drift_u'+str(regu)+'_d'+str(drift),{'user':regu,'item':10,'month':100,'drift':drift,'userday':20}))
for drift in [2,10,40]:
 configs.append(('asymmetric_'+str(drift),{'user':10,'item':10,'month':100,'drift_pos':drift,'drift_neg':drift,'userday':20}))
for reg in [10,50,100]:
 configs.append(('period_reg'+str(reg),{'user':10,'item':10,'month':100,'drift':20,'user180':reg,'item180':reg,'userday':reg}))
for name,config in configs:
 m=TemporalBias(config).fit(u[tr],i[tr],t[tr],y[tr],U);v=np.clip(m.predict(u[va],i[va],t[va]),1,5);q=np.clip(m.predict(u[te],i[te],t[te]),1,5)
 results[name]={'selection_MSE':float(np.mean((v-y[va])**2)),'evaluation_MSE':float(np.mean((q-y[te])**2)),'config':config};out['v_'+name]=v;out['t_'+name]=q
 print(name,results[name],flush=True)
 with open(str(ART/'grid_results.json'),'w') as f:json.dump(results,f,indent=2)
 np.savez_compressed(str(ART/'grid_predictions.npz'),**out)
