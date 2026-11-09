from pathlib import Path
ARTIFACTS=Path(__file__).resolve().parent/"artifacts"
ARTIFACTS.mkdir(exist_ok=True)
import json,numpy as np
from scipy.optimize import minimize
z=np.load(str(ARTIFACTS / 'task1_split.npz'));vy=z['validation_y'];ty=z['evaluation_y'];vp={};tp={}
for prefix,path in [('svd',str(ARTIFACTS / 'task1_round1_predictions.npz')),('cf',str(ARTIFACTS / 'task1_neighbors_predictions.npz')),('cf2',str(ARTIFACTS / 'task1_neighbors2_predictions.npz')),('tuned',str(ARTIFACTS / 'task1_neighbors_tune_predictions.npz'))]:
 p=np.load(path)
 for k in p.files:
  if k.startswith('v_'):
   name=prefix+'_'+k[2:];vp[name]=np.clip(p[k],1,5);tp[name]=np.clip(p['t_'+k[2:]],1,5)
# Exclude clearly poor configurations using selection MSE alone.
names=sorted(vp,key=lambda n:np.mean((vp[n]-vy)**2))[:12]
names += [n for n in ['svd_bias','svd_svd16_reg.1','svd_svd32_reg.1','svd_svd64_reg.1'] if n not in names]
A=np.column_stack([vp[n] for n in names]);B=np.column_stack([tp[n] for n in names])
obj=lambda w:np.mean((A@w-vy)**2)
res=minimize(obj,np.ones(len(names))/len(names),method='SLSQP',bounds=[(0,1)]*len(names),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':300,'ftol':1e-12})
w=np.maximum(0,res.x);w/=w.sum()
results={'weights':dict(zip(names,w.tolist())),'selection_MSE':float(obj(w)),'evaluation_MSE':float(np.mean((B@w-ty)**2)),'optimizer_success':bool(res.success),'best_single':{'name':names[0],'selection_MSE':float(np.mean((vp[names[0]]-vy)**2)),'evaluation_MSE':float(np.mean((tp[names[0]]-ty)**2))}}
print(json.dumps(results,indent=2))
with open(str(ARTIFACTS / 'task1_blend_results.json'),'w') as f:json.dump(results,f,indent=2)
