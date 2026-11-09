from pathlib import Path
ARTIFACTS=Path(__file__).resolve().parent/"artifacts"
ARTIFACTS.mkdir(exist_ok=True)
import json,numpy as np
from pathlib import Path
from scipy.optimize import minimize
z=np.load(str(ARTIFACTS/'task1_split.npz'));vy=z['validation_y'];ty=z['evaluation_y'];vp={};tp={}
paths=[('svd','task1_round1'),('svd2','task1_round2'),('cf','task1_neighbors'),('cf2','task1_neighbors2'),('tuned','task1_neighbors_tune'),('ridge','task1_learned_neighbors'),('new','task1_newmodels'),('cat','task1_catboost')]
for prefix,stem in paths:
 path=ARTIFACTS/(stem+'_predictions.npz')
 if not path.exists():continue
 p=np.load(path)
 for key in p.files:
  if key.startswith('v_'):
   name=prefix+'_'+key[2:];vp[name]=np.clip(p[key],1,5);tp[name]=np.clip(p['t_'+key[2:]],1,5)
names=[name for name in vp if np.mean((vp[name]-vy)**2)<.64]
A=np.column_stack([vp[n] for n in names]);B=np.column_stack([tp[n] for n in names])
objective=lambda w:np.mean((A@w-vy)**2)
start=np.zeros(len(names));best=min(names,key=lambda n:np.mean((vp[n]-vy)**2));start[names.index(best)]=1
fit=minimize(objective,start,method='SLSQP',bounds=[(0,1)]*len(names),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':500,'ftol':1e-12})
w=np.maximum(fit.x,0);w/=w.sum()
results={'weights':{name:float(weight) for name,weight in zip(names,w) if weight>1e-7},'selection_MSE':float(objective(w)),'evaluation_MSE':float(np.mean((B@w-ty)**2)),'optimizer_success':bool(fit.success),'models_available':list(vp)}
print(json.dumps(results,indent=2))
with open(str(ARTIFACTS/'task1_expanded_blend_results.json'),'w') as f:json.dump(results,f,indent=2)
