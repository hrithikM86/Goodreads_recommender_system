from task3_paths import DATA,ART
import os,json
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import minimize
z=np.load(str(ART/'split.npz'));order=z['order'];y=z['y'];a=int(.9*len(y));b=int(.95*len(y));vy=y[order[a:b]];ty=y[order[b:]];vp={};tp={}
for prefix,path in [('bias','predictions'),('grid','grid_predictions'),('neighbor','neighbors_predictions'),('kernel','kernel_predictions')]:
 p=np.load(str(ART)+'/'+path+'.npz')
 for k in p.files:
  if k.startswith('v_'):vp[prefix+'_'+k[2:]]=p[k];tp[prefix+'_'+k[2:]]=p['t_'+k[2:]]
names=list(vp);A=np.column_stack([vp[k] for k in names]);B=np.column_stack([tp[k] for k in names]);best=min(names,key=lambda k:np.mean((vp[k]-vy)**2));start=np.zeros(len(names));start[names.index(best)]=1
f=minimize(lambda w:np.mean((A@w-vy)**2),start,method='SLSQP',bounds=[(0,1)]*len(names),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':500,'ftol':1e-12})
w=np.maximum(f.x,0);w/=w.sum();results={'selection_MSE':float(np.mean((A@w-vy)**2)),'evaluation_MSE':float(np.mean((B@w-ty)**2)),'weights':{k:float(v) for k,v in zip(names,w) if v>1e-6},'success':bool(f.success),'best_single':best}
print(json.dumps(results,indent=2))
with open(str(ART/'blend_results.json'),'w') as out:json.dump(results,out,indent=2)
