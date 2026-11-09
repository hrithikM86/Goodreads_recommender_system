from task3_paths import DATA,ART
import os,json,numpy as np
os.environ['OPENBLAS_NUM_THREADS']='1'
from scipy.optimize import minimize
z=np.load(str(ART/'split.npz'));o=z['order'];y=z['y'];a=int(.9*len(y));b=int(.95*len(y));vy=y[o[a:b]];cols=[]
for prefix,path in [('bias','predictions'),('grid','grid_predictions'),('neighbor','neighbors_predictions'),('kernel','kernel_predictions')]:
 p=np.load(str(ART)+'/'+path+'.npz')
 for key in p.files:
  if key.startswith('v_'):cols.append(p[key])
A=np.column_stack(cols);idx=np.random.RandomState(17).permutation(len(vy));cut=int(.8*len(idx));tr=idx[:cut];va=idx[cut:]
start=np.zeros(A.shape[1]);start[np.argmin(np.mean((A[tr]-vy[tr,None])**2,axis=0))]=1
fit=minimize(lambda w:np.mean((A[tr]@w-vy[tr])**2),start,method='SLSQP',bounds=[(0,1)]*A.shape[1],constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':500,'ftol':1e-12})
linear=float(np.mean((A[va]@fit.x-vy[va])**2));meta=json.load(open(str(ART/'meta_results.json')));depth=min(meta['results'],key=lambda d:meta['results'][d]['internal_selection_MSE']);score=meta['results'][depth]['internal_selection_MSE']
result={'linear_internal_selection_MSE':linear,'meta_internal_selection_MSE':score,'meta_depth':int(depth),'chosen':'meta' if score<linear else 'linear','evaluation_MSE':meta['results'][depth]['evaluation_MSE'] if score<linear else json.load(open(str(ART/'blend_results.json')))['evaluation_MSE']}
print(result)
with open(str(ART/'selection.json'),'w') as f:json.dump(result,f,indent=2)
