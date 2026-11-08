import numpy as np,json
from scipy.optimize import minimize
from pathlib import Path
artifacts=Path(__file__).resolve().parent/"artifacts"
z=np.load(artifacts/'features.npz');vy=z['vy'];ty=z['ty']
vp={};tp={}
old=np.load(artifacts/'predictions.npz')
for name in ['seed_average','log_mae','rmse']:
 vp['original_'+name]=old['validation_'+name];tp['original_'+name]=old['evaluation_'+name]
for path,names in [(artifacts/'rich_predictions.npz',['rich_mae6','rich_mae8','rich_log6']),(artifacts/'ids_predictions.npz',['ids_mae','ids_log'])]:
 p=np.load(path)
 for name in names:vp[name]=p['v_'+name];tp[name]=p['t_'+name]
names=list(vp);A=np.column_stack([vp[n] for n in names]);B=np.column_stack([tp[n] for n in names])
def objective(w):return np.mean(abs(A@w-vy))
start=np.zeros(len(names));start[:3]=[.2,.6,.2]
fit=minimize(objective,start,method='SLSQP',bounds=[(0,1)]*len(names),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':200,'ftol':1e-8})
w=fit.x;w=np.maximum(0,w);w/=w.sum()
results={'weights':dict(zip(names,w.tolist())),'optimizer_success':bool(fit.success),'selection_MAE':float(np.mean(abs(A@w-vy))*86400),'evaluation_MAE':float(np.mean(abs(B@w-ty))*86400),'target_MAE':8457851}
print(json.dumps(results,indent=2))
with open(artifacts/'blend_round2_results.json','w') as f:json.dump(results,f,indent=2)
