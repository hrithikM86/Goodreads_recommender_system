import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from catboost import CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1]/'Assignment 1/assignment1_2026'
ARTIFACTS=Path(__file__).resolve().parent/'artifacts'
ARTIFACTS.mkdir(exist_ok=True)
CACHE=ARTIFACTS/'rich_features.npz'

def feat(t,r,k):
 g=np.diff(t[:k+1])/86400
 logs=np.log1p(g)
 last=list(logs[-10:][::-1]);last += [np.nan]*(10-len(last))
 out=last+[len(g),np.log1p(g.sum()),t[k]/86400, np.mean(r[:k+1]),r[k],np.mean(r[max(0,k-4):k+1]),np.mean(np.array(r[:k+1])==0)]
 for n in [0,3,5,10,20]:
  v=g if n==0 else g[-n:]
  q=np.quantile(v,[.1,.25,.5,.75,.9])
  out.extend(np.log1p(q));out.extend([np.log1p(v.mean()),np.log1p(v.std()),np.mean(v==0)])
 for a in [.1,.3,.7]:
  v=logs[0]
  for x in logs[1:]:v=a*x+(1-a)*v
  out.append(v)
 # Seasonal signals avoid converting anomalous dates through datetime.
 for period in [7,365.25]:
  angle=2*np.pi*(t[k]/86400 % period)/period
  out.extend([np.sin(angle),np.cos(angle)])
 return out

if not CACHE.exists():
 users=defaultdict(lambda:[[],[]])
 with (ROOT/'task_6_train.csv').open() as f:
  rr=csv.reader(f);next(rr)
  for u,i,t,r in rr:users[u][0].append(float(t));users[u][1].append(float(r))
 X=[];y=[];V=[];vy=[];T=[];ty=[]
 for idx,(times,r) in enumerate(users.values(),1):
  t=np.array(times);g=np.diff(t)/86400;n=len(g)
  if n<3:continue
  T.append(feat(t,r,n-1));ty.append(g[-1]);V.append(feat(t,r,n-2));vy.append(g[-2])
  # Same target indices as earlier experiments, so feature comparison is controlled.
  for k in np.unique(np.linspace(1,n-3,min(5,max(0,n-3)),dtype=int)):
   X.append(feat(t,r,k));y.append(g[k])
  if idx%10000==0:print('Features users',idx,flush=True)
 np.savez_compressed(CACHE,X=np.array(X,dtype=np.float32),y=y,V=np.array(V,dtype=np.float32),vy=vy,T=np.array(T,dtype=np.float32),ty=ty)
z=np.load(CACHE);X,y,V,vy,T,ty=[z[k] for k in ['X','y','V','vy','T','ty']]
print('Shapes',X.shape,V.shape,flush=True)
results={};pv={};pt={}
for name,depth,loss,log in [('rich_mae6',6,'MAE',False),('rich_mae8',8,'MAE',False),('rich_log6',6,'MAE',True)]:
 model=CatBoostRegressor(loss_function=loss,iterations=1500,depth=depth,learning_rate=.05,random_seed=42,thread_count=4,verbose=500,allow_writing_files=False)
 model.fit(X,np.log1p(y) if log else y,eval_set=(V,np.log1p(vy) if log else vy),early_stopping_rounds=100)
 a,b=model.predict(V),model.predict(T)
 if log:a,b=np.expm1(a),np.expm1(b)
 pv[name],pt[name]=np.maximum(0,a),np.maximum(0,b)
 results[name]={'selection_MAE':float(np.mean(abs(pv[name]-vy))*86400),'evaluation_MAE':float(np.mean(abs(pt[name]-ty))*86400),'iterations':model.get_best_iteration()+1}
 print(name,results[name],flush=True)
 model.save_model(str(ARTIFACTS/(name+'.cbm')))
 with open(ARTIFACTS/'rich_results.json','w') as f:json.dump(results,f,indent=2)
np.savez_compressed(ARTIFACTS/'rich_predictions.npz',**{'v_'+k:v for k,v in pv.items()},**{'t_'+k:v for k,v in pt.items()})
