import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
root=Path(__file__).resolve().parents[1]/'Assignment 1/assignment1_2026'
artifacts=Path(__file__).resolve().parent/'artifacts'
artifacts.mkdir(exist_ok=True)
users=defaultdict(list)
with (root/'task_6_train.csv').open() as f:
 r=csv.reader(f);next(r)
 for u,i,t,rating in r:users[u].append(i)
catX=[];catV=[];catT=[]
for u,items in users.items():
 n=len(items)-1
 if n<3:continue
 catT.append([u,items[n]]);catV.append([u,items[n-1]])
 for k in np.unique(np.linspace(1,n-3,min(5,max(0,n-3)),dtype=int)):catX.append([u,items[k+1]])
z=np.load(artifacts/'rich_features.npz');X,y,V,vy,T,ty=[z[k] for k in ['X','y','V','vy','T','ty']]
def frame(x,c):
 d=pd.DataFrame(x);d.columns=['f'+str(k) for k in range(x.shape[1])]
 d['user']=[v[0] for v in c];d['next_item']=[v[1] for v in c]
 return d
X,V,T=frame(X,catX),frame(V,catV),frame(T,catT)
results={};pv={};pt={}
for name,log in [('ids_mae',False),('ids_log',True)]:
 model=CatBoostRegressor(loss_function='MAE',iterations=1000,depth=6,learning_rate=.05,random_seed=42,thread_count=2,verbose=250,allow_writing_files=False,cat_features=['user','next_item'])
 model.fit(X,np.log1p(y) if log else y,eval_set=(V,np.log1p(vy) if log else vy),early_stopping_rounds=75)
 a,b=model.predict(V),model.predict(T)
 if log:a,b=np.expm1(a),np.expm1(b)
 pv[name],pt[name]=np.maximum(0,a),np.maximum(0,b)
 results[name]={'selection_MAE':float(np.mean(abs(pv[name]-vy))*86400),'evaluation_MAE':float(np.mean(abs(pt[name]-ty))*86400),'iterations':model.get_best_iteration()+1}
 print(name,results[name],flush=True)
 model.save_model(str(artifacts/(name+'.cbm')))
 with open(artifacts/'ids_results.json','w') as f:json.dump(results,f,indent=2)
np.savez_compressed(artifacts/'ids_predictions.npz',**{'v_'+k:v for k,v in pv.items()},**{'t_'+k:v for k,v in pt.items()})
