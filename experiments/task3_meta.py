from task3_paths import DATA,ART
import os,json
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from pathlib import Path
z=np.load(str(ART/'split.npz'));order=z['order'];y=z['y'];a=int(.9*len(y));b=int(.95*len(y));vy=y[order[a:b]];ty=y[order[b:]];vp={};tp={}
for prefix,path in [('bias','predictions'),('grid','grid_predictions'),('neighbor','neighbors_predictions'),('kernel','kernel_predictions')]:
 p=np.load(str(ART)+'/'+path+'.npz')
 for k in p.files:
  if k.startswith('v_'):vp[prefix+'_'+k[2:]]=p[k];tp[prefix+'_'+k[2:]]=p['t_'+k[2:]]
names=list(vp);A=np.column_stack([vp[k] for k in names]);B=np.column_stack([tp[k] for k in names]);basecount=A.shape[1]
p=pd.read_csv(DATA/'tasks_3_4_train.csv');p=p[p.rating>0].reset_index(drop=True);tr=p.iloc[order[:a]];va=p.iloc[order[a:b]];te=p.iloc[order[b:]]
def extras(train,query):
 cols=[query.timestamp.to_numpy()/86400/365.25]
 for key in ['user','item']:
  for field in ['rating','timestamp']:
   agg=train.groupby(key)[field].agg(['count','mean','std','min','max'])
   for stat in agg:
    value=query[key].map(agg[stat]).fillna(0).to_numpy()
    if field=='timestamp':value=value/86400/365.25
    cols.append(value)
   if field=='timestamp':
    mean=query[key].map(agg['mean']).fillna(train.timestamp.mean()).to_numpy();cols.append((query.timestamp.to_numpy()-mean)/86400/365.25)
 return np.nan_to_num(np.column_stack(cols),nan=0,posinf=0,neginf=0)
A=np.column_stack([A,extras(tr,va)]);B=np.column_stack([B,extras(tr,te)])
idx=np.random.RandomState(17).permutation(len(vy));cut=int(.8*len(idx));results={}
for depth in [2,4,6]:
 m=CatBoostRegressor(loss_function='RMSE',iterations=700,depth=depth,learning_rate=.03,l2_leaf_reg=10,random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
 m.fit(A[idx[:cut]],vy[idx[:cut]],eval_set=(A[idx[cut:]],vy[idx[cut:]]),early_stopping_rounds=75)
 iterations=m.get_best_iteration()+1
 final=CatBoostRegressor(loss_function='RMSE',iterations=iterations,depth=depth,learning_rate=.03,l2_leaf_reg=10,random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
 final.fit(A,vy);pred=np.clip(final.predict(B),1,5)
 results[str(depth)]={'internal_selection_MSE':float(m.get_best_score()['validation']['RMSE']**2),'evaluation_MSE':float(np.mean((pred-ty)**2)),'iterations':iterations}
 print(depth,results[str(depth)],flush=True)
 final.save_model(str(ART/'meta_')+str(depth)+'.cbm')
with open(str(ART/'meta_results.json'),'w') as f:json.dump({'results':results,'base_columns':names},f,indent=2)
