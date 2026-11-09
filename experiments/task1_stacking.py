from pathlib import Path
ARTIFACTS=Path(__file__).resolve().parent/"artifacts"
ARTIFACTS.mkdir(exist_ok=True)
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,numpy as np,pandas as pd
from catboost import CatBoostRegressor
from pathlib import Path
z=np.load(str(ARTIFACTS/'task1_split.npz'));vy=z['validation_y'];ty=z['evaluation_y'];vp={};tp={}
paths=[('svd','task1_round1'),('svd2','task1_round2'),('cf','task1_neighbors'),('cf2','task1_neighbors2'),('tuned','task1_neighbors_tune'),('ridge','task1_learned_neighbors'),('new','task1_newmodels'),('cat','task1_catboost')]
for prefix,stem in paths:
 path=ARTIFACTS/(stem+'_predictions.npz')
 if not path.exists():continue
 p=np.load(path)
 for key in p.files:
  if key.startswith('v_'):
   name=prefix+'_'+key[2:]
   if np.mean((np.clip(p[key],1,5)-vy)**2)<1:
    vp[name]=np.clip(p[key],1,5);tp[name]=np.clip(p['t_'+key[2:]],1,5)
names=list(vp);A=np.column_stack([vp[n] for n in names]);B=np.column_stack([tp[n] for n in names])
p=pd.read_csv(Path(__file__).resolve().parents[1]/'Assignment 1/assignment1_2026/tasks_1_2_train.csv');p=p[p.rating>0].reset_index(drop=True)
order=z['order'];a=int(len(p)*.9);b=int(len(p)*.95);train=p.iloc[order[:a]];val=p.iloc[order[a:b]];test=p.iloc[order[b:]]
for col in ['user','item']:
 stats=train.groupby(col).rating.agg(['count','mean','std'])
 for stat in stats:
  A=np.column_stack([A,val[col].map(stats[stat]).fillna(train.rating.mean()).to_numpy()]);B=np.column_stack([B,test[col].map(stats[stat]).fillna(train.rating.mean()).to_numpy()])
idx=np.random.RandomState(17).permutation(len(vy));cut=int(.8*len(idx));results={};pred={}
for depth in [2,4,6]:
 m=CatBoostRegressor(loss_function='RMSE',iterations=800,depth=depth,learning_rate=.03,random_seed=42,thread_count=2,verbose=False,allow_writing_files=False,l2_leaf_reg=10)
 m.fit(A[idx[:cut]],vy[idx[:cut]],eval_set=(A[idx[cut:]],vy[idx[cut:]]),early_stopping_rounds=75)
 it=m.get_best_iteration()+1
 final=CatBoostRegressor(loss_function='RMSE',iterations=it,depth=depth,learning_rate=.03,random_seed=42,thread_count=2,verbose=False,allow_writing_files=False,l2_leaf_reg=10)
 final.fit(A,vy);pt=np.clip(final.predict(B),1,5)
 results[str(depth)]={'meta_selection_MSE':float(m.get_best_score()['validation']['RMSE']**2),'evaluation_MSE':float(np.mean((pt-ty)**2)),'iterations':it};pred['t_'+str(depth)]=pt
 print(depth,results[str(depth)],flush=True)
 final.save_model(str(ARTIFACTS/'task1_stacker_')+str(depth)+'.cbm')
with open(str(ARTIFACTS/'task1_stacking_results.json'),'w') as f:json.dump({'results':results,'base_columns':names},f,indent=2)
np.savez_compressed(str(ARTIFACTS/'task1_stacking_predictions.npz'),**pred)
