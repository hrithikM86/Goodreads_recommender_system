from pathlib import Path
ARTIFACTS=Path(__file__).resolve().parent/"artifacts"
ARTIFACTS.mkdir(exist_ok=True)
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd,json
from catboost import CatBoostRegressor
root=str(Path(__file__).resolve().parents[1]/'Assignment 1/assignment1_2026')+'/'
p=pd.read_csv(root+'tasks_1_2_train.csv');p=p[p.rating>0].reset_index(drop=True)
order=np.load(str(ARTIFACTS/'task1_split.npz'))['order'];n=len(p);a=int(.9*n);b=int(.95*n)
train=p.iloc[order[:a]].copy();valid=p.iloc[order[a:b]].copy();test=p.iloc[order[b:]].copy()
# Features without any rating-target aggregates. CatBoost computes ordered target encodings.
u=train.groupby('user').timestamp.agg(['count','min','max','mean','std']);i=train.groupby('item').timestamp.agg(['count','min','max','mean','std'])
def make(frame):
 out=frame[['user','item']].copy()
 for name,stats in [('user',u),('item',i)]:
  for col in stats.columns:out[name+'_'+col]=frame[name].map(stats[col]).fillna(0).to_numpy()
 out['mean_time_difference']=out.user_mean-out.item_mean
 out['user_span']=out.user_max-out.user_min;out['item_span']=out.item_max-out.item_min
 return out
X=make(train);V=make(valid);T=make(test);vy=valid.rating.to_numpy();ty=test.rating.to_numpy()
results={};out={}
for depth in [6]:
 name='catboost_depth'+str(depth)
 m=CatBoostRegressor(loss_function='RMSE',iterations=700,depth=depth,learning_rate=.05,random_seed=42,thread_count=3,verbose=200,allow_writing_files=False,cat_features=['user','item'])
 m.fit(X,train.rating,eval_set=(V,vy),early_stopping_rounds=75)
 pv=np.clip(m.predict(V),1,5);pt=np.clip(m.predict(T),1,5)
 results[name]={'selection_MSE':float(np.mean((pv-vy)**2)),'evaluation_MSE':float(np.mean((pt-ty)**2)),'iterations':m.get_best_iteration()+1}
 out['v_'+name]=pv;out['t_'+name]=pt
 print(name,results[name],flush=True)
 m.save_model(str(ARTIFACTS/(name+'.cbm')))
 with open(str(ARTIFACTS/'task1_catboost_results.json'),'w') as f:json.dump(results,f,indent=2)
 np.savez_compressed(str(ARTIFACTS/'task1_catboost_predictions.npz'),**out)
