"""Low-compute Task 3 LightGBM stacking on cached out-of-sample features.
Run: PYTHONPATH=/workspace/work/task3_lgb:/workspace/work/task1_packages python experiments/task3_lightgbm_refresh.py
Requires existing audited Task 3 component caches. No base model training.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['SURPRISE_DATA_FOLDER']='/workspace/work/surprise_data'
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from task3_round3 import extended_design
from task3_optimize import dataset
W=Path('/workspace/work');DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026'
OLD=W/'task3_round3';OUT=W/'task3_lightgbm_refresh';OUT.mkdir(exist_ok=True)
def mse(y,p):return float(np.mean((y-p)**2))
def clip(p):return np.clip(p,1,5)
def context(train,q):
 qt=q.timestamp.to_numpy(float)/86400;valid=(qt>=0)&(qt<=47482);day=np.where(valid,qt,0)
 cols=[valid.astype(float),np.sin(day*2*np.pi/365.25),np.cos(day*2*np.pi/365.25),np.sin(day*2*np.pi/7),np.cos(day*2*np.pi/7)]
 names=['valid_timestamp','year_sin','year_cos','week_sin','week_cos']
 for key in ['user','item']:
  a=train.groupby(key).rating.agg(['count','std']);count=q[key].map(a['count']).fillna(0).to_numpy()
  cols.extend([np.log1p(count),q[key].map(a['std']).fillna(0).to_numpy()]);names.extend([key+'_log_count',key+'_rating_std'])
  good=train[(train.timestamp>=0)&(train.timestamp<=4102444800)]
  times={k:np.sort(v.timestamp.to_numpy(float)/86400) for k,v in good.groupby(key,sort=False)}
  vals=np.zeros((len(q),5),dtype='f')
  for k,ix in q.groupby(key,sort=False).indices.items():
   ts=times.get(k)
   if ts is None:continue
   t=qt[ix];j=np.searchsorted(ts,t);left=ts[np.maximum(0,j-1)];right=ts[np.minimum(len(ts)-1,j)]
   before=np.where(j>0,t-left,36525);after=np.where(j<len(ts),right-t,36525)
   vals[ix]=np.column_stack([np.log1p(np.clip(before,0,36525)),np.log1p(np.clip(after,0,36525)),np.log1p(np.clip(np.minimum(before,after),0,36525)),j/len(ts),((t<ts[0])|(t>ts[-1])).astype(float)])
  vals[~valid]=0;cols.extend(vals.T);names.extend([key+'_'+x for x in ['prev_gap','next_gap','nearest_gap','time_percentile','outside_history']])
 return np.column_stack(cols).astype('f'),names

def augment(X,names,train,q):
 c,cn=context(train,q)
 indices=[j for j,k in enumerate(names) if (j<48 or 71<=j<81) or (not k.startswith('prior_') and 'support' not in k)]
 P=X[:,indices];d=np.column_stack([P.mean(1),P.std(1),P.max(1)-P.min(1),np.quantile(P,.25,axis=1),np.quantile(P,.75,axis=1)]).astype('f')
 return np.c_[X,c,d].astype('f'),list(names)+cn+['model_mean','model_std','model_range','model_q25','model_q75']

def estimator(count):
 return lgb.LGBMRegressor(objective='regression',n_estimators=count,num_leaves=63,learning_rate=.025,min_child_samples=200,reg_lambda=30,reg_alpha=.1,max_bin=127,colsample_bytree=.9,subsample=.9,subsample_freq=1,n_jobs=2,random_state=42,verbosity=-1)
def main():
 z=np.load(OLD/'extended_design.npz');X=z['X'];y=z['y'];n=int(z['n']);names=list(map(str,z['names']))
 p,tr,q,*_=dataset(DATA,False);assert np.array_equal(y,q.rating.to_numpy()) and (tr.rating>0).all()
 X,names=augment(X,names,tr,q);print('FEATURES',X.shape,flush=True)
 idx=np.random.RandomState(17).permutation(n);a,b=idx[:int(.8*n)],idx[int(.8*n):]
 model=estimator(2500);model.fit(X[a],y[a],eval_set=[(X[b],y[b])],eval_metric='l2',callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(300)])
 count=int(model.best_iteration_);model.booster_.save_model(str(OUT/'selection.txt'))
 pi=clip(model.predict(X[b]));oldinner=.5*(np.load(OLD/'cat_6_res0_inner.npy')+np.load(OLD/'cat_8_res0_inner.npy'))
 assert len(oldinner)==len(b)
 d=pi-oldinner;weight=float(np.clip(np.dot(d,y[b]-oldinner)/max(np.dot(d,d),1e-12),0,1))
 report={'config':estimator(count).get_params(),'features':names,'iterations':count,'new_weight':weight,'inner_accepted_MSE':mse(y[b],oldinner),'inner_lightgbm_MSE':mse(y[b],pi),'inner_selected_blend_MSE':mse(y[b],weight*pi+(1-weight)*oldinner),'protocol':'Positive-only base bank fit90% training ratings. Meta inner split80/20 of next5% ratings, seed17. LightGBM iterations and blend weight selected only here. Final5% evaluates locked settings. Test features reuse full-data base models. Historical cohorts reused across experiments.'}
 (OUT/'locked_selection.json').write_text(json.dumps(report,indent=2))
 print('LOCKED',json.dumps({k:v for k,v in report.items() if k not in ['features','config']}),flush=True)
 model=estimator(count);model.fit(X[:n],y[:n]);model.booster_.save_model(str(OUT/'evaluation.txt'))
 pe=clip(model.predict(X[n:]));ev=np.load(OLD/'evaluation.npz');assert np.array_equal(ev['y'],y[n:]);old=ev['new'];blend=clip(weight*pe+(1-weight)*old)
 report.update(evaluation_accepted_MSE=mse(y[n:],old),evaluation_lightgbm_MSE=mse(y[n:],pe),evaluation_selected_blend_MSE=mse(y[n:],blend))
 np.savez_compressed(OUT/'evaluation.npz',y=y[n:],accepted=old,lightgbm=pe,blend=blend)
 print('EVALUATION',json.dumps({k:v for k,v in report.items() if k.startswith('evaluation')}),flush=True)
 FX,_,fn=extended_design(DATA,W/'task3',W/'task3_optimized',W/'task3_round2',OLD,True)
 _,train,test,*_=dataset(DATA,True);FX,fn=augment(FX,list(map(str,fn)),train,test);assert names==fn
 model=estimator(count);model.fit(X,y);model.booster_.save_model(str(OUT/'production.txt'))
 primary=clip(model.predict(FX));replay=clip(lgb.Booster(model_file=str(OUT/'production.txt')).predict(FX,num_threads=2));assert np.allclose(primary,replay,rtol=0,atol=1e-12)
 accepted=Path('/workspace/output/task3_round3_primary/task_3_predictions.csv');before=hashlib.sha256(accepted.read_bytes()).hexdigest();assert before=='51e9ea196aa12eee41c3bb89b85575413c425c581d0d5c5e575d44ddddc36407'
 old=pd.read_csv(accepted);pd.testing.assert_frame_equal(old.iloc[:,:-1],test.iloc[:,:-1]);mixed=clip(weight*primary+(1-weight)*old.rating.to_numpy())
 report['outputs']={}
 for label,pred in [('primary',primary),('blend',mixed)]:
  dest=Path('/workspace/output')/('task3_lightgbm_'+label);dest.mkdir(exist_ok=True);path=dest/'task_3_predictions.csv';frame=test.copy();frame['rating']=pred;frame.to_csv(path,index=False)
  check=pd.read_csv(path);pd.testing.assert_frame_equal(check.iloc[:,:-1],test.iloc[:,:-1]);assert len(check)==20000 and np.isfinite(check.rating).all() and check.rating.between(1,5).all()
  report['outputs'][label]={'path':str(path),'rows':len(check),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 assert hashlib.sha256(accepted.read_bytes()).hexdigest()==before
 report['input_sha256']={f:hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in ['tasks_3_4_train.csv','task_3_test.csv']}
 (OUT/'results.json').write_text(json.dumps(report,indent=2));print('EXPORTED',json.dumps(report['outputs']),flush=True)
if __name__=='__main__':main()
