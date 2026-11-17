"""One XGBoost configuration, cached positive-only Task 3 bank; no base refits.
Run with PYTHONPATH=/workspace/work/task4_xgb:/workspace/work/task3_lgb:/workspace/work/task1_packages python experiments/task3_xgboost_refresh.py
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from task3_lightgbm_refresh import augment,DATA,W,OLD,clip,mse
from task3_round3 import extended_design
from task3_optimize import dataset
OUT=W/'task3_xgboost_refresh';OUT.mkdir(exist_ok=True)
LGB=W/'task3_lightgbm_refresh'
def estimator(count,stopping=False):
 kw=dict(objective='reg:squarederror',eval_metric='rmse',tree_method='hist',n_estimators=count,max_depth=5,learning_rate=.03,min_child_weight=100,reg_lambda=30.,reg_alpha=.1,subsample=.9,colsample_bytree=.9,max_bin=127,n_jobs=2,random_state=42)
 if stopping:kw['early_stopping_rounds']=120
 return xgb.XGBRegressor(**kw)
def main():
 z=np.load(OLD/'extended_design.npz');X,y,n=z['X'],z['y'],int(z['n']);p,tr,q,*_=dataset(DATA,False)
 assert np.array_equal(y,q.rating.to_numpy()) and (tr.rating>0).all()
 oldreport=json.loads((LGB/'results.json').read_text())
 for f,h in oldreport['input_sha256'].items():assert hashlib.sha256((DATA/f).read_bytes()).hexdigest()==h
 X,names=augment(X,list(map(str,z['names'])),tr,q);assert names==oldreport['features']
 idx=np.random.RandomState(17).permutation(n);a,b=idx[:int(.8*n)],idx[int(.8*n):]
 print('FEATURES',X.shape,'XGBoost',xgb.__version__,flush=True)
 model=estimator(2000,True);model.fit(X[a],y[a],eval_set=[(X[b],y[b])],verbose=200)
 count=int(model.best_iteration)+1;model.save_model(OUT/'selection.ubj');pi=clip(model.predict(X[b]))
 ow=oldreport['new_weight'];cat=.5*(np.load(OLD/'cat_6_res0_inner.npy')+np.load(OLD/'cat_8_res0_inner.npy'))
 lp=clip(lgb.Booster(model_file=str(LGB/'selection.txt')).predict(X[b],num_threads=2));accepted_inner=ow*lp+(1-ow)*cat
 d=pi-accepted_inner;weight=float(np.clip(np.dot(d,y[b]-accepted_inner)/max(np.dot(d,d),1e-12),0,1))
 report={'config':estimator(count).get_params(),'iterations':count,'xgboost_weight':weight,'features':names,'inner_accepted_MSE':mse(y[b],accepted_inner),'inner_xgboost_MSE':mse(y[b],pi),'inner_blend_MSE':mse(y[b],weight*pi+(1-weight)*accepted_inner),'public_reference_MSE':.6818167471381362,'protocol':'Same269-column positive-only out-of-sample bank as LightGBM. One XGBoost configuration, early stopping on inner20% of next5% ratings. Blend weight selected on inner validation only, locked before final5% evaluation. Production meta fits all140536 held-out training ratings. No base models retrained. Historical splits reused across experiments.'}
 (OUT/'locked_selection.json').write_text(json.dumps(report,indent=2));print('LOCKED',json.dumps({k:v for k,v in report.items() if k not in ['features','config']}),flush=True)
 model=estimator(count);model.fit(X[:n],y[:n],verbose=False);model.save_model(OUT/'evaluation.ubj');pe=clip(model.predict(X[n:]))
 ev=np.load(LGB/'evaluation.npz');assert np.array_equal(ev['y'],y[n:]);accepted=ev['blend'];blend=clip(weight*pe+(1-weight)*accepted)
 report.update(evaluation_accepted_MSE=mse(y[n:],accepted),evaluation_xgboost_MSE=mse(y[n:],pe),evaluation_blend_MSE=mse(y[n:],blend))
 np.savez_compressed(OUT/'evaluation.npz',y=y[n:],accepted=accepted,xgboost=pe,blend=blend)
 print('EVALUATION',json.dumps({k:v for k,v in report.items() if k.startswith('evaluation')}),flush=True)
 FX,_,fn=extended_design(DATA,W/'task3',W/'task3_optimized',W/'task3_round2',OLD,True)
 _,train,test,*_=dataset(DATA,True);FX,fn=augment(FX,list(map(str,fn)),train,test);assert fn==names
 model=estimator(count);model.fit(X,y,verbose=False);model.save_model(OUT/'production.ubj');primary=clip(model.predict(FX));replay=xgb.XGBRegressor();replay.load_model(OUT/'production.ubj');assert np.array_equal(primary,clip(replay.predict(FX)))
 ap=Path('/workspace/output/task3_lightgbm_blend/task_3_predictions.csv');ah=hashlib.sha256(ap.read_bytes()).hexdigest();assert ah=='7888ef1000535c6b81eed807ccd0b096b71be312d0f7db4620e5be662dac7498';accepted=pd.read_csv(ap);pd.testing.assert_frame_equal(accepted.iloc[:,:-1],test.iloc[:,:-1]);mixed=clip(weight*primary+(1-weight)*accepted.rating.to_numpy())
 report['outputs']={}
 for label,pred in [('primary',primary),('blend',mixed)]:
  dest=Path('/workspace/output')/('task3_xgboost_'+label);dest.mkdir(exist_ok=True);path=dest/'task_3_predictions.csv';frame=test.copy();frame['rating']=pred;frame.to_csv(path,index=False);check=pd.read_csv(path)
  pd.testing.assert_frame_equal(check.iloc[:,:-1],test.iloc[:,:-1]);assert len(check)==20000 and np.isfinite(check.rating).all() and check.rating.between(1,5).all()
  report['outputs'][label]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(check)}
 assert hashlib.sha256(ap.read_bytes()).hexdigest()==ah
 report['input_sha256']=oldreport['input_sha256'];(OUT/'results.json').write_text(json.dumps(report,indent=2));print('EXPORTED',json.dumps(report['outputs']),flush=True)
if __name__=='__main__':main()
