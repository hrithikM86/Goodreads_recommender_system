"""Task 6: earlier-only quantile features, then median residual stacking.
Run with PYTHONPATH=/workspace/work/task3_lgb python experiments/task6_interval_features.py
Requires existing audited Task 6 feature banks and accepted model artifacts.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from task6_quantile import load, estimator, CONFIGS, fit_weights, mae, W, DATA
OUT=W/'task6_interval_features'
OUT.mkdir(exist_ok=True)
OLD=W/'task6_quantile'
def booster(path): return lgb.Booster(model_file=str(path))
def predict(model,x): return np.maximum(0,model.predict(x,num_threads=2))
def interval_features(a,b,c,anchor):
 lo=np.minimum(np.minimum(a,b),c);hi=np.maximum(np.maximum(a,b),c);width=hi-lo
 return np.column_stack([np.log1p(a),np.log1p(b),np.log1p(c),np.log1p(width),np.log1p(b-lo),np.log1p(hi-b),np.log1p(width/(b+1)),(hi+lo-2*b)/(width+1),np.sign(anchor-b)*np.log1p(abs(anchor-b)),((a>b)|(b>c)).astype(float)]).astype('f')
def quantile(x,y,q,stage):
 path=OUT/f'q{q}_{stage}.txt'
 if path.exists():return booster(path)
 model=estimator(CONFIGS['median_raw'],1171);model.set_params(alpha=q/100)
 model.fit(x,y);model.booster_.save_model(str(path))
 print('FITTED',path.name,flush=True)
 return model

def main():
 X,M,y,ref,n,m,tr,va,_,_=load()
 assert not np.intersect1d(tr,va).size
 lower=predict(quantile(X[:n],y[:n],25,'earlier'),X[n:])
 upper=predict(quantile(X[:n],y[:n],75,'earlier'),X[n:])
 median=predict(booster(OLD/'median_raw_selection.txt'),X[n:])
 A=np.c_[M,interval_features(lower,median,upper,ref)].astype('f')
 assert A.shape==(2*m,143) and np.isfinite(A[:,-10:]).all()
 # Existing feature bank intentionally contains missing values handled by LightGBM.
 oldreport=json.loads((OLD/'results.json').read_text())
 innerold=np.load(OLD/'inner.npz')
 inner={'accepted':sum(w*innerold[k] for k,w in oldreport['weights'].items())}
 counts={'median_correction15':1525,'median_correction31':1263}
 for name,count in counts.items():
  path=OUT/f'{name}_inner.txt'
  if path.exists():model=booster(path)
  else:
   model=estimator(CONFIGS[name],count);model.fit(A[tr],y[n:][tr]-ref[tr]);model.booster_.save_model(str(path))
  inner[name]=np.maximum(0,ref[va]+model.predict(A[va],num_threads=2))
  print('INNER',name,mae(y[n:][va],inner[name]),flush=True)
 names=list(inner);weights=fit_weights(np.column_stack([inner[k] for k in names]),y[n:][va])
 selected=dict(zip(names,map(float,weights)))
 report={'weights':selected,'inner_baseline_MAE':mae(y[n:][va],inner['accepted']),'inner_new_MAE':mae(y[n:][va],sum(selected[k]*inner[k] for k in names)),'public_accepted_MAE':8408503.875509774,'protocol':'Q25/Q75 fixed1171 trees fit only earlier192034 targets. Meta features for penultimate/final targets are out of sample. Correction counts fixed from accepted model. Weights fit only inner20% users; locked before final-gap evaluation. Production correction fits historical out-of-sample quantile features, test uses full-data quantile refits. Previously reused cohorts limit independence.'}
 (OUT/'locked_weights.json').write_text(json.dumps(report,indent=2))
 oldeval=np.load(OLD/'evaluation.npz');evaluation={'accepted':oldeval['primary']}
 assert np.allclose(oldeval['y'],y[n+m:])
 for name,count in counts.items():
  path=OUT/f'{name}_evaluation.txt'
  if path.exists():model=booster(path)
  else:
   model=estimator(CONFIGS[name],count);model.fit(A[:m],y[n:n+m]-ref[:m]);model.booster_.save_model(str(path))
  evaluation[name]=np.maximum(0,ref[m:]+model.predict(A[m:],num_threads=2))
 primary=sum(selected[k]*evaluation[k] for k in names)
 report.update(evaluation_baseline_MAE=mae(y[n+m:],evaluation['accepted']),evaluation_new_MAE=mae(y[n+m:],primary),evaluation_components={k:mae(y[n+m:],v) for k,v in evaluation.items()})
 print('EVALUATION',json.dumps(report),flush=True)
 np.savez_compressed(OUT/'evaluation.npz',y=y[n+m:],primary=primary,**evaluation)
 full=np.load(W/'task6_improved/full.npz');peer=np.load(W/'task6_next/full_peer.npz')
 FX=np.c_[full['X'],peer['X']].astype('f');Q=np.c_[full['Q'],peer['Q']].astype('f')
 anchor=pd.read_csv(W/'task6_improved/blend/task_6_predictions.csv').delta.to_numpy()/86400
 q25=predict(quantile(FX,full['y'],25,'production'),Q)
 q75=predict(quantile(FX,full['y'],75,'production'),Q)
 q50=predict(booster(OLD/'median_raw_production.txt'),Q)
 AQ=np.c_[Q,np.log1p(anchor),interval_features(q25,q50,q75,anchor)].astype('f')
 test={'accepted':pd.read_csv('/workspace/output/task6_quantile_ensemble/task_6_predictions.csv').delta.to_numpy()/86400}
 for name,count in counts.items():
  if selected[name]==0:continue
  path=OUT/f'{name}_production.txt'
  if path.exists():model=booster(path)
  else:
   model=estimator(CONFIGS[name],count);model.fit(A,y[n:]-ref);model.booster_.save_model(str(path))
  test[name]=np.maximum(0,anchor+model.predict(AQ,num_threads=2))
  assert np.allclose(test[name],np.maximum(0,anchor+booster(path).predict(AQ,num_threads=2)))
 pred=sum(selected[k]*v for k,v in test.items())*86400
 frame=pd.read_csv(DATA/'task_6_test.csv');frame['delta']=pred
 assert len(frame)==55403 and np.isfinite(pred).all() and (pred>=0).all()
 dest=Path('/workspace/output/task6_interval_features');dest.mkdir(exist_ok=True)
 path=dest/'task_6_predictions.csv';frame.to_csv(path,index=False)
 reread=pd.read_csv(path);source=pd.read_csv(DATA/'task_6_test.csv')
 pd.testing.assert_frame_equal(reread[['user','item']],source[['user','item']])
 report.update(output=str(path),rows=len(frame),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),features_added=10)
 (OUT/'results.json').write_text(json.dumps(report,indent=2))
 print('EXPORTED',str(path),flush=True)
if __name__=='__main__':main()
