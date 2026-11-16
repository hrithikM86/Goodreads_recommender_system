"""Task6 median LightGBM experts and selection-only convex ensemble.
No hidden labels; preserve the public accepted8,452,714.299367478 CSV.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import argparse,json,hashlib,gc,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
from scipy.optimize import minimize
from catboost import CatBoostRegressor
import lightgbm as lgb
from task6_next_models import reference
W=Path('/workspace/work');DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026';OUT=W/'task6_quantile';OUT.mkdir(exist_ok=True)
CONFIGS={
 'median_raw':dict(kind='raw',num_leaves=31,min_child_samples=200,reg_lambda=50.,learning_rate=.035),
 'median_log':dict(kind='log',num_leaves=63,min_child_samples=200,reg_lambda=50.,learning_rate=.035),
 'median_correction15':dict(kind='residual',num_leaves=15,min_child_samples=150,reg_lambda=30.,learning_rate=.025),
 'median_correction31':dict(kind='residual',num_leaves=31,min_child_samples=200,reg_lambda=50.,learning_rate=.025),
}
def mae(y,p):return float(np.mean(abs(y-p))*86400)
def estimator(c,count=2000):
 return lgb.LGBMRegressor(objective='quantile',alpha=.5,n_estimators=count,max_bin=127,
   colsample_bytree=.9,subsample=.9,subsample_freq=1,reg_alpha=.1,
   path_smooth=10,random_state=3139,n_jobs=2,verbosity=-1,
   **{k:v for k,v in c.items() if k!='kind'})
def transform(pred,kind,anchor=None):
 if kind=='log':pred=np.expm1(np.minimum(pred,15))
 if kind=='residual':pred=pred+anchor
 return np.maximum(0,pred)
def load():
 z=np.load(W/'task6_improved/validation.npz');X=np.c_[z['X'],np.load(W/'task6_next/validation_peer.npz')['X']].astype('f');y=z['y'];n,m=int(z['n']),int(z['m']);p=pd.read_csv(DATA/'task_6_train.csv');rows=z['rows'];assert np.unique(p.user.iloc[rows[n:n+m]]).size==m and np.array_equal(p.user.iloc[rows[n:n+m]].to_numpy(),p.user.iloc[rows[n+m:]].to_numpy())
 rv,rt=reference(SimpleNamespace(legacy=W,previous=W/'task6_improved'),p,z);ref=np.r_[rv,rt];M=np.c_[X[n:],np.log1p(ref)].astype('f');idx=np.random.RandomState(91).permutation(m);tr,va=idx[:int(.8*m)],idx[int(.8*m):]
 baseline=np.load(W/'task6_next/stack_evaluation.npz');assert np.allclose(baseline['y'],y[n+m:]);assert abs(mae(y[n+m:],baseline['primary'])-7155338.850502782)<1e-4
 old=CatBoostRegressor();old.load_model(str(W/'task6_next/cal5_inner.cbm'));inner=np.maximum(0,rv[va]+old.predict(M[va]));return X,M,y,ref,n,m,tr,va,inner,baseline['primary']
def fit_weights(P,y):
 best=None
 for init in [np.ones(P.shape[1])/P.shape[1]]+list(np.eye(P.shape[1])):
  r=minimize(lambda w:float(np.mean(abs(P@w-y))),init,method='SLSQP',bounds=[(0,1)]*P.shape[1],constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'ftol':1e-9,'maxiter':300})
  if np.isfinite(r.fun) and (best is None or r.fun<best.fun):best=r
 w=np.maximum(best.x,0);w[w<1e-6]=0;w/=w.sum();return w

def experiment():
 X,M,y,ref,n,m,tr,va,accepted_inner,accepted_eval=load();inner={'accepted':accepted_inner};evaluation={'accepted':accepted_eval};runs={};started=time.time()
 for name,c in CONFIGS.items():
  kind=c['kind'];path=OUT/(name+'_selection.txt');info=OUT/(name+'_info.json')
  if path.exists():r=json.loads(info.read_text());model=lgb.Booster(model_file=str(path),params={'num_threads':2})
  else:
   model=estimator(c)
   if kind=='residual':A,target,V,vy=M[tr],y[n:][tr]-ref[tr],M[va],y[n:][va]-ref[va]
   else:
    A,target,V,vy=X[:n],y[:n],X[n:][tr],y[n:][tr]
    if kind=='log':target=np.log1p(target);vy=np.log1p(vy)
   model.fit(A,target,eval_set=[(V,vy)],eval_metric='quantile',callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(300)]);count=int(model.best_iteration_);model.booster_.save_model(str(path));r={'config':c,'iterations':count};info.write_text(json.dumps(r,indent=2))
  frame=M[va] if kind=='residual' else X[n:][va];inner[name]=transform(model.predict(frame),kind,ref[va] if kind=='residual' else None);r['inner_MAE_seconds']=mae(y[n:][va],inner[name]);runs[name]=r
  print('INNER',name,json.dumps(r),'elapsed',round(time.time()-started),flush=True)
 names=list(inner);weights=fit_weights(np.column_stack([inner[k] for k in names]),y[n:][va]);selected={k:float(v) for k,v in zip(names,weights)};inner_score=mae(y[n:][va],sum(selected[k]*inner[k] for k in names));print('INNER_SELECTED',selected,inner_score,'accepted',mae(y[n:][va],accepted_inner),flush=True)
 # Once weights/configurations are locked, inspect the separate final-gap targets.
 for name,c in CONFIGS.items():
  kind=c['kind'];path=OUT/(name+'_evaluation.txt')
  if kind=='residual':
   if path.exists():model=lgb.Booster(model_file=str(path),params={'num_threads':2})
   else:model=estimator(c,runs[name]['iterations']);model.fit(M[:m],y[n:n+m]-ref[:m]);model.booster_.save_model(str(path))
  else:model=lgb.Booster(model_file=str(OUT/(name+'_selection.txt')),params={'num_threads':2})
  frame=M[m:] if kind=='residual' else X[n+m:];evaluation[name]=transform(model.predict(frame),kind,ref[m:] if kind=='residual' else None);runs[name]['evaluation_MAE_seconds']=mae(y[n+m:],evaluation[name]);print('EVALUATION',name,runs[name]['evaluation_MAE_seconds'],flush=True)
 primary=sum(selected[k]*evaluation[k] for k in names);target=y[n+m:];delta=abs(primary-target)-abs(accepted_eval-target);se=delta.std(ddof=1)/np.sqrt(m);r={'weights':selected,'runs':runs,'inner_accepted_MAE_seconds':mae(y[n:][va],accepted_inner),'inner_new_MAE_seconds':inner_score,'evaluation_accepted_MAE_seconds':mae(target,accepted_eval),'evaluation_new_MAE_seconds':mae(target,primary),'paired_MAE_change_seconds':float(delta.mean()*86400),'approx_user_IID95pct_change_seconds':[float((delta.mean()-1.96*se)*86400),float((delta.mean()+1.96*se)*86400)],'selection_users':m,'inner_selection_train_users':len(tr),'inner_selection_holdout_users':len(va),'evaluation_users':m,'public_reference_MAE_seconds':8452714.299367478,'protocol':'Direct experts fit earlier targets only; counts stop using80% penultimate targets. Correction experts fit80% penultimate targets and stop on remaining20%. Convex weights use inner20% only and are fixed before final-gap evaluation. Production refits all available known targets. Historical cohorts reused; local gains do not establish public/hidden rank.'};np.savez_compressed(OUT/'evaluation.npz',y=target,primary=primary,**evaluation);np.savez_compressed(OUT/'inner.npz',y=y[n:][va],**inner);(OUT/'results.json').write_text(json.dumps(r,indent=2));print('RESULT',json.dumps({k:v for k,v in r.items() if k!='runs'}),flush=True)
 return r

def produce():
 r=json.loads((OUT/'results.json').read_text());X,M,y,ref,n,m,tr,va,ai,ae=load();full=np.load(W/'task6_improved/full.npz');peer=np.load(W/'task6_next/full_peer.npz');FX=np.c_[full['X'],peer['X']].astype('f');Q=np.c_[full['Q'],peer['Q']].astype('f');anchor=pd.read_csv(W/'task6_improved/blend/task_6_predictions.csv').delta.to_numpy()/86400;MQ=np.c_[Q,np.log1p(anchor)].astype('f');old=pd.read_csv('/workspace/output/task6_round2_primary/task_6_predictions.csv');pred={'accepted':old.delta.to_numpy()/86400};oldmodel=CatBoostRegressor();oldmodel.load_model(str(W/'task6_next/cal5_full.cbm'));assert np.allclose(pred['accepted'],np.maximum(0,anchor+oldmodel.predict(MQ)),atol=1e-8)
 for name,c in CONFIGS.items():
  if r['weights'].get(name,0)==0:continue
  kind=c['kind'];path=OUT/(name+'_production.txt')
  if path.exists():model=lgb.Booster(model_file=str(path),params={'num_threads':2})
  else:
   model=estimator(c,r['runs'][name]['iterations']);F,target=(M,y[n:]-ref) if kind=='residual' else (FX,np.log1p(full['y']) if kind=='log' else full['y']);model.fit(F,target);model.booster_.save_model(str(path))
  pred[name]=transform(model.predict(MQ if kind=='residual' else Q),kind,anchor if kind=='residual' else None);print('PRODUCTION_READY',name,flush=True)
 value=sum(r['weights'][k]*v for k,v in pred.items())*86400;assert np.isfinite(value).all() and (value>=0).all();q=pd.read_csv(DATA/'task_6_test.csv');frame=q.copy();frame['delta']=value;dest=Path('/workspace/output/task6_quantile_ensemble');dest.mkdir(exist_ok=True);path=dest/'task_6_predictions.csv';frame.to_csv(path,index=False);read=pd.read_csv(path);pd.testing.assert_frame_equal(read.iloc[:,:-1],q.iloc[:,:-1]);assert len(read)==55403 and (read.delta>=0).all() and np.isfinite(read.delta).all();np.savez_compressed(OUT/'test.npz',primary=value,**pred);r.update(output=str(path),rows=len(read),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mean_absolute_change_vs_accepted_seconds=float(np.mean(abs(value-old.delta.to_numpy()))),input_sha256={f:hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in ['task_6_train.csv','task_6_test.csv']});(OUT/'results.json').write_text(json.dumps(r,indent=2));print('EXPORTED',r['output'],r['sha256'],flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['experiment','produce'],required=True);a=ap.parse_args();experiment() if a.stage=='experiment' else produce()
