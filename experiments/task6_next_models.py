"""Peer timing regressors and monotone-target MAE experts, selection-only blending."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize
from catboost import CatBoostRegressor
import lightgbm as lgb
from task6_improve_models import framing,user_weights,old_prediction

def reg(name,iters=1800):
 if name=='balanced':
  from task6_improve_models import model
  return model('balanced',iters)
 if name=='catlog':return CatBoostRegressor(iterations=min(iters,1200),depth=7,learning_rate=.04,l2_leaf_reg=10,loss_function='MAE',random_seed=83,thread_count=1,verbose=300,allow_writing_files=False,cat_features=['user','next_item'],max_ctr_complexity=1,ctr_leaf_count_limit=50000,model_size_reg=2)
 return lgb.LGBMRegressor(objective='regression_l1',num_leaves=63 if name=='log' else 31,n_estimators=iters,learning_rate=.035,min_child_samples=100,reg_lambda=25,max_bin=127,colsample_bytree=.9,subsample=.9,subsample_freq=1,n_jobs=1,random_state=81,verbosity=-1)
def anchor(X):
 a=np.expm1(np.minimum(np.asarray(X)[:,86],15));fallback=np.expm1(np.asarray(X)[:,19]);return np.where(np.isfinite(a),a,fallback)
def forecast(M,X,logged,base=None):
 p=np.expm1(np.minimum(M.predict(X),15)) if logged else M.predict(X)
 return np.maximum(0,p if base is None else p+base)
def reference(a,p,z):
 rv,rt=old_prediction(a.legacy);n=int(z['n']);m=int(z['m']);out=[];cfg=json.loads((a.previous/'results.json').read_text())['blend_weights']
 for label,XX,rr,base in [('v',z['X'][n:n+m],z['rows'][n:n+m],rv),('t',z['X'][n+m:],z['rows'][n+m:],rt)]:
  pred=cfg['accepted']*base
  for name in ['balanced','cat']:
   if name=='cat':M=CatBoostRegressor();M.load_model(str(a.previous/(name+'_validation.model')));Q=framing(XX,p,rr)
   else:M=lgb.Booster(model_file=str(a.previous/(name+'_validation.model')),params={'num_threads':1});Q=XX
   pred+=cfg[name]*np.maximum(0,M.predict(Q))
  out.append(pred)
 return out

def main():
 ap=argparse.ArgumentParser()
 for key in ['data-dir','legacy','previous','out','accepted']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--predict',action='store_true');ap.add_argument('--only-model',choices=['raw','log','catlog','offset','balanced']);a=ap.parse_args();z=np.load(a.previous/'validation.npz');peer=np.load(a.out/'validation_peer.npz');X=np.c_[z['X'],peer['X']];y=z['y'];rows=z['rows'];n=int(z['n']);m=int(z['m']);p=pd.read_csv(a.data_dir/'task_6_train.csv');rv,rt=reference(a,p,z);mae=lambda yy,pp:float(np.mean(abs(yy-pp))*86400);predv={'accepted':rv};predt={'accepted':rt};runs={};weight=user_weights(p,rows[:n])
 for name in ([a.only_model] if a.only_model else ['raw','log','catlog','offset','balanced']):
  logged=name in ['log','catlog'];offset=name=='offset';path=a.out/(name+'_validation.model');info=a.out/(name+'_info.json');M=reg(name);TX=X[:n];V=X[n:n+m];T=X[n+m:]
  if name=='catlog':TX=framing(TX,p,rows[:n]);V=framing(V,p,rows[n:n+m]);T=framing(T,p,rows[n+m:])
  if path.exists():
   cfg=json.loads(info.read_text())
   if name=='catlog':M.load_model(str(path))
   else:M=lgb.Booster(model_file=str(path),params={'num_threads':1})
  else:
   target=np.log1p(y[:n]) if logged else (y[:n]-anchor(X[:n]) if offset else y[:n]);vy=np.log1p(y[n:n+m]) if logged else (y[n:n+m]-anchor(X[n:n+m]) if offset else y[n:n+m])
   if name=='catlog':M.fit(TX,target,eval_set=(V,vy),early_stopping_rounds=125);iters=M.tree_count_;M.save_model(str(path))
   else:M.fit(TX,target,sample_weight=weight if logged or name=='balanced' else None,eval_set=[(V,vy)],callbacks=[lgb.early_stopping(125,verbose=False),lgb.log_evaluation(300)]);iters=M.best_iteration_;M.booster_.save_model(str(path))
   cfg={'iterations':int(iters),'logged_target':logged,'offset_target':offset,'balanced_weighted':name=='balanced'};info.write_text(json.dumps(cfg))
  predv[name]=forecast(M,V,logged,anchor(X[n:n+m]) if offset else None);predt[name]=forecast(M,T,logged,anchor(X[n+m:]) if offset else None);cfg['selection_MAE']=mae(y[n:n+m],predv[name]);cfg['evaluation_MAE']=mae(y[n+m:],predt[name]);runs[name]=cfg;print(name,cfg,flush=True)
 if a.only_model:return
 names=list(predv);P=np.column_stack(list(predv.values()));best=None
 for init in [np.ones(len(names))/len(names)]+list(np.eye(len(names))):
  r=minimize(lambda w:np.mean(abs(P@w-y[n:n+m])),init,method='SLSQP',bounds=[(0,1)]*len(names),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':200,'ftol':1e-8})
  if best is None or r.fun<best.fun:best=r
 w=np.maximum(0,best.x);w[w<1e-6]=0;w/=w.sum();weights=dict(zip(names,w.tolist()));v=sum(weights[k]*predv[k] for k in names);t=sum(weights[k]*predt[k] for k in names);summary={'weights':weights,'runs':runs,'reference_evaluation_MAE':mae(y[n+m:],rt),'new_evaluation_MAE':mae(y[n+m:],t),'reference_selection_MAE':mae(y[n:n+m],rv),'new_selection_MAE':mae(y[n:n+m],v),'reference_public_MAE':8890395.0326342};(a.out/'results.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(a.out/'evaluation.npz',y=y[n+m:],reference=rt,new=t);np.savez_compressed(a.out/'level1.npz',**{'v_'+k:v for k,v in predv.items()},**{'t_'+k:v for k,v in predt.items()});print('RESULT',summary,flush=True)
 if not a.predict:return
 full=np.load(a.previous/'full.npz');extra=np.load(a.out/'full_peer.npz');XX=np.c_[full['X'],extra['X']];Q=np.c_[full['Q'],extra['Q']];test=pd.read_csv(a.data_dir/'task_6_test.csv');accepted=pd.read_csv(a.accepted);pd.testing.assert_frame_equal(accepted.iloc[:,:-1],test.iloc[:,:-1]);preds={'accepted':accepted.delta.to_numpy()/86400}
 for name in runs:
  if weights[name]==0:continue
  cfg=runs[name];M=reg(name,cfg['iterations']);path=a.out/(name+'_full.model');F=XX;FQ=Q
  if name=='catlog':F=framing(F,p,full['rows']);FQ=framing(FQ,p,test=test)
  if path.exists():
   if name=='catlog':M.load_model(str(path))
   else:M=lgb.Booster(model_file=str(path),params={'num_threads':1})
  else:
   yy=np.log1p(full['y']) if cfg['logged_target'] else (full['y']-anchor(XX) if cfg.get('offset_target') else full['y'])
   if name=='catlog':M.fit(F,yy);M.save_model(str(path))
   else:M.fit(F,yy,sample_weight=user_weights(p,full['rows']) if cfg['logged_target'] or cfg.get('balanced_weighted') else None);M.booster_.save_model(str(path))
  preds[name]=forecast(M,FQ,cfg['logged_target'],anchor(Q) if cfg.get('offset_target') else None)
 new=sum(weights[k]*preds[k] for k in preds)*86400
 for label,pp in [('primary',new),('blend50',.5*new+.5*accepted.delta.to_numpy())]:
  d=a.out/label;d.mkdir(exist_ok=True);f=test.copy();f['delta']=pp;assert np.isfinite(pp).all() and (pp>=0).all();f.to_csv(d/'task_6_predictions.csv',index=False)
 np.savez_compressed(a.out/'test_predictions.npz',primary=new,**preds)
if __name__=='__main__':main()
