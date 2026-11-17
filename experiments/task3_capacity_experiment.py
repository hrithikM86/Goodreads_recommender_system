"""96/128-factor temporal SVD++ -> separate full-bank XGBoost ensembles.
PYTHONPATH=/workspace/work/task4_xgb:/workspace/work/task3_lgb:/workspace/work/task1_packages python experiments/task3_capacity_experiment.py
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import json,gzip,hashlib,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
import xgboost as xgb,lightgbm as lgb
from task3_capacity_models import run_worker,compile_kernel,OUT
from task3_lightgbm_refresh import W,DATA,OLD,augment,clip,mse
from task3_optimize import dataset
from task3_round3 import extended_design
from task3_xgboost_refresh import estimator
SC=W/'task3_scaled_temporal';PREV=W/'task3_xgboost_refresh';LG=W/'task3_lightgbm_refresh'
def workers(specs):
 with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
  jobs={pool.submit(run_worker,k,stage,epochs):k for k,stage,epochs in specs};result={}
  for f in as_completed(jobs):result[jobs[f]]=f.result()
 return result

def accepted_inner(X,Y,a,b,sr):
 lr=json.loads((LG/'results.json').read_text());pr=json.loads((PREV/'results.json').read_text());cat=.5*(np.load(OLD/'cat_6_res0_inner.npy')+np.load(OLD/'cat_8_res0_inner.npy'));lp=clip(lgb.Booster(model_file=str(LG/'selection.txt')).predict(X[b,:269],num_threads=2));old=lr['new_weight']*lp+(1-lr['new_weight'])*cat;m=xgb.XGBRegressor();m.load_model(PREV/'selection.ubj');old=pr['xgboost_weight']*clip(m.predict(X[b,:269]))+(1-pr['xgboost_weight'])*old;m=estimator(sr['iterations']);m.fit(X[a],Y[a],verbose=False);pred=sr['new_weight']*clip(m.predict(X[b]))+(1-sr['new_weight'])*old;assert abs(mse(Y[b],pred)-sr['inner_blend_MSE'])<1e-10;return pred

def append(X,p,names,k):
 old=X[:,names.index('joint48strong_e50')];neighbor=.5*(X[:,names.index('learned_rating25_365')]+X[:,names.index('wide_learned_rating25_365')]);return np.c_[X,p,p-old,p-neighbor].astype('f')
def main():
 sr=json.loads((SC/'results.json').read_text());manifest={'input_sha256':sr['input_sha256'],'kernel_sha256':hashlib.sha256(Path(__file__).with_name('task3_joint_kernel.cpp').read_bytes()).hexdigest(),'factors':[48,96,128],'original_split_seed':42,'nested_split_seed':91,'nested_training_fraction':.9,'fresh_checkpoints':[15,30,50,75,100,150,200],'params':{'seed':47,'learning_rate':.005,'factor_regularization':.08,'day_penalty':30,'item_period_penalty':40,'user_period_penalty':10,'dimension_normalized_initialization':True}}
 for f,h in sr['input_sha256'].items():assert hashlib.sha256((DATA/f).read_bytes()).hexdigest()==h
 guard=OUT/'manifest.json'
 if guard.exists():assert json.loads(guard.read_text())==manifest,'Changed inputs/config; use a fresh directory'
 else:guard.write_text(json.dumps(manifest,indent=2))
 ap=Path('/workspace/output/task3_scaled_temporal/task_3_predictions.csv');ah=hashlib.sha256(ap.read_bytes()).hexdigest();assert ah==sr['sha256'];compile_kernel()
 fresh=workers([(k,'fresh',None) for k in [48,96,128]]);print('FRESH_RESULTS',json.dumps(fresh),flush=True)
 f48=np.load(OUT/'48/fresh_scores.npz');tune=int(f48['tune']);reference=f48['prediction'][tune:];report={'fresh':fresh,'public_reference_MSE':.6804250167145823,'fresh_protocol':'RandomState91 split within original RandomState42 base-training90% pool:81% of full positives fit fresh models;9% new holdout split into equal epoch-selection/evaluation cohorts. This entire holdout is disjoint from cached meta10%. Matched48-factor model uses same regularization/seed and initialization-variance normalization. Evaluation half never selects epochs.','production_parameter_note':'Large experimental parameter states round-tripped through RAM, not persisted. Configurations, source and prediction arrays retained for reproducible refitting; accepted model checkpoints unchanged.'}
 for k in [96,128]:
  z=np.load(OUT/str(k)/'fresh_scores.npz');assert np.array_equal(z['y'],f48['y']);delta=(z['prediction'][tune:]-z['y'][tune:])**2-(reference-z['y'][tune:])**2;fresh[k]['evaluation_MSE_change_vs48']=float(delta.mean())
 (OUT/'fresh_summary.json').write_text(json.dumps(report,indent=2))
 workers([(k,'validation',fresh[k]['selected_epochs']) for k in [96,128]])
 z=np.load(OLD/'extended_design.npz');Y=z['y'];n=int(z['n']);_,train,q,*_=dataset(DATA,False);assert np.array_equal(Y,q.rating.to_numpy()) and (Y>0).all();X,names=augment(z['X'],list(map(str,z['names'])),train,q);X=np.c_[X,np.load(SC/'validation_features.npz')['features']];names+=['scaled_temporal_prediction','user_time_scale','scaled_item_effect'];assert names==sr['features'];idx=np.random.RandomState(17).permutation(n);a,b=idx[:int(.8*n)],idx[int(.8*n):];ci=accepted_inner(X,Y,a,b,sr);ev=np.load(SC/'evaluation.npz');ce=ev['blend'];assert np.array_equal(ev['y'],Y[n:]);runs={}
 for k in [96,128]:
  path=OUT/str(k);p=np.load(path/'validation_scores.npz')['prediction'];A=append(X,p,names,k);m=estimator(2000,True);m.fit(A[a],Y[a],eval_set=[(A[b],Y[b])],verbose=200);count=int(m.best_iteration)+1;pi=clip(m.predict(A[b]));d=pi-ci;weight=float(np.clip(np.dot(d,Y[b]-ci)/max(np.dot(d,d),1e-12),0,1));r={'factors':k,'factor_epochs':fresh[k]['selected_epochs'],'combiner_iterations':count,'new_weight':weight,'inner_accepted_MSE':mse(Y[b],ci),'inner_augmented_MSE':mse(Y[b],pi),'inner_blend_MSE':mse(Y[b],weight*pi+(1-weight)*ci)};(path/'locked_selection.json').write_text(json.dumps(r,indent=2));print('META_LOCKED',json.dumps(r),flush=True)
  m=estimator(count);m.fit(A[:n],Y[:n],verbose=False);pe=clip(m.predict(A[n:]));blend=clip(weight*pe+(1-weight)*ce);r.update(evaluation_accepted_MSE=mse(Y[n:],ce),evaluation_augmented_MSE=mse(Y[n:],pe),evaluation_blend_MSE=mse(Y[n:],blend));np.savez_compressed(path/'evaluation.npz',y=Y[n:],accepted=ce,augmented=pe,blend=blend);runs[k]=r;print('META_EVALUATION',json.dumps(r),flush=True)
 report['runs']=runs;(OUT/'evaluation_summary.json').write_text(json.dumps(report,indent=2));workers([(k,'full',fresh[k]['selected_epochs']) for k in [96,128]])
 FX,_,fn=extended_design(DATA,W/'task3',W/'task3_optimized',W/'task3_round2',OLD,True);_,full,test,*_=dataset(DATA,True);FX,fn=augment(FX,list(map(str,fn)),full,test);FX=np.c_[FX,np.load(SC/'test_features.npz')['features']];fn+=['scaled_temporal_prediction','user_time_scale','scaled_item_effect'];assert fn==names;accepted=pd.read_csv(ap);pd.testing.assert_frame_equal(accepted.iloc[:,:-1],test.iloc[:,:-1]);report['outputs']={}
 for k in [96,128]:
  path=OUT/str(k);p=np.load(path/'validation_scores.npz')['prediction'];fp=np.load(path/'full_scores.npz')['prediction'];A=append(X,p,names,k);F=append(FX,fp,names,k);r=runs[k];m=estimator(r['combiner_iterations']);m.fit(A,Y,verbose=False);pn=clip(m.predict(F))
  with gzip.open(path/'production_combiner.ubj.gz','wb') as f:f.write(m.get_booster().save_raw(raw_format='ubj'))
  replay=xgb.XGBRegressor()
  with gzip.open(path/'production_combiner.ubj.gz','rb') as f:replay.load_model(bytearray(f.read()))
  assert np.array_equal(pn,clip(replay.predict(F)));value=clip(r['new_weight']*pn+(1-r['new_weight'])*accepted.rating.to_numpy());dest=Path('/workspace/output')/('task3_timesvdpp'+str(k));dest.mkdir(exist_ok=True);target=dest/'task_3_predictions.csv';frame=test.copy();frame['rating']=value;frame.to_csv(target,index=False);check=pd.read_csv(target);pd.testing.assert_frame_equal(check.iloc[:,:-1],test.iloc[:,:-1]);assert len(check)==20000 and np.isfinite(check.rating).all() and check.rating.between(1,5).all();report['outputs'][str(k)]={'path':str(target),'rows':len(check),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'factor_parameter_replay':json.loads((path/'full_results.json').read_text())['in_memory_checkpoint_replay'],'combiner_replay':True};print('EXPORTED',k,str(target),flush=True)
 assert hashlib.sha256(ap.read_bytes()).hexdigest()==ah
 report.update(input_manifest=manifest,accepted_CSV_unchanged=True,zero_ratings_excluded=True,meta_protocol='Original272 feature bank plus capacity score and differences from48-factor and learned-neighborhood predictions. New-factor epochs selected exclusively inside original base-training pool; meta count/weight selected inside original inner20% before final holdout evaluation. Separate files per capacity. Reused meta cohorts; no leaderboard guarantee.');(OUT/'results.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
