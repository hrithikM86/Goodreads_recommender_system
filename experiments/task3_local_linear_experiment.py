"""Add local-linear residual features to accepted128-factor Task3 ensemble.
PYTHONPATH=/workspace/work/task4_xgb:/workspace/work/task3_lgb:/workspace/work/task1_packages python experiments/task3_local_linear_experiment.py
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import json,gzip,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import xgboost as xgb
from task3_local_linear_features import generate,unit_checks
from task3_capacity_experiment import accepted_inner,append
from task3_lightgbm_refresh import W,DATA,OLD,augment,clip,mse
from task3_optimize import dataset
from task3_round3 import extended_design
from task3_xgboost_refresh import estimator
OUT=W/'task3_local_linear';OUT.mkdir(exist_ok=True);CAP=W/'task3_capacity_extended';SC=W/'task3_scaled_temporal'
def main():
 checks=unit_checks();cr=json.loads((CAP/'results.json').read_text());sr=json.loads((SC/'results.json').read_text());cap=cr['runs']['128'];manifest={'input':sr['input_sha256'],'settings':'user30/90/365 lambda5/20, item90/365 lambda20/80, temporal-bias training residuals','accepted_hash':cr['outputs']['128']['sha256']}
 for f,h in manifest['input'].items():assert hashlib.sha256((DATA/f).read_bytes()).hexdigest()==h
 guard=OUT/'manifest.json'
 if guard.exists():assert json.loads(guard.read_text())==manifest
 else:guard.write_text(json.dumps(manifest,indent=2))
 ap=Path('/workspace/output/task3_timesvdpp128/task_3_predictions.csv');ah=hashlib.sha256(ap.read_bytes()).hexdigest();assert ah==manifest['accepted_hash']
 print('UNIT_CHECKS',checks,flush=True);L,ln,const=generate(OUT,False);z=np.load(OLD/'extended_design.npz');Y=z['y'];n=int(z['n']);_,train,q,*_=dataset(DATA,False);assert np.array_equal(Y,q.rating.to_numpy());X,names=augment(z['X'],list(map(str,z['names'])),train,q);X=np.c_[X,np.load(SC/'validation_features.npz')['features']];names+=['scaled_temporal_prediction','user_time_scale','scaled_item_effect'];X=append(X,np.load(CAP/'128/validation_scores.npz')['prediction'],names,128);assert X.shape[1]==275
 idx=np.random.RandomState(17).permutation(n);a,b=idx[:int(.8*n)],idx[int(.8*n):];precap=accepted_inner(X[:,:272],Y,a,b,sr);m=estimator(cap['combiner_iterations']);m.fit(X[a],Y[a],verbose=False);ci=cap['new_weight']*clip(m.predict(X[b]))+(1-cap['new_weight'])*precap;assert abs(mse(Y[b],ci)-cap['inner_blend_MSE'])<1e-10
 ev=np.load(CAP/'128/evaluation.npz');ce=ev['blend'];assert np.array_equal(ev['y'],Y[n:]);runs={}
 for name,cols in [('constant_control',const),('local_linear',list(range(L.shape[1])))]:
  A=np.c_[X,L[:,cols]];m=estimator(2000,True);m.fit(A[a],Y[a],eval_set=[(A[b],Y[b])],verbose=200);count=int(m.best_iteration)+1;pi=clip(m.predict(A[b]));d=pi-ci;weight=float(np.clip(np.dot(d,Y[b]-ci)/max(np.dot(d,d),1e-12),0,1));r={'variant':name,'feature_indices':cols,'combiner_iterations':count,'new_weight':weight,'inner_accepted_MSE':mse(Y[b],ci),'inner_augmented_MSE':mse(Y[b],pi),'inner_blend_MSE':mse(Y[b],weight*pi+(1-weight)*ci)};(OUT/(name+'_locked.json')).write_text(json.dumps(r,indent=2));print('LOCKED',json.dumps(r),flush=True)
  m=estimator(count);m.fit(A[:n],Y[:n],verbose=False);pe=clip(m.predict(A[n:]));pb=clip(weight*pe+(1-weight)*ce);r.update(evaluation_accepted_MSE=mse(Y[n:],ce),evaluation_augmented_MSE=mse(Y[n:],pe),evaluation_blend_MSE=mse(Y[n:],pb));np.savez_compressed(OUT/(name+'_evaluation.npz'),y=Y[n:],accepted=ce,augmented=pe,blend=pb);runs[name]=r;print('EVALUATION',json.dumps(r),flush=True)
 report={'runs':runs,'checks':checks,'public_accepted_MSE':.6801266912314015,'feature_names':ln,'protocol':'Residuals from accepted temporal-bias component fitted only original90% positive ratings. Fit local weighted intercept and signed slope around query time with exponential kernels; shrink sparse groups to zero. User tau30/90/365 and item tau90/365 fixed. Both sides of query timestamp allowed because Task3 is rating prediction. All275 accepted-capacity features retained. Matched local-constant control shares kernels/residuals/support. Meta iterations/weights selected only inner20% before final5% evaluation, independently per variant; historical cohorts reused. Production smoothers use all positive training rows, and meta fits all140536 known out-of-sample ratings.'};(OUT/'evaluation_summary.json').write_text(json.dumps(report,indent=2))
 LF,lfn,_=generate(OUT,True);assert lfn==ln;FX,_,fn=extended_design(DATA,W/'task3',W/'task3_optimized',W/'task3_round2',OLD,True);_,full,test,*_=dataset(DATA,True);FX,fn=augment(FX,list(map(str,fn)),full,test);FX=np.c_[FX,np.load(SC/'test_features.npz')['features']];fn+=['scaled_temporal_prediction','user_time_scale','scaled_item_effect'];assert fn==names;FX=append(FX,np.load(CAP/'128/full_scores.npz')['prediction'],names,128)
 # User requested the local-linear candidate; keep control diagnostic only.
 r=runs['local_linear'];A=np.c_[X,L];F=np.c_[FX,LF];m=estimator(r['combiner_iterations']);m.fit(A,Y,verbose=False);pn=clip(m.predict(F))
 with gzip.open(OUT/'production_xgboost.ubj.gz','wb') as f:f.write(m.get_booster().save_raw(raw_format='ubj'))
 replay=xgb.XGBRegressor()
 with gzip.open(OUT/'production_xgboost.ubj.gz','rb') as f:replay.load_model(bytearray(f.read()))
 assert np.array_equal(pn,clip(replay.predict(F)));accepted=pd.read_csv(ap);pd.testing.assert_frame_equal(accepted.iloc[:,:-1],test.iloc[:,:-1]);pred=clip(r['new_weight']*pn+(1-r['new_weight'])*accepted.rating.to_numpy());dest=Path('/workspace/output/task3_local_linear');dest.mkdir(exist_ok=True);path=dest/'task_3_predictions.csv';frame=test.copy();frame['rating']=pred;frame.to_csv(path,index=False);check=pd.read_csv(path);pd.testing.assert_frame_equal(check.iloc[:,:-1],test.iloc[:,:-1]);assert len(check)==20000 and np.isfinite(check.rating).all() and check.rating.between(1,5).all() and hashlib.sha256(ap.read_bytes()).hexdigest()==ah
 report.update(output=str(path),rows=len(check),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_manifest=manifest);report['checks'].update(training_row_alignment_verified=True,features_finite=True,production_model_replay=True,accepted_CSV_unchanged=True,zero_ratings_excluded=True);(OUT/'results.json').write_text(json.dumps(report,indent=2));print('EXPORTED',str(path),flush=True)
if __name__=='__main__':main()
