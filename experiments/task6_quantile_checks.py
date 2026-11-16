"""Replay final Task6 median ensemble and audit identical evaluation/test cohorts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from task6_quantile import OUT,W,DATA,CONFIGS,load,transform,mae
r=json.loads((OUT/'results.json').read_text());X,M,y,ref,n,m,tr,va,bi,be=load();assert not np.intersect1d(tr,va).size;assert len(tr)+len(va)==m;assert abs(sum(r['weights'].values())-1)<1e-10 and min(r['weights'].values())>=0
saved=np.load(W/'task6_next/stack_evaluation.npz');assert np.allclose(ref[m:],saved['reference'],atol=1e-8);old=CatBoostRegressor();old.load_model(str(W/'task6_next/cal5_evaluation.cbm'));assert np.allclose(be,np.maximum(0,ref[m:]+old.predict(M[m:])),atol=1e-8)
evaluation=np.load(OUT/'evaluation.npz');assert np.array_equal(evaluation['y'],y[n+m:]);ev=sum(r['weights'][k]*evaluation[k] for k in r['weights']);assert np.allclose(ev,evaluation['primary'],rtol=0,atol=1e-10);assert abs(mae(evaluation['y'],ev)-r['evaluation_new_MAE_seconds'])<1e-6
full=np.load(W/'task6_improved/full.npz');peer=np.load(W/'task6_next/full_peer.npz');Q=np.c_[full['Q'],peer['Q']].astype('f');anchor=pd.read_csv(W/'task6_improved/blend/task_6_predictions.csv').delta.to_numpy()/86400;MQ=np.c_[Q,np.log1p(anchor)].astype('f');accepted=pd.read_csv('/workspace/output/task6_round2_primary/task_6_predictions.csv').delta.to_numpy()/86400;result=r['weights']['accepted']*accepted
for name,c in CONFIGS.items():
 if r['weights'][name]==0:continue
 model=lgb.Booster(model_file=str(OUT/(name+'_production.txt')),params={'num_threads':2});assert model.params['objective']=='quantile';assert abs(float(model.params['alpha'])-.5)<1e-12;pred=transform(model.predict(MQ if c['kind']=='residual' else Q),c['kind'],anchor if c['kind']=='residual' else None);assert np.isfinite(pred).all() and (pred>=0).all();result+=r['weights'][name]*pred
result*=86400;assert np.allclose(result,np.load(OUT/'test.npz')['primary'],rtol=0,atol=1e-7);q=pd.read_csv(DATA/'task_6_test.csv');f=pd.read_csv(r['output']);pd.testing.assert_frame_equal(f.iloc[:,:-1],q.iloc[:,:-1]);assert len(f)==m==55403 and np.isfinite(f.delta).all() and (f.delta>=0).all();assert np.allclose(f.delta,result,rtol=1e-14,atol=1e-7)
report={'rows':len(f),'test_ids_order_header_preserved':f.columns.tolist()==q.columns.tolist(),'finite_nonnegative_seconds':True,'saved_model_ensemble_replay':True,'same_final_gap_evaluation_targets':True,'inner_reader_overlap':0,'quantile_alpha_half_verified':True,'evaluation_MAE_seconds':mae(evaluation['y'],ev),'input_feature_sha256':{str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [W/'task6_improved/validation.npz',W/'task6_improved/full.npz',W/'task6_next/validation_peer.npz',W/'task6_next/full_peer.npz']},'output_sha256':hashlib.sha256(Path(r['output']).read_bytes()).hexdigest()};assert report['output_sha256']==r['sha256'];(OUT/'checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
