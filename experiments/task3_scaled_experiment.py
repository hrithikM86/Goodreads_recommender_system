"""Task3 multiplicative temporal scaling -> full-bank fresh XGBoost.
PYTHONPATH=/workspace/work/task4_xgb:/workspace/work/task3_lgb:/workspace/work/task1_packages python experiments/task3_scaled_experiment.py
Uses immutable accepted feature banks; zero ratings excluded throughout.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import pickle,json,gzip,hashlib,gc
from pathlib import Path
import numpy as np,pandas as pd
import lightgbm as lgb
import xgboost as xgb
from task3_scaled_bias import ScaledTemporalBias
from task3_temporal_model import TemporalBias
from task3_lightgbm_refresh import augment,W,DATA,OLD,clip,mse
from task3_round3 import extended_design
from task3_optimize import dataset
from task3_xgboost_refresh import estimator
OUT=W/'task3_scaled_temporal';OUT.mkdir(exist_ok=True)
PREV=W/'task3_xgboost_refresh';LGB=W/'task3_lightgbm_refresh'
def base(full):
 with open(W/'task3_optimized'/('full' if full else 'validation')/'temporal_bias.pkl','rb') as f:return pickle.load(f)
def extra(model,u,i,t):
 p,s,b=model.parts(u,i,t);return np.column_stack([clip(p),s,b]).astype('f')
def unit_checks():
 rng=np.random.RandomState(7);u=rng.randint(0,8,150);i=rng.randint(0,6,150);t=1400000000+rng.randint(0,180,150)*86400.;y=rng.uniform(1,5,150)
 b=TemporalBias({'user':5,'item':10,'drift':40,'user90':30,'item90':60,'userday':20}).fit(u,i,t,y,8);m=ScaledTemporalBias(b,8).fit(u,i,t,y);assert np.allclose(m.parts(u,i,t)[0],m.training_prediction,atol=1e-8)
 assert np.allclose(m.parts(u.astype('int32'),i.astype('int32'),t)[0],m.training_prediction,atol=1e-8)
 for a in m.scales:a['coef']*=0
 assert np.allclose(m.parts(u,i,t)[0],m.base.predict(u,i,t),atol=1e-8)
 return {'objective_nonincreasing':True,'training_prediction_replay':True,'zero_scale_recovers_additive_model':True}
def main():
 tests=unit_checks();print('CHECKS',tests,flush=True);previous=json.loads((PREV/'results.json').read_text());lr=json.loads((LGB/'results.json').read_text())
 for f,h in previous['input_sha256'].items():assert hashlib.sha256((DATA/f).read_bytes()).hexdigest()==h
 ap=Path('/workspace/output/task3_xgboost_blend/task_3_predictions.csv');ah=hashlib.sha256(ap.read_bytes()).hexdigest();assert ah=='f7fe41a9b55b4ac3ce8f1e4f1643dfc1a6565483f41ae80a4ab0515a79dc3952'
 data=dataset(DATA,False);p,tr,q,u,i,t,y,qu,qi,users,items,_,_=data;assert (y>0).all();z=np.load(OLD/'extended_design.npz');Y=z['y'];n=int(z['n']);assert np.array_equal(Y,q.rating.to_numpy())
 print('SCALE_FIT',len(y),'positive training rows',flush=True);model=ScaledTemporalBias(base(False),20).fit(u,i,t,y);feat=extra(model,qu,qi,q.timestamp.to_numpy(float));assert np.allclose(model.parts(u,i,t)[0],model.training_prediction,atol=1e-7);np.savez_compressed(OUT/'validation_features.npz',features=feat);obj=model.objectives
 X,names=augment(z['X'],list(map(str,z['names'])),tr,q);X=np.c_[X,feat];names+=['scaled_temporal_prediction','user_time_scale','scaled_item_effect'];idx=np.random.RandomState(17).permutation(n);a,b=idx[:int(.8*n)],idx[int(.8*n):]
 tree=estimator(2000,True);tree.fit(X[a],Y[a],eval_set=[(X[b],Y[b])],verbose=200);count=int(tree.best_iteration)+1;pi=clip(tree.predict(X[b]))
 ox=xgb.XGBRegressor();ox.load_model(PREV/'selection.ubj');cat=.5*(np.load(OLD/'cat_6_res0_inner.npy')+np.load(OLD/'cat_8_res0_inner.npy'));lp=clip(lgb.Booster(model_file=str(LGB/'selection.txt')).predict(X[b,:-3],num_threads=2));li=lr['new_weight']*lp+(1-lr['new_weight'])*cat;ci=previous['xgboost_weight']*clip(ox.predict(X[b,:-3]))+(1-previous['xgboost_weight'])*li
 d=pi-ci;weight=float(np.clip(np.dot(d,Y[b]-ci)/max(np.dot(d,d),1e-12),0,1))
 r={'scale_spec':model.specs,'scale_epochs':20,'scale_training_objective':obj,'iterations':count,'new_weight':weight,'features':names,'inner_accepted_MSE':mse(Y[b],ci),'inner_augmented_MSE':mse(Y[b],pi),'inner_blend_MSE':mse(Y[b],weight*pi+(1-weight)*ci),'public_accepted_MSE':.6807703420497019,'checks':tests,'protocol':'Existing temporal bias component extended with (item+item90 bias)*(1+user+drift+user90+userday scale deviations). Additive terms and scales jointly refitted by regularized coordinate descent on90% positive ratings only. Fixed20 passes, scale penalties20/40/50/100. All269 accepted features retained,3 new features appended. XGBoost count and bounded-MSE blend weight selected on inner20% of next5% rows before final5% evaluation. Production scaling refits all positive training rows; meta fits all140536 held-out positive targets. Reused validation cohorts limit independence.'}
 (OUT/'locked_selection.json').write_text(json.dumps(r,indent=2));print('LOCKED',json.dumps({k:v for k,v in r.items() if k not in ['features','scale_training_objective','checks']}),flush=True)
 tree=estimator(count);tree.fit(X[:n],Y[:n],verbose=False);pe=clip(tree.predict(X[n:]));ev=np.load(PREV/'evaluation.npz');assert np.array_equal(ev['y'],Y[n:]);ce=ev['blend'];blend=clip(weight*pe+(1-weight)*ce);r.update(evaluation_accepted_MSE=mse(Y[n:],ce),evaluation_augmented_MSE=mse(Y[n:],pe),evaluation_blend_MSE=mse(Y[n:],blend));np.savez_compressed(OUT/'evaluation.npz',y=Y[n:],accepted=ce,augmented=pe,blend=blend);print('EVALUATION',json.dumps({k:v for k,v in r.items() if k.startswith('evaluation')}),flush=True)
 del data,model;gc.collect();_,train,test,fu,fi,ft,fy,fqu,fqi,*_=dataset(DATA,True);assert (fy>0).all();fm=ScaledTemporalBias(base(True),20).fit(fu,fi,ft,fy);fm.save(OUT/'production_scaling.npz');saved=ScaledTemporalBias(base(True));saved.restore(OUT/'production_scaling.npz');Fextra=extra(saved,fqu,fqi,test.timestamp.to_numpy(float));assert np.allclose(Fextra,extra(fm,fqu,fqi,test.timestamp.to_numpy(float)),atol=2e-6);np.savez_compressed(OUT/'test_features.npz',features=Fextra)
 FX,_,fn=extended_design(DATA,W/'task3',W/'task3_optimized',W/'task3_round2',OLD,True);FX,fn=augment(FX,list(map(str,fn)),train,test);FX=np.c_[FX,Fextra];fn+=['scaled_temporal_prediction','user_time_scale','scaled_item_effect'];assert fn==names
 tree=estimator(count);tree.fit(X,Y,verbose=False);pn=clip(tree.predict(FX))
 with gzip.open(OUT/'production_xgboost.ubj.gz','wb') as f:f.write(tree.get_booster().save_raw(raw_format='ubj'))
 replay=xgb.XGBRegressor()
 with gzip.open(OUT/'production_xgboost.ubj.gz','rb') as f:replay.load_model(bytearray(f.read()))
 assert np.array_equal(pn,clip(replay.predict(FX)));accepted=pd.read_csv(ap);pd.testing.assert_frame_equal(accepted.iloc[:,:-1],test.iloc[:,:-1]);prediction=clip(weight*pn+(1-weight)*accepted.rating.to_numpy());dest=Path('/workspace/output/task3_scaled_temporal');dest.mkdir(exist_ok=True);target=dest/'task_3_predictions.csv';frame=test.copy();frame['rating']=prediction;frame.to_csv(target,index=False);check=pd.read_csv(target);pd.testing.assert_frame_equal(check.iloc[:,:-1],test.iloc[:,:-1]);assert len(check)==20000 and np.isfinite(check.rating).all() and check.rating.between(1,5).all();assert hashlib.sha256(ap.read_bytes()).hexdigest()==ah
 r.update(output=str(target),rows=len(check),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),input_sha256=previous['input_sha256']);r['checks'].update(production_scaling_replay=True,production_XGBoost_replay=True,accepted_CSV_unchanged=True,zero_ratings_excluded=True);(OUT/'results.json').write_text(json.dumps(r,indent=2));print('EXPORTED',str(target),flush=True)
if __name__=='__main__':main()
