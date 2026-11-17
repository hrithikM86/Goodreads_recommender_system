"""Task3 rating-distribution expert + user-disjoint stacking robustness check.
No class reweighting: expectation is evaluated against actual rating prevalence.
PYTHONPATH=/workspace/work/task4_xgb:/workspace/work/task3_lgb:/workspace/work/task1_packages python experiments/task3_distribution_experiment.py
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import json,gzip,hashlib,pickle,gc
from pathlib import Path
import numpy as np,pandas as pd
import xgboost as xgb,lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from task3_lightgbm_refresh import W,DATA,OLD,augment,clip,mse,estimator as lgb_estimator
from task3_xgboost_refresh import estimator as regressor
from task3_optimize import dataset
from task3_round3 import extended_design
OUT=W/'task3_distribution';OUT.mkdir(exist_ok=True);SC=W/'task3_scaled_temporal';PREV=W/'task3_xgboost_refresh';LG=W/'task3_lightgbm_refresh'
def rating_mse(y,p):return float(np.mean((np.asarray(p).reshape(-1,5)@np.arange(1,6)-(y+1))**2))
def classifier(count,early=False):
 kw=dict(objective='multi:softprob',num_class=5,n_estimators=count,tree_method='hist',max_depth=4,min_child_weight=50,learning_rate=.035,reg_lambda=40,reg_alpha=.1,max_bin=127,subsample=.9,colsample_bytree=.9,random_state=271,n_jobs=2,eval_metric=rating_mse)
 if early:kw['callbacks']=[xgb.callback.EarlyStopping(rounds=100,metric_name='rating_mse',data_name='validation_0',maximize=False,save_best=True)]
 return xgb.XGBClassifier(**kw)
def mean_predict(model,X):
 p=model.predict_proba(X);assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
 return p@np.arange(1,6)
def prior_inner(X,indices):
 lr=json.loads((LG/'results.json').read_text());pr=json.loads((PREV/'results.json').read_text());sr=json.loads((SC/'results.json').read_text())
 lp=clip(lgb.Booster(model_file=str(LG/'selection.txt')).predict(X[indices,:269],num_threads=2));cats=.5*(np.load(OLD/'cat_6_res0_inner.npy')+np.load(OLD/'cat_8_res0_inner.npy'));li=lr['new_weight']*lp+(1-lr['new_weight'])*cats
 m=xgb.XGBRegressor();m.load_model(PREV/'selection.ubj');return pr['xgboost_weight']*clip(m.predict(X[indices,:269]))+(1-pr['xgboost_weight'])*li

def grouped_reference(X,y,tr,va):
 # Every meta component is refitted: reusing full original stackers here
 # would leak this new validation cohort's rating labels.
 preds={}
 for depth,count in [(6,1550),(8,908)]:
  m=CatBoostRegressor(iterations=count,depth=depth,learning_rate=.025,l2_leaf_reg=20,random_seed=42,thread_count=2,loss_function='RMSE',allow_writing_files=False,verbose=False);m.fit(X[tr,:245],y[tr]);preds[str(depth)]=clip(m.predict(X[va,:245]));print('GROUPED_REFERENCE_CAT',depth,flush=True)
 old=.5*(preds['6']+preds['8']);lr=json.loads((LG/'results.json').read_text());pr=json.loads((PREV/'results.json').read_text());sr=json.loads((SC/'results.json').read_text())
 m=lgb_estimator(lr['iterations']);m.fit(X[tr,:269],y[tr]);old=lr['new_weight']*clip(m.predict(X[va,:269]))+(1-lr['new_weight'])*old
 m=regressor(pr['iterations']);m.fit(X[tr,:269],y[tr],verbose=False);old=pr['xgboost_weight']*clip(m.predict(X[va,:269]))+(1-pr['xgboost_weight'])*old
 m=regressor(sr['iterations']);m.fit(X[tr],y[tr],verbose=False);old=sr['new_weight']*clip(m.predict(X[va]))+(1-sr['new_weight'])*old
 return old

def main():
 z=np.load(OLD/'extended_design.npz');Y=z['y'];n=int(z['n']);p,train,q,*_=dataset(DATA,False);assert np.array_equal(Y,q.rating.to_numpy()) and (Y>0).all();sr=json.loads((SC/'results.json').read_text())
 for f,h in sr['input_sha256'].items():assert hashlib.sha256((DATA/f).read_bytes()).hexdigest()==h
 accepted_path=Path('/workspace/output/task3_scaled_temporal/task_3_predictions.csv');ah=hashlib.sha256(accepted_path.read_bytes()).hexdigest();assert ah==sr['sha256']
 X,names=augment(z['X'],list(map(str,z['names'])),train,q);X=np.c_[X,np.load(SC/'validation_features.npz')['features']];names+=['scaled_temporal_prediction','user_time_scale','scaled_item_effect'];assert names==sr['features'];idx=np.random.RandomState(17).permutation(n);a,b=idx[:int(.8*n)],idx[int(.8*n):]
 model=classifier(1800,True);model.fit(X[a],Y[a].astype(int)-1,eval_set=[(X[b],Y[b].astype(int)-1)],verbose=150);count=int(model.best_iteration)+1;pi=mean_predict(model,X[b])
 old=prior_inner(X,b);m=regressor(sr['iterations']);m.fit(X[a],Y[a],verbose=False);ci=sr['new_weight']*clip(m.predict(X[b]))+(1-sr['new_weight'])*old;assert abs(mse(Y[b],ci)-sr['inner_blend_MSE'])<1e-10
 d=pi-ci;weight=float(np.clip(np.dot(d,Y[b]-ci)/max(np.dot(d,d),1e-12),0,1));report={'classifier_trees':count,'classifier_config':{k:v for k,v in classifier(count).get_params().items() if k not in ['eval_metric','callbacks']},'new_weight':weight,'inner_accepted_MSE':mse(Y[b],ci),'inner_distribution_MSE':mse(Y[b],pi),'inner_blend_MSE':mse(Y[b],weight*pi+(1-weight)*ci),'features':names,'protocol':'Five-class probabilities, expectation for MSE; no class balancing. Same272 label-safe base features. Count/weight selected on original inner validation only. Historical final holdout evaluates locked settings. Additional seed83 user-disjoint80/20 split across all140536 meta rows refits every accepted meta component with fixed settings; no new weight search there. Base models remain fit on original90% positive training rows, excluding every meta target. This is a second meta-level validation, not a fully independent base-model split. Historical cohorts reused.'}
 (OUT/'locked_selection.json').write_text(json.dumps(report,indent=2));print('LOCKED',json.dumps({k:v for k,v in report.items() if k not in ['features','classifier_config']}),flush=True)
 model=classifier(count);model.fit(X[:n],Y[:n].astype(int)-1,verbose=False);pe=mean_predict(model,X[n:]);ev=np.load(SC/'evaluation.npz');assert np.array_equal(ev['y'],Y[n:]);ce=ev['blend'];blend=clip(weight*pe+(1-weight)*ce);report.update(evaluation_accepted_MSE=mse(Y[n:],ce),evaluation_distribution_MSE=mse(Y[n:],pe),evaluation_blend_MSE=mse(Y[n:],blend));np.savez_compressed(OUT/'evaluation.npz',y=Y[n:],accepted=ce,distribution=pe,blend=blend);print('HISTORICAL_EVALUATION',json.dumps({k:v for k,v in report.items() if k.startswith('evaluation')}),flush=True)
 gtr,gva=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=83).split(X,Y,groups=q.user.to_numpy()));assert not set(q.user.iloc[gtr])&set(q.user.iloc[gva]);print('GROUPED_START',len(gtr),len(gva),flush=True)
 cref=grouped_reference(X,Y,gtr,gva);model=classifier(count);model.fit(X[gtr],Y[gtr].astype(int)-1,verbose=False);pp=mean_predict(model,X[gva]);pb=clip(weight*pp+(1-weight)*cref);report.update(grouped_accepted_MSE=mse(Y[gva],cref),grouped_distribution_MSE=mse(Y[gva],pp),grouped_blend_MSE=mse(Y[gva],pb),grouped_train_rows=len(gtr),grouped_validation_rows=len(gva),grouped_user_overlap=0);np.savez_compressed(OUT/'grouped_evaluation.npz',rows=gva,y=Y[gva],accepted=cref,distribution=pp,blend=pb);print('GROUPED_EVALUATION',json.dumps({k:v for k,v in report.items() if k.startswith('grouped')}),flush=True)
 report['consistent_improvement']=bool(report['evaluation_blend_MSE']<report['evaluation_accepted_MSE'] and report['grouped_blend_MSE']<report['grouped_accepted_MSE'] and weight>0)
 # Export experimental candidate even if the robustness check disagrees;
 # preserve the accepted file and label the result honestly in the report.
 FX,_,fn=extended_design(DATA,W/'task3',W/'task3_optimized',W/'task3_round2',OLD,True);_,full,test,*_=dataset(DATA,True);FX,fn=augment(FX,list(map(str,fn)),full,test);FX=np.c_[FX,np.load(SC/'test_features.npz')['features']];fn+=['scaled_temporal_prediction','user_time_scale','scaled_item_effect'];assert fn==names
 model=classifier(count);model.fit(X,Y.astype(int)-1,verbose=False);pn=mean_predict(model,FX)
 with gzip.open(OUT/'production_classifier.ubj.gz','wb') as f:f.write(model.get_booster().save_raw(raw_format='ubj'))
 replay=xgb.XGBClassifier()
 with gzip.open(OUT/'production_classifier.ubj.gz','rb') as f:replay.load_model(bytearray(f.read()))
 assert np.allclose(pn,mean_predict(replay,FX),rtol=0,atol=1e-12)
 accepted=pd.read_csv(accepted_path);pd.testing.assert_frame_equal(accepted.iloc[:,:-1],test.iloc[:,:-1]);pred=clip(weight*pn+(1-weight)*accepted.rating.to_numpy());dest=Path('/workspace/output/task3_distribution');dest.mkdir(exist_ok=True);path=dest/'task_3_predictions.csv';frame=test.copy();frame['rating']=pred;frame.to_csv(path,index=False);check=pd.read_csv(path);pd.testing.assert_frame_equal(check.iloc[:,:-1],test.iloc[:,:-1]);assert len(check)==20000 and np.isfinite(check.rating).all() and check.rating.between(1,5).all() and hashlib.sha256(accepted_path.read_bytes()).hexdigest()==ah
 report.update(output=str(path),rows=len(check),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_sha256=sr['input_sha256'],public_reference_MSE=.6804250167145823);(OUT/'results.json').write_text(json.dumps(report,indent=2));print('EXPORTED',str(path),'consistent',report['consistent_improvement'],flush=True)
if __name__=='__main__':main()
