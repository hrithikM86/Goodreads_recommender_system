"""Task 2: single XGBoost configuration on all cached implicit model scores.
Run: PYTHONPATH=/workspace/work/task4_xgb python experiments/task2_xgboost_refresh.py
Requires accepted Task 2 feature caches. Includes rating-zero interactions.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
W=Path('/workspace/work');DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026'
BASE=W/'task2_optimized/with_zero';OUT=W/'task2_xgboost_refresh';OUT.mkdir(exist_ok=True)
def accuracy(m):return float(np.mean((m>0)+.5*(m==0)))
def load_features(stage):
 z=np.load(BASE/stage/'features.npz');names=list(z.files);F=np.column_stack([z[k] for k in names]).astype('f');n=len(F)//2
 return names,F[:n]-F[n:],F[:n]+F[n:]
def design(d,s):return np.c_[d,s,abs(d)].astype('f')
def mirrored(d,s):return np.r_[design(d,s),design(-d,s)]
def labels(n):return np.r_[np.ones(n),np.zeros(n)]
def margin(model,d,s):return (model.predict_proba(design(d,s))[:,1]-model.predict_proba(design(-d,s))[:,1])/2

def estimator(count,early=False):
 kw=dict(objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_estimators=count,max_depth=5,min_child_weight=50,learning_rate=.035,reg_lambda=20.,reg_alpha=.1,max_bin=127,subsample=.9,colsample_bytree=.9,random_state=42,n_jobs=2)
 if early:kw['early_stopping_rounds']=100
 return xgb.XGBClassifier(**kw)
def catmargin(model,d,s):return (model.predict_proba(np.c_[d,s])[:,1]-model.predict_proba(np.c_[-d,s])[:,1])/2

def main():
 manifest=json.loads((BASE/'input_manifest.json').read_text())
 for f in ['tasks_1_2_train.csv','task_2_test.csv']:assert hashlib.sha256((DATA/f).read_bytes()).hexdigest()==manifest[f]
 accepted_path=Path('/workspace/output/task2_with_zero/task_2_predictions.csv');accepted_hash=hashlib.sha256(accepted_path.read_bytes()).hexdigest();assert accepted_hash=='6e948a3daaa6d6bafa4a36819cf22492697d04ac1d85e1f979d0a50eb90ff9da'
 names,d,s=load_features('validation');saved=np.load(BASE/'validation/validation_margins.npz');sel=saved['selection'];ev=saved['evaluation'];cfg=json.loads((BASE/'validation/results.json').read_text());assert names==cfg['feature_names'] and cfg['selected']['nonlinear_weight']==1
 idx=np.random.RandomState(17).permutation(np.flatnonzero(sel));cut=int(.8*len(idx));tr,va=idx[:cut],idx[cut:];assert not np.intersect1d(tr,va).size and not (sel&ev).any()
 print('FEATURES',len(names),'models/signals, training pairs',len(tr),'holdout pairs',len(va),flush=True)
 model=estimator(1600,True);model.fit(mirrored(d[tr],s[tr]),labels(len(tr)),eval_set=[(mirrored(d[va],s[va]),labels(len(va)))],verbose=200)
 count=int(model.best_iteration)+1;model.save_model(OUT/'selection.ubj');xp=margin(model,d[va],s[va])
 # Reproduce accepted CatBoost architecture on inner training only. Never use
 # its selection-wide model to estimate inner blend weights.
 c=CatBoostClassifier(iterations=cfg['selected']['iterations'],depth=cfg['selected']['depth'],learning_rate=.035,l2_leaf_reg=10,loss_function='Logloss',random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
 c.fit(np.r_[np.c_[d[tr],s[tr]],np.c_[-d[tr],s[tr]]],labels(len(tr)));c.save_model(str(OUT/'reference_inner.cbm'));cp=catmargin(c,d[va],s[va])
 search={str(float(w)):accuracy(w*xp+(1-w)*cp) for w in np.linspace(0,1,21)}
 weight=max(map(float,search),key=lambda w:(search[str(w)],-w))
 report={'feature_names':names,'config':estimator(count).get_params(),'iterations':count,'xgboost_weight':weight,'inner_reference_accuracy':accuracy(cp),'inner_xgboost_accuracy':accuracy(xp),'inner_blend_accuracy':accuracy(weight*xp+(1-weight)*cp),'blend_search':search,'public_accepted_accuracy':.9505,'protocol':'All16 accepted with-zero model scores reused. Symmetric pair features: score differences, sums and absolute differences. Mirrored rows stay in the same split. Existing90/5/5 row split; inner80/20 within selection, seed17. Count and blend weight locked before final5% evaluation. Production XGBoost fits all186165 known held-out pairs with original sampled negatives; full-data base scores reused for test. Historical cohorts reused; synthetic negative distribution may differ from leaderboard.'}
 (OUT/'locked_selection.json').write_text(json.dumps(report,indent=2));print('LOCKED',json.dumps({k:v for k,v in report.items() if k not in ['feature_names','config','blend_search']}),flush=True)
 model=estimator(count);model.fit(mirrored(d[sel],s[sel]),labels(int(sel.sum())),verbose=False);model.save_model(OUT/'evaluation.ubj');xe=margin(model,d[ev],s[ev]);ce=saved['margin'][ev]
 ref=CatBoostClassifier();ref.load_model(str(BASE/'validation/stacker.cbm'));assert np.allclose(ce,catmargin(ref,d[ev],s[ev]),atol=1e-12)
 report.update(evaluation_reference_accuracy=accuracy(ce),evaluation_xgboost_accuracy=accuracy(xe),evaluation_blend_accuracy=accuracy(weight*xe+(1-weight)*ce))
 np.savez_compressed(OUT/'evaluation.npz',reference=ce,xgboost=xe,blend=weight*xe+(1-weight)*ce)
 print('EVALUATION',json.dumps({k:v for k,v in report.items() if k.startswith('evaluation')}),flush=True)
 model=estimator(count);model.fit(mirrored(d,s),labels(len(d)),verbose=False);model.save_model(OUT/'production.ubj')
 fn,td,ts=load_features('full');assert fn==names
 tx=margin(model,td,ts);tc=catmargin(ref,td,ts);test=pd.read_csv(DATA/'task_2_test.csv');old=pd.read_csv(accepted_path)
 pd.testing.assert_frame_equal(old.iloc[:,:-1],test.iloc[:,:-1]);assert np.array_equal(old.which.to_numpy(),np.where(tc>=0,test.item1,test.item2))
 replay=xgb.XGBClassifier();replay.load_model(OUT/'production.ubj');assert np.array_equal(tx,margin(replay,td,ts)) and np.allclose(tx,-margin(model,-td,ts),atol=1e-7)
 report['outputs']={}
 for name,value in [('primary',tx),('blend',weight*tx+(1-weight)*tc)]:
  dest=Path('/workspace/output')/('task2_xgboost_'+name);dest.mkdir(exist_ok=True);path=dest/'task_2_predictions.csv';frame=test.copy();frame['which']=np.where(value>=0,test.item1,test.item2);assert np.isfinite(value).all();frame.to_csv(path,index=False)
  check=pd.read_csv(path);pd.testing.assert_frame_equal(check.iloc[:,:-1],test.iloc[:,:-1]);assert len(check)==20000 and ((check.which==check.item1)|(check.which==check.item2)).all()
  report['outputs'][name]={'path':str(path),'rows':len(check),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'changed_choices_vs_accepted':int(np.sum(check.which.to_numpy()!=old.which.to_numpy()))}
 assert hashlib.sha256(accepted_path.read_bytes()).hexdigest()==accepted_hash
 report['input_manifest']=manifest;(OUT/'results.json').write_text(json.dumps(report,indent=2));print('EXPORTED',json.dumps(report['outputs']),flush=True)
if __name__=='__main__':main()
