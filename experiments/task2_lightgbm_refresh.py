"""Task2 LightGBM combiner: original16 signals, no Mult-VAE.
PYTHONPATH=/workspace/work/task3_lgb:/workspace/work/task4_xgb python experiments/task2_lightgbm_refresh.py
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import gzip,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from task2_xgboost_refresh import W,DATA,BASE,load_features,design,mirrored,labels,margin,accuracy,catmargin
OUT=W/'task2_lightgbm_refresh';OUT.mkdir(exist_ok=True);PREV=W/'task2_xgboost_refresh'
def estimator(count):
 return lgb.LGBMClassifier(objective='binary',n_estimators=count,num_leaves=31,min_child_samples=150,learning_rate=.025,reg_lambda=30.,reg_alpha=.1,max_bin=127,colsample_bytree=.9,subsample=.9,subsample_freq=1,random_state=42,n_jobs=2,verbosity=-1)
def score(model,d,s):
 if isinstance(model,lgb.Booster):return (model.predict(design(d,s),num_threads=2)-model.predict(design(-d,s),num_threads=2))/2
 return margin(model,d,s)
def main():
 prev=json.loads((PREV/'results.json').read_text());manifest=prev['input_manifest']
 for f in ['tasks_1_2_train.csv','task_2_test.csv']:assert hashlib.sha256((DATA/f).read_bytes()).hexdigest()==manifest[f]
 path=Path('/workspace/output/task2_xgboost_blend/task_2_predictions.csv');accepted_hash=hashlib.sha256(path.read_bytes()).hexdigest();assert accepted_hash=='9dd469b074af077ca895b1e1ed53b3b2fae456b3a98c9b8dde603d0169816880'
 names,d,s=load_features('validation');assert names==prev['feature_names'];z=np.load(BASE/'validation/validation_margins.npz');sel,ev=z['selection'],z['evaluation'];idx=np.random.RandomState(17).permutation(np.flatnonzero(sel));cut=int(.8*len(idx));tr,va=idx[:cut],idx[cut:];assert not np.intersect1d(tr,va).size and not (sel&ev).any()
 model=estimator(2500);model.fit(mirrored(d[tr],s[tr]),labels(len(tr)),eval_set=[(mirrored(d[va],s[va]),labels(len(va)))],eval_metric='binary_logloss',callbacks=[lgb.early_stopping(120,verbose=False),lgb.log_evaluation(300)])
 count=int(model.best_iteration_);pi=score(model,d[va],s[va]);oldx=xgb.XGBClassifier();oldx.load_model(PREV/'selection.ubj');cat=CatBoostClassifier();cat.load_model(str(PREV/'reference_inner.cbm'));ow=prev['xgboost_weight'];ci=ow*margin(oldx,d[va],s[va])+(1-ow)*catmargin(cat,d[va],s[va])
 search={str(float(w)):accuracy(w*pi+(1-w)*ci) for w in np.linspace(0,1,21)};weight=max(map(float,search),key=lambda w:(search[str(w)],-w))
 r={'config':estimator(count).get_params(),'feature_names':names,'iterations':count,'new_weight':weight,'inner_accepted_accuracy':accuracy(ci),'inner_lightgbm_accuracy':accuracy(pi),'inner_blend_accuracy':accuracy(weight*pi+(1-weight)*ci),'blend_search':search,'public_reference_accuracy':.9513,'protocol':'Original16 with-zero cached signals, no Mult-VAE. Differences/sums/absolute differences with mirrored pairs kept in same split. Seed17 inner80/20 of selection cohort determines count and weight; locked before final5% evaluation. Reference inner predictions from inner-trained accepted components. Production fits all186165 known holdout pairs using original sampled negatives. Full-data component scores reused. Historical splits reused; sampled negatives may differ from leaderboard.'}
 (OUT/'locked_selection.json').write_text(json.dumps(r,indent=2));print('LOCKED',json.dumps({k:v for k,v in r.items() if k not in ['config','feature_names','blend_search']}),flush=True)
 model=estimator(count);model.fit(mirrored(d[sel],s[sel]),labels(int(sel.sum())));pe=score(model,d[ev],s[ev]);ce=np.load(PREV/'evaluation.npz')['blend'];blend=weight*pe+(1-weight)*ce
 r.update(evaluation_accepted_accuracy=accuracy(ce),evaluation_lightgbm_accuracy=accuracy(pe),evaluation_blend_accuracy=accuracy(blend));np.savez_compressed(OUT/'evaluation.npz',reference=ce,lightgbm=pe,blend=blend);print('EVALUATION',json.dumps({k:v for k,v in r.items() if k.startswith('evaluation')}),flush=True)
 model=estimator(count);model.fit(mirrored(d,s),labels(len(d)));fn,td,ts=load_features('full');assert fn==names
 pn=score(model,td,ts)
 with gzip.open(OUT/'production.txt.gz','wt') as f:f.write(model.booster_.model_to_string())
 with gzip.open(OUT/'production.txt.gz','rt') as f:replay=lgb.Booster(model_str=f.read())
 assert np.allclose(pn,score(replay,td,ts),rtol=0,atol=1e-12) and np.allclose(pn,-score(model,-td,ts),atol=1e-12)
 oldx.load_model(PREV/'production.ubj');cat.load_model(str(BASE/'validation/stacker.cbm'));pc=ow*margin(oldx,td,ts)+(1-ow)*catmargin(cat,td,ts)
 test=pd.read_csv(DATA/'task_2_test.csv');accepted=pd.read_csv(path);pd.testing.assert_frame_equal(accepted.iloc[:,:-1],test.iloc[:,:-1]);assert np.array_equal(accepted.which.to_numpy(),np.where(pc>=0,test.item1,test.item2))
 r['outputs']={}
 for name,v in [('primary',pn),('blend',weight*pn+(1-weight)*pc)]:
  dest=Path('/workspace/output')/('task2_lightgbm_'+name);dest.mkdir(exist_ok=True);target=dest/'task_2_predictions.csv';frame=test.copy();frame['which']=np.where(v>=0,test.item1,test.item2);frame.to_csv(target,index=False);check=pd.read_csv(target);pd.testing.assert_frame_equal(check.iloc[:,:-1],test.iloc[:,:-1]);assert len(check)==20000 and np.isfinite(v).all() and ((check.which==check.item1)|(check.which==check.item2)).all()
  r['outputs'][name]={'path':str(target),'rows':len(check),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'changed_vs_accepted':int(np.sum(check.which.to_numpy()!=accepted.which.to_numpy()))}
 assert hashlib.sha256(path.read_bytes()).hexdigest()==accepted_hash
 r['input_manifest']=manifest;r['checks']={'production_model_replay':True,'candidate_swap_antisymmetry':True,'accepted_predictions_replay':True,'accepted_CSV_unchanged':True,'all16_original_features_used':True,'MultVAE_excluded':True}
 (OUT/'results.json').write_text(json.dumps(r,indent=2));print('EXPORTED',json.dumps(r['outputs']),flush=True)
if __name__=='__main__':main()
