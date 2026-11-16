"""Fresh standalone CatBoost/XGBoost/LightGBM over the complete accepted Task5 bank.
Fixed initial configurations; no manual blend weights, no evaluation-label tuning.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import sys,argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
sys.path[:0]=['/workspace/work/task4_xgb','/workspace/work/task3_lgb']
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from task5_preference_round3 import load
W=Path('/workspace/work');DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026';OUT=W/'task5_boosting';OUT.mkdir(exist_ok=True)
def args():return SimpleNamespace(data_dir=DATA,baseline=W/'task5/results',previous=W/'task5_improved',reference=W/'task5_next',out=W/'task5_round3')
def matrices(full=False):
 d,s,parts,extra=load(args(),full);names=json.loads((W/'task5/results/results.json').read_text())['feature_names']+parts[-1]+extra;assert d.shape[1]==len(names)==260 and np.isfinite(d).all() and np.isfinite(s).all();return d,s,names

def margin(model,kind,d,s):
 kw={'thread_count':2} if kind=='catboost' else {};return model.predict_proba(np.c_[d,s],**kw)[:,1]-model.predict_proba(np.c_[-d,s],**kw)[:,1]
def accuracy(x):return float(np.mean((x>0)+.5*(x==0)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['catboost','xgboost','lightgbm'],required=True);a=ap.parse_args();kind=a.kind;d,s,names=matrices();U=len(d)//2;assert U==86786;X=np.r_[np.c_[d[:U],s[:U]],np.c_[-d[:U],s[:U]]];y=np.r_[np.ones(U),np.zeros(U)];common={'random_seed':1429,'learning_rate':.05,'iterations':800,'depth':6,'l2_leaf_reg':20} if kind=='catboost' else ({'random_state':1429,'learning_rate':.04,'n_estimators':1000,'max_depth':5,'min_child_weight':30,'reg_lambda':30,'subsample':.9,'colsample_bytree':.8} if kind=='xgboost' else {'random_state':1429,'learning_rate':.04,'n_estimators':1000,'num_leaves':31,'min_child_samples':200,'reg_lambda':30,'reg_alpha':.1,'subsample':.9,'subsample_freq':1,'colsample_bytree':.8});print('TRAIN',kind,common,'shape',X.shape,flush=True)
 if kind=='catboost':m=CatBoostClassifier(**common,loss_function='Logloss',thread_count=2,verbose=200,allow_writing_files=False);m.fit(X,y)
 elif kind=='xgboost':m=xgb.XGBClassifier(**common,n_jobs=2,tree_method='hist',objective='binary:logistic',eval_metric='logloss');m.fit(X,y)
 else:m=lgb.LGBMClassifier(**common,n_jobs=2,verbosity=-1,objective='binary');m.fit(X,y)
 ev=margin(m,kind,d[U:],s[U:]);reference=np.load(W/'task5_round3/preference/evaluation.npz')['primary'];assert len(ev)==len(reference);r={'kind':kind,'configuration':common,'feature_count':260,'features':names,'selection_pairs':U,'evaluation_pairs':len(ev),'evaluation_accuracy':accuracy(ev),'accepted_neural_ensemble_evaluation':accuracy(reference),'protocol':'Complete accepted component bank unchanged; fixed initial combiner settings, selected without using evaluation outcomes. Train on penultimate-event selection pairs only; final-event evaluation labels excluded. Pair mirroring and antisymmetric prediction. No percentages or inherited final ensemble scores in the feature bank.','ratings_used':False,'numeric_timestamps_used':False,'zero_rating_interactions_retained':True,'source_input_hashes':{f:hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in ['task_5_train.csv','task_5_test.csv']}};print('EVALUATION',kind,r['evaluation_accuracy'],'accepted',r['accepted_neural_ensemble_evaluation'],flush=True)
 if kind=='catboost':m.save_model(str(OUT/(kind+'.cbm')))
 elif kind=='xgboost':m.save_model(OUT/(kind+'.ubj'))
 else:m.booster_.save_model(str(OUT/(kind+'.txt')))
 np.savez_compressed(OUT/(kind+'_evaluation.npz'),margin=ev);del X,y,d,s;fd,fs,fn=matrices(True);assert fn==names;p=margin(m,kind,fd,fs);assert np.allclose(p,-margin(m,kind,-fd,fs));q=pd.read_csv(DATA/'task_5_test.csv');assert len(p)==len(q)==86786;assert np.isfinite(p).all();f=q.copy();f['which']=np.where(p>=0,f.item1,f.item2);dest=Path('/workspace/output')/('task5_boosting_'+kind);dest.mkdir(exist_ok=True);path=dest/'task_5_predictions.csv';f.to_csv(path,index=False);read=pd.read_csv(path);assert read.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert ((read.which==read.item1)|(read.which==read.item2)).all();np.savez_compressed(OUT/(kind+'_test.npz'),margin=p);old=pd.read_csv('/workspace/output/task5_round3_primary/task_5_predictions.csv');r['output_audit']={'rows':len(read),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'changed_vs_accepted':int((read.which!=old.which).sum()),'pair_swap_antisymmetry':True};(OUT/(kind+'_results.json')).write_text(json.dumps(r,indent=2));print('EXPORTED',kind,r['output_audit'],flush=True)
if __name__=='__main__':main()
