"""XGBoost/LightGBM replacements for accepted Task4 CatBoost, identical feature bank.
Selection is inner-only; evaluation targets never choose settings/checkpoints.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import sys,argparse,json,pickle,hashlib,zipfile,ast
from pathlib import Path
import numpy as np,pandas as pd
sys.path[:0]=['/workspace/work/task4_xgb','/workspace/work/task3_lgb']
import xgboost as xgb
import lightgbm as lgb
W=Path('/workspace/work');ROOT=W/'task4_two_approaches';OUT=W/'task4_tree_combiners';OUT.mkdir(exist_ok=True);DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026';NSEL=71462

def accuracy(p):return float(np.mean((p>0)+.5*(p==0)))
def matrices(full=False):
 z=np.load(ROOT/('fresh_full' if full else 'fresh_validation')/'scores.npz')['scores'];f=np.load(ROOT/('fm_full.npz' if full else 'fm_validation.npz'));F=np.c_[z,np.column_stack([f[k] for k in f.files])].astype('f');n=len(F)//2;d=F[:n]-F[n:];s=F[:n]+F[n:];assert np.isfinite(d).all() and np.isfinite(s).all();return d,s

def predict(model,kind,d,s,count=None):
 if kind=='xgboost':
  kw={} if count is None else {'iteration_range':(0,count)}
 else:kw={'num_iteration':count}
 return model.predict_proba(np.c_[d,s],**kw)[:,1]-model.predict_proba(np.c_[-d,s],**kw)[:,1]

def fit(kind,params,count,xx,yy,validation=None):
 if kind=='xgboost':
  model=xgb.XGBClassifier(n_estimators=count,learning_rate=.035,reg_lambda=20,subsample=.9,colsample_bytree=.9,n_jobs=2,random_state=619,tree_method='hist',objective='binary:logistic',eval_metric='logloss',**params,**({'early_stopping_rounds':100} if validation is not None else {}))
  model.fit(xx,yy,**({'eval_set':[validation],'verbose':False} if validation is not None else {}))
 else:
  model=lgb.LGBMClassifier(n_estimators=count,learning_rate=.035,reg_lambda=20,subsample=.9,subsample_freq=1,colsample_bytree=.9,n_jobs=2,random_state=619,verbosity=-1,objective='binary',**params)
  model.fit(xx,yy,**({'eval_set':[validation],'callbacks':[lgb.early_stopping(100,verbose=False)]} if validation is not None else {}))
 return model

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['xgboost','lightgbm'],required=True);ap.add_argument('--finish-completed',action='store_true');a=ap.parse_args();kind=a.kind;d,s=matrices();idx=np.random.RandomState(17).permutation(NSEL);cut=int(.8*NSEL);tr,va=idx[:cut],idx[cut:];X=np.r_[np.c_[d[tr],s[tr]],np.c_[-d[tr],s[tr]]];y=np.r_[np.ones(len(tr)),np.zeros(len(tr))];V=(np.r_[np.c_[d[va],s[va]],np.c_[-d[va],s[va]]],np.r_[np.ones(len(va)),np.zeros(len(va))]);configs=([{'max_depth':4,'min_child_weight':20},{'max_depth':6,'min_child_weight':30},{'max_depth':8,'min_child_weight':50}] if kind=='xgboost' else [{'num_leaves':15,'min_child_samples':100},{'num_leaves':31,'min_child_samples':150},{'num_leaves':63,'min_child_samples':250}]);best=(-1,None,None);reports=[]
 if a.finish_completed:
  for line in (W/('task4_'+kind+'.log')).read_text().splitlines():
   prefix='INNER '+kind+' '
   if line.startswith(prefix):
    r=ast.literal_eval(line[len(prefix):]);reports.append(r)
    if r['inner_accuracy']>best[0]:best=(r['inner_accuracy'],r['params'],r['iterations'])
  assert reports,'No completed inner-validation trials to refit'
 for params in ([] if a.finish_completed else configs):
  model=fit(kind,params,1600,X,y,V);limit=model.best_iteration+1 if kind=='xgboost' else model.best_iteration_;counts=sorted(set([limit]+list(range(100,limit+1,100))));score,count=max([(accuracy(predict(model,kind,d[va],s[va],count)),count) for count in counts],key=lambda x:(x[0],-x[1]));r={'params':params,'iterations':count,'inner_accuracy':score,'logloss_stopping_iteration':limit};reports.append(r);print('INNER',kind,r,flush=True)
  if score>best[0]:best=(score,params,count)
 score,params,count=best;X=np.r_[np.c_[d[:NSEL],s[:NSEL]],np.c_[-d[:NSEL],s[:NSEL]]];y=np.r_[np.ones(NSEL),np.zeros(NSEL)];model=fit(kind,params,count,X,y);ev=predict(model,kind,d[NSEL:],s[NSEL:]);old=np.load(ROOT/'fresh_margins.npz')['evaluation'];assert len(old)==len(ev);r={'kind':kind,'package_version':xgb.__version__ if kind=='xgboost' else lgb.__version__,'selected':{'params':params,'iterations':count,'inner_accuracy':score},'search':reports,'search_status':'Completed configurations only; unfinished exploratory fits excluded' if a.finish_completed else 'Full three-configuration search','evaluation_accuracy':accuracy(ev),'accepted_catboost_evaluation':accuracy(old),'feature_bank':'same fresh BPR/ALS/time-period/neighborhood/drift/FM features as accepted .9478 solution','component_refit':False,'protocol':'seed17 inner split before mirroring; selection rows only for final combiner; evaluation excluded from tuning','input_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['fresh_validation/scores.npz','fresh_full/scores.npz','fm_validation.npz','fm_full.npz']}};print('RESULT',kind,r,flush=True);(OUT/(kind+'_results.json')).write_text(json.dumps(r,indent=2));
 if kind=='xgboost':model.save_model(OUT/(kind+'.ubj'))
 else:model.booster_.save_model(str(OUT/(kind+'.txt')))
 np.savez_compressed(OUT/(kind+'_evaluation.npz'),margin=ev);fd,fs=matrices(True);p=predict(model,kind,fd,fs);assert np.allclose(p,-predict(model,kind,-fd,fs));test=pd.read_csv(DATA/'task_4_test.csv');assert len(test)==len(p) and np.isfinite(p).all();test['which']=np.where(p>=0,test.item1,test.item2);dest=Path('/workspace/output')/('task4_'+kind);dest.mkdir(exist_ok=True);csv=dest/'task_4_predictions.csv';test.to_csv(csv,index=False);np.savez_compressed(OUT/(kind+'_test.npz'),margin=p);f=pd.read_csv(csv);original=pd.read_csv(DATA/'task_4_test.csv');assert f.drop(columns='which').equals(original.drop(columns='which',errors='ignore'));assert ((f.which==f.item1)|(f.which==f.item2)).all();r['output_audit']={'rows':len(f),'sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),'changed_vs_accepted':int((f.which!=pd.read_csv('/workspace/output/task4_all_models_retrained_fm/task_4_predictions.csv').which).sum()),'pair_swap_antisymmetry':True};(OUT/(kind+'_results.json')).write_text(json.dumps(r,indent=2));print('EXPORTED',kind,r['output_audit'],flush=True)
if __name__=='__main__':main()
