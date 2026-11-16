"""Task5 chronological cohort checks and serialized combiner replay."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from task5_boosting import W,DATA,OUT,matrices,args,accuracy,xgb,lgb,CatBoostClassifier
from task5_improve import data

def cohorts():
 a=args();history,contexts,qu,qi,U,I,q=data(DATA,a.baseline,False);assert len(contexts)==2*U
 for row in range(2*U):assert qi[row] not in contexts[row]
 assert np.array_equal(qu[:2*U],qu[2*U:]);assert np.array_equal(qu[:U],qu[U:2*U]);print('All chronological selection/evaluation targets excluded from own query histories; cached query IDs aligned',flush=True)

def replay():
 d,s,names=matrices(True);q=pd.read_csv(DATA/'task_5_test.csv');reference=np.load(W/'task5_round3/preference/test_predictions.npz')['primary'];accepted_path=Path('/workspace/output/task5_round3_primary/task_5_predictions.csv');accepted=pd.read_csv(accepted_path);assert accepted.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert np.array_equal(accepted.which,np.where(reference>=0,q.item1,q.item2));report={}
 for kind in ['catboost','xgboost','lightgbm']:
  if kind=='catboost':m=CatBoostClassifier();m.load_model(str(OUT/(kind+'.cbm')));p=m.predict_proba(np.c_[d,s],thread_count=2)[:,1]-m.predict_proba(np.c_[-d,s],thread_count=2)[:,1]
  elif kind=='xgboost':m=xgb.XGBClassifier(n_jobs=2);m.load_model(OUT/(kind+'.ubj'));p=m.predict_proba(np.c_[d,s])[:,1]-m.predict_proba(np.c_[-d,s])[:,1]
  else:m=lgb.Booster(model_file=str(OUT/(kind+'.txt')));p=m.predict(np.c_[d,s],num_threads=2)-m.predict(np.c_[-d,s],num_threads=2)
  assert np.allclose(p,np.load(OUT/(kind+'_test.npz'))['margin'],atol=1e-7);r=json.loads((OUT/(kind+'_results.json')).read_text());assert names==r['features'] and r['feature_count']==260
  for name,h in r['source_input_hashes'].items():assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==h
  path=Path('/workspace/output')/('task5_boosting_'+kind)/'task_5_predictions.csv';f=pd.read_csv(path);assert len(f)==86786 and f.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert np.array_equal(f.which,np.where(p>=0,q.item1,q.item2));assert np.isfinite(p).all();assert ((f.which==f.item1)|(f.which==f.item2)).all();assert hashlib.sha256(path.read_bytes()).hexdigest()==r['output_audit']['sha256'];report[kind]={'rows':len(f),'saved_model_replay':True,'input_hashes_match':True,'features_match':True,'sha256':r['output_audit']['sha256']}
 report['accepted_reference_replay']=True;report['chronological_target_exclusion']=True;(OUT/'audit.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');a=ap.parse_args();cohorts()
 if a.replay:replay()
