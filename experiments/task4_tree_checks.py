"""Replay serialized combiners against cached component predictions and all output rows."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from task4_tree_combiners import matrices,predict,OUT,DATA,xgb,lgb

def main():
 d,s=matrices(True);test=pd.read_csv(DATA/'task_4_test.csv');report={}
 for kind in ['xgboost','lightgbm']:
  r=json.loads((OUT/(kind+'_results.json')).read_text())
  if kind=='xgboost':
   m=xgb.XGBClassifier(n_jobs=2);m.load_model(OUT/(kind+'.ubj'));p=predict(m,kind,d,s)
  else:
   m=lgb.Booster(model_file=str(OUT/(kind+'.txt')));p=m.predict(np.c_[d,s],num_threads=2)-m.predict(np.c_[-d,s],num_threads=2)
  saved=np.load(OUT/(kind+'_test.npz'))['margin'];assert np.allclose(p,saved,atol=1e-7);path=Path('/workspace/output')/('task4_'+kind)/'task_4_predictions.csv';f=pd.read_csv(path);assert len(f)==len(test);assert f.drop(columns='which').equals(test.drop(columns='which',errors='ignore'));assert np.array_equal(f.which,np.where(p>=0,test.item1,test.item2));assert np.isfinite(p).all();report[kind]={'rows':len(f),'saved_model_replay':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'local_accuracy':r['evaluation_accuracy']}
 (OUT/'replay_audit.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
