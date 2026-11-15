"""Select enlarged rating ensemble on inner holdout, then refit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from task1_improve_stack import features,cfg
from task1_improve import clip

def matrices(a,full=False):
 args=SimpleNamespace(data_dir=a.data_dir,legacy=a.legacy,out=a.previous,accepted=a.original);X,names,old,q,n=features(args,full);part=a.out/('full' if full else 'validation')
 for kind in ['svdpp','svdppstrong','itemspp','neighbors']:
  z=np.load(part/(kind+'.npz'));X=np.c_[X,np.column_stack([z[k] for k in z.files])];names+=z.files
 return X,names,q,n

def save(model,path):
 temp=path.with_suffix('.tmp.cbm');model.save_model(str(temp));os.replace(temp,path)

def main():
 ap=argparse.ArgumentParser()
 for key in ['data-dir','legacy','previous','out','accepted','original']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--predict',action='store_true');a=ap.parse_args();X,names,q,n=matrices(a);y=q.rating.to_numpy();idx=np.random.RandomState(17).permutation(n);tr=idx[:int(.8*n)];va=idx[int(.8*n):];prev=json.loads((a.previous/'results.json').read_text());refinner=0;refeval=0;mse=lambda y,p:float(np.mean((y-p)**2))
 for key,w in prev['weights'].items():
  m=CatBoostRegressor();m.load_model(str(a.previous/(key+'_inner.cbm')));refinner+=w*clip(m.predict(X[va,:23]));m.load_model(str(a.previous/(key+'_evaluation.cbm')));refeval+=w*clip(m.predict(X[n:,:23]))
 reports={};inner={};evals={}
 for depth in [4,6]:
  key=f'depth{depth}';path=a.out/(key+'_inner.cbm');m=CatBoostRegressor(**cfg(depth,2000))
  if path.exists():m.load_model(str(path))
  else:m.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=125);save(m,path)
  inner[key]=clip(m.predict(X[va]));reports[key]={'depth':depth,'iterations':m.tree_count_,'inner_MSE':mse(y[va],inner[key])};print('INNER',reports[key],flush=True)
 choices=[]
 for d in [0,.25,.5,.75,1]:
  raw=d*inner['depth4']+(1-d)*inner['depth6']
  for w in [0,.25,.5,.75,1]:choices.append((mse(y[va],w*raw+(1-w)*refinner),d,w))
 score,d,w=min(choices);weights={'depth4':d,'depth6':1-d};new=0
 for key,mix in weights.items():
  if mix==0:continue
  r=reports[key];path=a.out/(key+'_evaluation.cbm');m=CatBoostRegressor(**cfg(r['depth'],r['iterations']))
  if path.exists():m.load_model(str(path))
  else:m.fit(X[:n],y[:n]);save(m,path)
  evals[key]=clip(m.predict(X[n:]));new+=mix*evals[key]
 pred=w*new+(1-w)*refeval;summary={'features':names,'runs':reports,'depth_weights':weights,'new_weight':w,'inner_MSE':score,'reference_evaluation_MSE':mse(y[n:],refeval),'new_evaluation_MSE':mse(y[n:],new),'primary_evaluation_MSE':mse(y[n:],pred),'reference_public_MSE':.5385673542232966};(a.out/'results.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(a.out/'evaluation.npz',y=y[n:],reference=refeval,new=new,primary=pred);print('RESULT',summary,flush=True)
 if not a.predict:return
 FX,fn,test,_=matrices(a,True);assert fn==names;ref=pd.read_csv(a.accepted);pd.testing.assert_frame_equal(ref.iloc[:,:-1],test.iloc[:,:-1]);raw=0
 for key,mix in weights.items():
  if mix==0:continue
  r=reports[key];path=a.out/(key+'_full.cbm');m=CatBoostRegressor(**cfg(r['depth'],r['iterations']))
  if path.exists():m.load_model(str(path))
  else:m.fit(X,y);save(m,path)
  raw+=mix*clip(m.predict(FX))
 primary=w*raw+(1-w)*ref.rating.to_numpy()
 for label,pred in [('primary',primary),('new_only',raw),('blend50',.5*primary+.5*ref.rating.to_numpy())]:
  dest=a.out/label;dest.mkdir(exist_ok=True);f=test.copy();f['rating']=clip(pred);assert np.isfinite(pred).all();f.to_csv(dest/'task_1_predictions.csv',index=False)
 np.savez_compressed(a.out/'test_predictions.npz',primary=primary,new_only=raw)
if __name__=='__main__':main()
