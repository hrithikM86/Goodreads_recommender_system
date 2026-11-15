"""Recent-event MAE residual calibration, with inner-user model selection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor

def config(depth,iters=1000):return dict(depth=depth,iterations=iters,learning_rate=.035,l2_leaf_reg=25,loss_function='MAE',random_seed=91,thread_count=1,verbose=250,allow_writing_files=False)
def main():
 ap=argparse.ArgumentParser()
 for key in ['data-dir','previous','out','accepted']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--predict',action='store_true');a=ap.parse_args();z=np.load(a.previous/'validation.npz');peer=np.load(a.out/'validation_peer.npz');
 if (a.out/'level1.npz').exists():base=dict(np.load(a.out/'level1.npz'))
 else:
  from task6_next_models import reference
  from types import SimpleNamespace
  rv,rt=reference(SimpleNamespace(legacy=a.previous.parent,previous=a.previous),pd.read_csv(a.data_dir/'task_6_train.csv'),z);base={'v_accepted':rv,'t_accepted':rt}
 n=int(z['n']);m=int(z['m']);keys=['accepted'];X=np.c_[z['X'][n:],peer['X'][n:],np.log1p(np.column_stack([np.r_[base['v_'+k],base['t_'+k]] for k in keys]))];y=z['y'][n:];ref=np.r_[base['v_accepted'],base['t_accepted']];idx=np.random.RandomState(91).permutation(m);tr=idx[:int(.8*m)];va=idx[int(.8*m):];mae=lambda y,p:float(np.mean(abs(y-p))*86400);reports={};inner={}
 for depth in [3,5]:
  key=f'cal{depth}';path=a.out/(key+'_inner.cbm');M=CatBoostRegressor(**config(depth))
  if path.exists():M.load_model(str(path))
  else:M.fit(X[tr],y[tr]-ref[tr],eval_set=(X[va],y[va]-ref[va]),early_stopping_rounds=100);M.save_model(str(path))
  inner[key]=M.predict(X[va]);reports[key]={'depth':depth,'iterations':M.tree_count_};print('INNER',key,mae(y[va],np.maximum(0,ref[va]+inner[key])),flush=True)
 choices=[]
 for v in [0,.5,1]:
  delta=v*inner['cal3']+(1-v)*inner['cal5']
  for w in [0,.25,.5,.75,1]:choices.append((mae(y[va],np.maximum(0,ref[va]+w*delta)),v,w))
 score,v,w=min(choices);weights={'cal3':v,'cal5':1-v};delta=0
 for key,mix in weights.items():
  if mix==0:continue
  r=reports[key];M=CatBoostRegressor(**config(r['depth'],r['iterations']));path=a.out/(key+'_evaluation.cbm')
  if path.exists():M.load_model(str(path))
  else:M.fit(X[:m],y[:m]-ref[:m]);M.save_model(str(path))
  delta+=mix*M.predict(X[m:])
 primary=np.maximum(0,ref[m:]+w*delta);summary={'weights':weights,'correction_weight':w,'inner_MAE':score,'reference_evaluation_MAE':mae(y[m:],ref[m:]),'new_evaluation_MAE':mae(y[m:],primary),'runs':reports,'expert_names':keys};(a.out/'stack_results.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(a.out/'stack_evaluation.npz',y=y[m:],reference=ref[m:],primary=primary);print('STACK_RESULT',summary,flush=True)
 if not a.predict:return
 full=np.load(a.previous/'full.npz');extras=np.load(a.out/'full_peer.npz');anchor=pd.read_csv(a.accepted).delta.to_numpy()/86400;Q=np.c_[full['Q'],extras['Q'],np.log1p(anchor)];correction=0
 for key,mix in weights.items():
  if mix==0:continue
  r=reports[key];M=CatBoostRegressor(**config(r['depth'],r['iterations']));path=a.out/(key+'_full.cbm')
  if path.exists():M.load_model(str(path))
  else:M.fit(X,y-ref);M.save_model(str(path))
  correction+=mix*M.predict(Q)
 primary=np.maximum(0,anchor+w*correction)*86400;test=pd.read_csv(a.data_dir/'task_6_test.csv')
 for label,pred in [('stack_primary',primary),('stack_blend50',.5*primary+.5*anchor*86400)]:
  folder=a.out/label;folder.mkdir(exist_ok=True);frame=test.copy();frame['delta']=pred;assert np.isfinite(pred).all();frame.to_csv(folder/'task_6_predictions.csv',index=False)
if __name__=='__main__':main()
