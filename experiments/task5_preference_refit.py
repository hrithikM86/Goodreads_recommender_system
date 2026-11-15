"""Production-only expanded-data refit with approximately fixed SGD update budget."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,pickle,math
from pathlib import Path
import numpy as np
from task5_preference_round3 import load
from task5_round3 import nn_fit,nn_margin
from task5_improve import data

def main():
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','previous','reference','out','accepted']:ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();meta=a.out/'preference';r=json.loads((meta/'results.json').read_text());base=json.loads((a.out/'results.json').read_text());assert base['weights']['nn']==1 and base['new_weight']==1;d,s,parts,names=load(a);od,oscore=parts[:2];basepath=meta/'base_budget_refit.pkl';base_epochs=max(1,math.ceil(base['runs']['nn']['epochs']/2))
 if basepath.exists():bm=pickle.loads(basepath.read_bytes())
 else:bm,_=nn_fit(od,oscore,base_epochs);basepath.write_bytes(pickle.dumps(bm,protocol=5))
 models={};epochs={}
 for key,w in r['weights'].items():
  if w==0 or r['new_weight']==0:continue
  cfg=r['runs'][key];epochs[key]=max(1,math.ceil(cfg['epochs']/2));path=meta/(key+'_budget_refit.pkl')
  if path.exists():m=pickle.loads(path.read_bytes())
  else:m,_=nn_fit(d,s,epochs[key],seed=cfg['seed'],hidden=cfg['hidden'],alpha=cfg['alpha']);path.write_bytes(pickle.dumps(m,protocol=5))
  models[key]=m
 fd,fs,fp,fn=load(a,True);assert fn==names;ref=nn_margin(*bm,fp[0],fp[1]);new=sum(r['weights'][key]*nn_margin(*m,fd,fs) for key,m in models.items()) if models else np.zeros(len(ref));primary=r['new_weight']*new+(1-r['new_weight'])*ref;h,ctx,qu,qi,U,I,test=data(a.data_dir,a.baseline,True);folder=meta/'larger_refit';folder.mkdir(exist_ok=True);test['which']=np.where(primary>=0,test.item1,test.item2);test.to_csv(folder/'task_5_predictions.csv',index=False);np.savez_compressed(meta/'larger_refit_margins.npz',primary=primary);(meta/'larger_refit_results.json').write_text(json.dumps({'training_pairs':len(d),'base_epochs':base_epochs,'preference_epochs':epochs,'local_accuracy':None,'protocol':'Both historical held-out target sets become production meta-training examples after validation. Epoch counts are halved to roughly preserve the number of SGD minibatch updates. No independent accuracy is claimed for this refit.'},indent=2));print('PREFERENCE_LARGER_REFIT',len(test),'rows',flush=True)
if __name__=='__main__':main()
