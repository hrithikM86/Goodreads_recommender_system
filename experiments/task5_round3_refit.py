"""Production-only NN refit on all known out-of-sample chronological pairs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,pickle
from pathlib import Path
import numpy as np
from task5_round3 import load,nn_fit,nn_margin
from task5_improve import data

def main():
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','previous','reference','out','accepted']:ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();r=json.loads((a.out/'results.json').read_text());assert r['weights']['nn']==1 and r['new_weight']==1,'This refit variant expects the selected NN architecture.';d,s,*_=load(a);epochs=r['runs']['nn']['epochs'];path=a.out/'nn_larger_refit.pkl'
 if path.exists():model=pickle.loads(path.read_bytes())
 else:model,_=nn_fit(d,s,epochs);temp=path.with_suffix('.tmp.pkl');temp.write_bytes(pickle.dumps(model,protocol=5));temp.replace(path)
 fd,fs,*_=load(a,True);value=nn_margin(*model,fd,fs);h,ctx,qu,qi,U,I,test=data(a.data_dir,a.baseline,True);folder=a.out/'larger_refit';folder.mkdir(exist_ok=True);test['which']=np.where(value>=0,test.item1,test.item2);test.to_csv(folder/'task_5_predictions.csv',index=False);np.savez_compressed(a.out/'larger_refit_margins.npz',primary=value);(a.out/'larger_refit_results.json').write_text(json.dumps({'model':'same selected neural ranker, same epochs','training_pairs':len(d),'labels':'penultimate and final known training events, each with an independently sampled negative','local_evaluation_accuracy':None,'reason':'Final-event labels join the production meta-training set after validation; no independent score is claimed for this refit.'},indent=2));print('LARGER_REFIT exported',len(test),'rows',flush=True)
if __name__=='__main__':main()
