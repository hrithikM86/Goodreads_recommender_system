"""Train-only weights for accepted, SGD-FM-augmented and Bayesian-FM models."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from task3_round3 import extended_design,dataset,clip,mse
from task3_fm_stack import values

def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','legacy','prior','round2','round3','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--predict',action='store_true');a=ap.parse_args();s=json.loads((a.out/'results.json').read_text());bs=json.loads((a.out/'bayes_validation/results.json').read_text());bayes=np.load(a.out/'bayes_validation/features.npz')[bs['selected_feature']];X,_,names=extended_design(a.data_dir,a.legacy,a.prior,a.round2,a.round3);v=values(a.out);Z=np.c_[X,np.column_stack([v[k] for k in s['selection']['fm_features']])].astype(np.float32);p,tr,q,*_=dataset(a.data_dir,False);y=q.rating.to_numpy();n=s['meta_selection_rows'];idx=np.random.RandomState(17).permutation(n)[int(.8*n):];old=[];new=[]
    for depth in [6,8]:
        m=CatBoostRegressor();m.load_model(str(a.round3/f'cat_{depth}_res0_inner.cbm'));old.append(clip(m.predict(X[idx])))
    reference=.5*(old[0]+old[1]);raw=np.zeros(len(idx))
    for key,weight in s['selected']['depth_weights'].items():
        m=CatBoostRegressor();m.load_model(str(a.out/(key+'_inner.cbm')));raw+=weight*clip(m.predict(Z[idx]))
    candidates=[]
    for fw in [0,.25,.5,.75,1]:
        for bw in [0,.025,.05,.1,.15,.2,.3]:
            if fw+bw>1:continue
            pred=fw*raw+bw*bayes[idx]+(1-fw-bw)*reference;candidates.append((mse(y[idx],pred),fw,bw))
    score,fw,bw=min(candidates);ev=np.load(a.out/'evaluation.npz');pred=fw*ev['augmented']+bw*bayes[n:]+(1-fw-bw)*ev['reference'];result={'weights':{'sgd_augmented':fw,'bayesian_fm':bw,'accepted':1-fw-bw},'inner_MSE':score,'evaluation_MSE':mse(ev['y'],pred),'reference_evaluation_MSE':s['reference_evaluation_MSE'],'bayesian_checkpoint':bs['checkpoint'],'bayesian_evaluation_MSE':mse(ev['y'],bayes[n:]),'bayesian_selection_MSE':mse(y[:n],bayes[:n])};(a.out/'bayes_blend_results.json').write_text(json.dumps(result,indent=2));np.savez_compressed(a.out/'bayes_evaluation.npz',y=ev['y'],pred=pred,bayesian=bayes[n:],reference=ev['reference']);print(json.dumps(result,indent=2),flush=True)
    if not a.predict or bw==0:return
    subprocess.run([sys.executable,str(Path(__file__).with_name('task3_bayesian_fm.py')),'--data-dir',str(a.data_dir),'--out',str(a.out/'bayes_full'),'--full','--selection',str(a.out/'bayes_validation/results.json')],check=True)
    bayes=np.load(a.out/'bayes_full/features.npz')[bs['selected_feature']];z=np.load(a.out/'test_predictions.npz');pred=clip(fw*z['augmented']+bw*bayes+(1-fw-bw)*z['reference']);frame=pd.read_csv(a.data_dir/'task_3_test.csv');frame['rating']=pred;folder=a.out/'bayes_blend';folder.mkdir(exist_ok=True);frame.to_csv(folder/'task_3_predictions.csv',index=False)
if __name__=='__main__':main()
