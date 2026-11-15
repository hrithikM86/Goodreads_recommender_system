"""Controlled refit: accepted round-2 architectures, all held-out train labels."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from task3_round3 import design,clip,dataset

def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','legacy','prior','round2','out']:ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    X,anchor,names=design(args.data_dir,args.legacy,args.prior,args.round2,args.out,include_user=False)
    FX,_,fn=design(args.data_dir,args.legacy,args.prior,args.round2,args.out,True,include_user=False);assert names==fn
    p,tr,q,*_=dataset(args.data_dir,False);y=q.rating.to_numpy();preds={};summary={}
    for bank in ['plain','common']:
        r=json.loads((args.round2/'models'/bank/'results.json').read_text());keep=list(range(167))+[names.index(k) for k in r['selected_features']];s=r['selected'];assert s['new_weight']==1
        cfg=dict(iterations=s['iterations'],depth=s['depth'],learning_rate=.025,l2_leaf_reg=15,random_seed=42,thread_count=1,loss_function='RMSE',allow_writing_files=False,verbose=250)
        m=CatBoostRegressor(**cfg);m.fit(X[:,keep],y);m.save_model(str(args.out/(bank+'_refit.cbm')));preds[bank]=clip(m.predict(FX[:,keep]));summary[bank]={'config':cfg,'feature_indices':keep,'meta_rows':len(y)}
    pred=clip(.5*preds['plain']+.5*preds['common']);test=pd.read_csv(args.data_dir/'task_3_test.csv');test['rating']=pred;folder=args.out/'refit';folder.mkdir(exist_ok=True);test.to_csv(folder/'task_3_predictions.csv',index=False);np.savez_compressed(args.out/'refit_predictions.npz',blend=pred,**preds);(args.out/'refit_results.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
