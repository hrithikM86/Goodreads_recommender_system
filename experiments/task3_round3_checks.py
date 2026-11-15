"""Independently replay learned predictors and audit exported Task 3 files."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix,load_npz
from catboost import CatBoostRegressor
from task3_round3 import dataset,extended_design,clip

def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','legacy','prior','round2','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--components-only',action='store_true')
    args=ap.parse_args();p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(args.data_dir,True);U,I=len(users),len(items);qt=q.timestamp.to_numpy(float)
    hashes={f:hashlib.sha256((args.data_dir/f).read_bytes()).hexdigest() for f in ['tasks_3_4_train.csv','task_3_test.csv']}
    for stage,full in [('validation',False),('full',True)]:assert json.loads((args.out/stage/'manifest.json').read_text())==dict(full=full,**hashes)
    with open(args.prior/'full/temporal_bias.pkl','rb') as f:bias=pickle.load(f)
    base=bias.predict(qu.astype(np.int64),qi.astype(np.int64),qt);r=y-bias.training_prediction;shape=(U,I);R=coo_matrix((r,(u,i)),shape=shape).tocsr();Y=coo_matrix((y,(u,i)),shape=shape).tocsr();T=coo_matrix((t,(u,i)),shape=shape).tocsr();RC=R.tocsc();YC=Y.tocsc();TC=T.tocsc()
    known=(qu>=0)&(qi>=0);assert np.all(np.asarray(Y[qu[known],qi[known]]).ravel()==0), 'Test pairs unexpectedly overlap observed positive training pairs'
    rng=np.random.RandomState(41);sample=rng.choice(np.flatnonzero((qu>=0)&(qi>=0)),64,replace=False);values=dict(np.load(args.out/'full/user_features.npz'));values.update(dict(np.load(args.out/'full/ordinal_features.npz')))
    for kind,reg in [('rating',5),('rating',25),('rating',100),('presence',25),('presence',100)]:
        W=load_npz(args.out/'full'/f'user_{kind}{reg}.npz').tocsc();assert np.count_nonzero(W.diagonal())==0
        for z in sample:
            lo,hi=W.indptr[qu[z]:qu[z]+2];nb=W.indices[lo:hi];weights=W.data[lo:hi];present=np.asarray(Y[nb,qi[z]].toarray()).ravel()>0;v=np.asarray(R[nb,qi[z]].toarray()).ravel() if kind=='rating' else present.astype(float);times=np.asarray(T[nb,qi[z]].toarray()).ravel();tw=np.exp(-np.abs(times-qt[z])/86400/365)*((times>=0)&(times<=4102444800))
            for tau in [0,365]:assert np.allclose(clip(base[z]+np.dot(weights,v*(tw if tau else 1))),values[f'user_ridge_{kind}{reg}_{tau}'][z],atol=1e-6)
    for kind in ['joint','ordinal']:
        blocks=2 if kind=='joint' else 5
        for reg in [10,50]:
            W=load_npz(args.out/'full'/f'book_{kind}{reg}.npz').tocsc();rr,cc=W.nonzero();assert np.all(rr%I!=cc)
            for z in sample:
                lo,hi=W.indptr[qi[z]:qi[z]+2];indices=W.indices[lo:hi];nb=indices%I;block=indices//I;weights=W.data[lo:hi];ratings=np.asarray(Y[qu[z],nb].toarray()).ravel();res=np.asarray(R[qu[z],nb].toarray()).ravel();times=np.asarray(T[qu[z],nb].toarray()).ravel();v=np.where(block==0,ratings>0,res) if kind=='joint' else (ratings==block+1).astype(float);tw=np.exp(-np.abs(times-qt[z])/86400/365)*((times>=0)&(times<=4102444800))
                for tau in [0,365]:assert np.allclose(clip(base[z]+np.dot(weights,v*(tw if tau else 1))),values[f'book_{kind}{reg}_{tau}'][z],atol=1e-6)
    if args.components_only:
        print('PASS: independent user, joint and ordinal coefficient replay; all self predictors excluded.')
        return
    summary=json.loads((args.out/'results.json').read_text());X,anchor,names=extended_design(args.data_dir,args.legacy,args.prior,args.round2,args.out,True);assert names==summary['features'];preds={}
    for name,weight in summary['selected']['weights'].items():
        residual=next(r['residual'] for r in summary['search'] if r['name']==name);model=CatBoostRegressor();model.load_model(str(args.out/(name+'_full.cbm')));preds[name]=clip(model.predict(X)+(anchor if residual else 0))
    primary=clip(sum(summary['selected']['weights'][name]*v for name,v in preds.items()));reference=np.load(args.round2/'test_predictions.npz')['blend'];conservative=clip(.5*primary+.5*reference);audit={}
    for label,v in [('primary',primary),('conservative',conservative)]:
        path=args.out/label/'task_3_predictions.csv';frame=pd.read_csv(path);assert len(frame)==20000;pd.testing.assert_frame_equal(frame.drop(columns='rating'),q.drop(columns='rating'));assert np.isfinite(frame.rating).all() and frame.rating.between(1,5).all();assert np.allclose(frame.rating,v,atol=1e-12);audit[label]={'rows':len(frame),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_replay_passed':True,'input_columns_preserved':True,'mean_absolute_change_from_reference':float(np.mean(np.abs(v-reference)))}
    if (args.out/'refit_results.json').exists():
        refit=json.loads((args.out/'refit_results.json').read_text());rp=[]
        for bank,r in refit.items():
            m=CatBoostRegressor();m.load_model(str(args.out/(bank+'_refit.cbm')));rp.append(clip(m.predict(X[:,r['feature_indices']])));assert r['meta_rows']==140536
        v=.5*(rp[0]+rp[1]);path=args.out/'refit/task_3_predictions.csv';frame=pd.read_csv(path);pd.testing.assert_frame_equal(frame.drop(columns='rating'),q.drop(columns='rating'));assert len(frame)==20000 and frame.rating.between(1,5).all() and np.allclose(frame.rating,v,atol=1e-12);audit['refit']={'rows':20000,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_replay_passed':True,'input_columns_preserved':True,'mean_absolute_change_from_reference':float(np.mean(np.abs(v-reference)))};summary['reference_refit']={'local_MSE':None,'reason':'Final stackers use all meta-training rows; no remaining holdout','models':refit}
    order=np.random.RandomState(42).permutation(len(p));held=order[int(.9*len(p)):];train=order[:int(.9*len(p))];presence=coo_matrix((np.ones(len(train)),(u[train],i[train])),shape=shape).tocsr();assert np.all(np.asarray(presence[u[held],i[held]]).ravel()==0)
    ev=np.load(args.out/'evaluation.npz');difference=(ev['y']-ev['reference'])**2-(ev['y']-.5*(ev['new']+ev['reference']))**2;eval_users=u[order[int(.95*len(p)):]];counts=np.bincount(eval_users,minlength=U);sums=np.bincount(eval_users,weights=difference,minlength=U);groups=int(np.count_nonzero(counts));se=float(np.sqrt(np.sum((sums-difference.mean()*counts)**2)/len(difference)**2*groups/(groups-1)));summary['local_gain_uncertainty']={'mean_MSE_reduction':float(difference.mean()),'user_clustered_standard_error':se,'approx_95pct_interval':[float(difference.mean()-1.96*se),float(difference.mean()+1.96*se)],'users':groups,'rows':len(difference),'scope':'Conditional local-split uncertainty; production all-meta refit has no separate holdout'}
    summary['audit']=audit;summary['checks']={'independent_user_regression_replay':True,'independent_joint_and_ordinal_replay':True,'self_predictors_excluded':True,'validation_pairs_absent_from_component_fit':True,'final_ensemble_replay':True,'zero_training_ratings_excluded':int(len(p))==1405354};summary['source_sha256']={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ['task3_round3.py','task3_round3_stack.py','task3_round3_refit.py','task3_round3_checks.py']};(args.out/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'audit':audit,'checks':summary['checks']},indent=2))
if __name__=='__main__':main()
