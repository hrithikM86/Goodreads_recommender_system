"""Verify diagonal-penalty ridge, saved coefficients and exported ensembles."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,pickle,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix,load_npz
from catboost import CatBoostRegressor
from task3_round4 import ridge_spectrum,dataset,clip
from task3_round3 import extended_design
from task3_round4_stack import load_components

def numerical_check():
    rng=np.random.RandomState(7)
    for n in [7,50]:
        A=np.c_[rng.randint(0,2,(n,6)),rng.normal(size=(n,6))];y=rng.normal(size=n)
        for ratio in [.25,1,4,16]:
            scale=np.r_[np.ones(6),np.full(6,1/np.sqrt(ratio))];left,rhs,eig=ridge_spectrum(A*scale,y)
            for penalty in [10,50,200]:
                w=(left@(rhs/(eig+penalty)))*scale;expected=np.linalg.solve(A.T@A+np.diag(np.r_[np.full(6,penalty),np.full(6,penalty*ratio)]),A.T@y);assert np.allclose(w,expected,atol=1e-12)
    print('PASS: primal/dual whitening equals direct separate-penalty ridge.',flush=True)

def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','legacy','prior','round2','round3','out']:ap.add_argument('--'+key,type=Path)
    ap.add_argument('--numerical-only',action='store_true');args=ap.parse_args();numerical_check()
    if args.numerical_only:return
    p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(args.data_dir,True);U,I=len(users),len(items);qt=q.timestamp.to_numpy(float);summary=json.loads((args.out/'results.json').read_text());selected=summary['component_selection'];values=load_components(args.out,True)
    with open(args.prior/'full/temporal_bias.pkl','rb') as f:bias=pickle.load(f)
    base=bias.predict(qu.astype(np.int64),qi.astype(np.int64),qt);R=coo_matrix((y-bias.training_prediction,(u,i)),shape=(U,I)).tocsr();X=coo_matrix((np.ones(len(y)),(u,i)),shape=(U,I)).tocsr();T=coo_matrix((t,(u,i)),shape=(U,I)).tocsr();known=(qu>=0)&(qi>=0);assert np.all(np.asarray(X[qu[known],qi[known]]).ravel()==0);sample=np.random.RandomState(47).choice(np.flatnonzero(known),64,replace=False)
    for name in selected['component_configs']:
        W=load_npz(args.out/'full'/(name+'.npz')).tocsc();rows,cols=W.nonzero();assert np.all(rows%I!=cols)
        ratio=float(name.split('_ratio')[1].split('_')[0]);penalty=float(name.split('_pen')[1])
        # Independently check the fitted normal equations on actual target books.
        # This does not reuse whitening or the eigensolver.
        for j in np.unique(qi[sample[:3]]):
            if W.indptr[j]==W.indptr[j+1]:continue
            lo,hi=W.indptr[j:j+2];ids=W.indices[lo:hi];weights=W.data[lo:hi].astype(float);nb=ids%I;blocks=ids//I;us=X[:,j].nonzero()[0];target=np.asarray(R[us,j].toarray()).ravel();design=np.column_stack([np.asarray((X if block==0 else R)[us,int(book)].toarray()).ravel() for book,block in zip(nb,blocks)]);gradient=design.T@(target-design@weights);expected=penalty*np.where(blocks==0,1,ratio)*weights;assert np.allclose(gradient,expected,rtol=1e-4,atol=1e-4),'Saved coefficients violate ridge normal equations'
        for z in sample:
            lo,hi=W.indptr[qi[z]:qi[z]+2];indices=W.indices[lo:hi];nb=indices%I;block=indices//I;weights=W.data[lo:hi];presence=np.asarray(X[qu[z],nb].toarray()).ravel();residual=np.asarray(R[qu[z],nb].toarray()).ravel();v=np.where(block==0,presence,residual);times=np.asarray(T[qu[z],nb].toarray()).ravel();tw=np.exp(-np.abs(times-qt[z])/86400/365)*((times>=0)&(times<=4102444800))
            for tau in [0,365]:assert np.allclose(clip(base[z]+weights@(v*(tw if tau else 1))),values[name+f'_t{tau}'][z],atol=1e-6)
    FX,_,names=extended_design(args.data_dir,args.legacy,args.prior,args.round2,args.round3,True);assert names==summary['base_features'];FZ=np.c_[FX,np.column_stack([values[k] for k in selected['component_features']])].astype(np.float32);preds={}
    for name in summary['selected']['depth_weights']:
        m=CatBoostRegressor();m.load_model(str(args.out/(name+'_full.cbm')));preds[name]=clip(m.predict(FZ))
    raw=clip(sum(summary['selected']['depth_weights'][k]*v for k,v in preds.items()));ref=np.load(args.round3/'test_predictions.npz')['primary'];w=summary['selected']['new_weight'];primary=clip(w*raw+(1-w)*ref);conservative=clip(.5*primary+.5*ref);audit={}
    for label,pred in [('primary',primary),('conservative',conservative),('raw',raw)]:
        path=args.out/label/'task_3_predictions.csv';frame=pd.read_csv(path);assert len(frame)==20000 and frame.rating.between(1,5).all() and np.isfinite(frame.rating).all();pd.testing.assert_frame_equal(frame.drop(columns='rating'),q.drop(columns='rating'));assert np.allclose(frame.rating,pred,atol=1e-12);audit[label]={'rows':len(frame),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'serialized_model_replay':True,'input_columns_preserved':True,'mean_absolute_change_from_accepted':float(np.mean(np.abs(pred-ref)))}
    inputs={f:hashlib.sha256((args.data_dir/f).read_bytes()).hexdigest() for f in ['tasks_3_4_train.csv','task_3_test.csv']};assert inputs==summary['input_manifest']
    for stage,full in [('validation',False),('full',True)]:
        for width in [64,256]:
            guard=json.loads((args.out/stage/f'manifest{width}.json').read_text());assert guard['inputs']==inputs and guard['full']==full
    order=np.random.RandomState(42).permutation(len(p));train=order[:int(.9*len(p))];held=order[int(.9*len(p)):];presence=coo_matrix((np.ones(len(train)),(u[train],i[train])),shape=(U,I)).tocsr();assert np.all(np.asarray(presence[u[held],i[held]]).ravel()==0)
    summary['audit']=audit;summary['checks']={'generalized_ridge_direct_solve_equivalence':True,'saved_normal_equations_verified':True,'independent_coefficient_replay':True,'target_books_excluded':True,'component_targets_held_out':True,'stage_and_data_hashes_verified':True,'full_ensemble_replay':True};summary['source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['task3_round4.py','task3_round4_stack.py','task3_round4_checks.py']};(args.out/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'audit':audit,'checks':summary['checks']},indent=2))
if __name__=='__main__':main()
