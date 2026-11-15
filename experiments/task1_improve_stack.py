"""Calibrated Task 1 SVD++/learned-neighborhood ensemble, positive ratings only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from task1_improve import dataset,WEIGHTS,clip

def features(a,full=False):
    part=a.out/('full' if full else 'validation');lib=dict(np.load(part/'library.npz'));nb=dict(np.load(part/'neighbors.npz'));p,tr,q,u,i,qu,qi,users,items,n=dataset(a.data_dir,full);cos=np.column_stack([nb[f'cos{k}_{power}'] for k,power in [(10,1),(20,2),(10,2)]]);nw=WEIGHTS[3:]
    if full:
        ref=pd.read_csv(a.accepted);pd.testing.assert_frame_equal(ref.drop(columns='rating'),q.drop(columns='rating'));baseline=ref.rating.to_numpy();svd=clip((baseline-cos@nw)/WEIGHTS[:3].sum())
    else:
        svds=np.column_stack([lib[f'svd{k}'] for k in [16,32,64]]);svd=(svds@WEIGHTS[:3])/WEIGHTS[:3].sum();baseline=svds@WEIGHTS[:3]+cos@nw
        z=np.load(a.legacy/'task1_neighbors_tune_predictions.npz')
        for j,(k,power) in enumerate([(10,1),(20,2),(10,2)]):assert np.allclose(cos[:,j],np.r_[z[f'v_{k}_{power}'],z[f't_{k}_{power}']],atol=1e-10)
    columns={'accepted':baseline,'svd_aggregate':svd,**{f'cos{k}_{power}':cos[:,j] for j,(k,power) in enumerate([(10,1),(20,2),(10,2)])},**{k:lib[k] for k in lib if k.startswith('svdpp')},**{k:nb[k] for k in nb if k.startswith('joint')}}
    for key in ['user','item']:
        stats=tr.groupby(key).rating.agg(['count','mean','std'])
        for stat in ['count','mean','std']:columns[key+'_'+stat]=q[key].map(stats[stat]).fillna(float(tr.rating.mean()) if stat=='mean' else 0).to_numpy()
    return np.column_stack(list(columns.values())).astype(np.float32),list(columns),baseline,q,n

def workers(a,full=False):
    jobs=[]
    for kind in ['library','neighbors']:
        cmd=[sys.executable,str(Path(__file__).with_name('task1_improve.py')),'--data-dir',str(a.data_dir),'--legacy',str(a.legacy),'--out',str(a.out),'--worker',kind]+(['--full'] if full else []);h=open(a.out/(kind+('_full' if full else '_validation')+'.log'),'a');jobs.append((subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT),h))
    for proc,h in jobs:
        rc=proc.wait();h.close();assert rc==0,'Task1 worker failed'
def cfg(depth,iters=1800):return dict(iterations=iters,depth=depth,learning_rate=.03,l2_leaf_reg=15,random_seed=42,thread_count=1,loss_function='RMSE',verbose=300,allow_writing_files=False)
def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','legacy','out','accepted']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--predict',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);workers(a);X,names,ref,q,n=features(a);y=q.rating.to_numpy();idx=np.random.RandomState(17).permutation(n);tr=idx[:int(.8*n)];va=idx[int(.8*n):];mse=lambda y,p:float(np.mean((y-p)**2));reports={};inner={}
    for depth in [4,6]:
        name=f'depth{depth}';path=a.out/(name+'_inner.cbm')
        if path.exists():m=CatBoostRegressor();m.load_model(str(path))
        else:m=CatBoostRegressor(**cfg(depth));m.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=125);m.save_model(str(path))
        inner[name]=clip(m.predict(X[va]));reports[name]={'depth':depth,'iterations':m.tree_count_,'inner_MSE':mse(y[va],inner[name])};print('TASK1_INNER',reports[name],flush=True)
    score,w=min((mse(y[va],w*inner['depth4']+(1-w)*inner['depth6']),w) for w in [0,.25,.5,.75,1]);weights={k:v for k,v in {'depth4':w,'depth6':1-w}.items() if v>0};evals={}
    for name in weights:
        r=reports[name];m=CatBoostRegressor(**cfg(r['depth'],r['iterations']));m.fit(X[:n],y[:n]);m.save_model(str(a.out/(name+'_evaluation.cbm')));evals[name]=clip(m.predict(X[n:]))
    new=sum(weights[k]*v for k,v in evals.items());blend=.5*new+.5*ref[n:];summary={'features':names,'runs':reports,'weights':weights,'inner_MSE':score,'reference_evaluation_MSE':mse(y[n:],ref[n:]),'new_evaluation_MSE':mse(y[n:],new),'blend50_evaluation_MSE':mse(y[n:],blend),'selection_rows':n,'evaluation_rows':len(y)-n,'final_meta_rows':len(y),'reference_public_MSE':.5538262939223092,'input_manifest':json.loads((a.out/'validation/manifest.json').read_text())['inputs']};np.savez_compressed(a.out/'evaluation.npz',y=y[n:],reference=ref[n:],new=new,blend50=blend);(a.out/'results.json').write_text(json.dumps(summary,indent=2));print('TASK1_LOCAL',summary,flush=True)
    if not a.predict:return
    workers(a,True);FX,fn,reference,test,_=features(a,True);assert fn==names;preds={}
    for name in weights:
        r=reports[name];m=CatBoostRegressor(**cfg(r['depth'],r['iterations']));m.fit(X,y);m.save_model(str(a.out/(name+'_full.cbm')));preds[name]=clip(m.predict(FX))
    new=clip(sum(weights[k]*v for k,v in preds.items()));blend=clip(.5*new+.5*reference)
    for label,pred in [('new',new),('blend50',blend)]:
        folder=a.out/label;folder.mkdir(exist_ok=True);frame=test.copy();frame['rating']=pred;assert len(frame)==20000 and np.isfinite(pred).all();frame.to_csv(folder/'task_1_predictions.csv',index=False)
    np.savez_compressed(a.out/'test_predictions.npz',new=new,blend50=blend,reference=reference,**preds)
if __name__=='__main__':main()
