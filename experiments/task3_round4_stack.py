"""Select wider split-penalty neighborhoods and stack using train labels only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,hashlib,itertools,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from task3_round3 import extended_design,dataset,clip,mse
from task3_round4 import RATIOS,PENALTIES,config_name

def config(depth,iterations=1800):
    return dict(iterations=iterations,depth=depth,learning_rate=.025,l2_leaf_reg=20,random_seed=42,thread_count=2,loss_function='RMSE',allow_writing_files=False,verbose=250)

def worker(args,full=False):
    jobs=[]
    for width in [64,256]:
        cmd=[sys.executable,str(Path(__file__).with_name('task3_round4.py')),'--data-dir',str(args.data_dir),'--prior',str(args.prior),'--round2',str(args.round2),'--out',str(args.out),'--width',str(width)]
        if full:cmd+=['--full','--selection',str(args.out/'selection.json')]
        handle=open(args.out/(f'worker{width}_'+('full' if full else 'validation')+'.log'),'a');jobs.append((subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT),handle))
    for job,handle in jobs:
        rc=job.wait();handle.close();assert rc==0,'Neighborhood worker failed'

def load_components(out,full=False):
    values={}
    for width in [64,256]:
        z=np.load(out/('full' if full else 'validation')/f'features{width}.npz');values.update({k:z[k] for k in z.files})
    return values

def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','legacy','prior','round2','round3','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--predict',action='store_true');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest={f:hashlib.sha256((args.data_dir/f).read_bytes()).hexdigest() for f in ['tasks_3_4_train.csv','task_3_test.csv']};assert json.loads((args.round3/'stack_manifest.json').read_text())['input_manifest']==manifest
    worker(args);X,anchor,names=extended_design(args.data_dir,args.legacy,args.prior,args.round2,args.round3);p,tr,q,*_=dataset(args.data_dir,False);y=q.rating.to_numpy();n=int(.95*len(p))-int(.9*len(p));idx=np.random.RandomState(17).permutation(n);a=idx[:int(.8*n)];b=idx[int(.8*n):];values=load_components(args.out)
    previous=np.load(args.round3/'validation/ordinal_features.npz')
    for tau in [0,365]:assert np.allclose(values[f'jointw64_ratio1_pen50_t{tau}'],previous[f'book_joint50_{tau}'],atol=1e-6),'Equal-penalty control differs from round 3'
    # Component hyperparameters are screened using inner TRAIN labels only.
    # Keep one presence penalty for each width/ratio family, retain both dates.
    selected=[];screen=[]
    for width in [64,256]:
        for ratio in RATIOS:
            options=[]
            for pen in PENALTIES:
                name=config_name(width,ratio,pen);scores={str(tau):mse(y[a],values[name+f'_t{tau}'][a]) for tau in [0,365]};options.append((min(scores.values()),name));screen.append(dict(name=name,inner_training_MSE=scores))
            selected.append(min(options)[1])
    # The equal-penalty 64-book model and both support counts already exist in X.
    # Omit duplicate columns so the experiment adds new signals, not extra copies.
    screened=selected.copy();selected=[k for k in selected if k!='jointw64_ratio1_pen50']
    features=[name+f'_t{tau}' for name in selected for tau in [0,365]];G=np.column_stack([values[k] for k in features]);Z=np.c_[X,G].astype(np.float32)
    selection={'component_configs':selected,'screened_family_winners':screened,'component_features':features,'screening':screen,'input_manifest':manifest};sp=args.out/'selection.json'
    if sp.exists():assert json.loads(sp.read_text())==selection,'Changed component selection'
    sp.write_text(json.dumps(selection,indent=2))
    guard=args.out/'stack_manifest.json';expected={'input_manifest':manifest,'base_features':names,'component_features':features,'configs':{str(depth):config(depth) for depth in [6,8]},'inner_seed':17}
    if guard.exists():assert json.loads(guard.read_text())==expected,'Changed model configuration: use a fresh output directory'
    guard.write_text(json.dumps(expected,indent=2))
    controls=sorted([(k,mse(y[:n],v[:n])) for k,v in values.items() if not k.endswith('support')],key=lambda r:r[1]);print('COMPONENTS',controls[:8],flush=True)
    oldinner=[]
    for depth in [6,8]:
        m=CatBoostRegressor();m.load_model(str(args.round3/f'cat_{depth}_res0_inner.cbm'));oldinner.append(clip(m.predict(X[b])))
    reference_inner=.5*(oldinner[0]+oldinner[1]);inner={'reference':reference_inner};runs={};search=[]
    for depth in [6,8]:
        key=f'depth{depth}';path=args.out/(key+'_inner.cbm');cfg=config(depth)
        if path.exists():m=CatBoostRegressor();m.load_model(str(path))
        else:m=CatBoostRegressor(**cfg);m.fit(Z[a],y[a],eval_set=(Z[b],y[b]),early_stopping_rounds=120);m.save_model(str(path))
        inner[key]=clip(m.predict(Z[b]));runs[key]={'depth':depth,'iterations':m.tree_count_};search.append({'name':key,'inner_MSE':mse(y[b],inner[key]),**runs[key]});print('INNER',search[-1],flush=True)
    # Choose the new-depth mixture inside validation, then reference shrinkage.
    choices=[]
    for w in [0,.25,.5,.75,1]:
        pn=w*inner['depth6']+(1-w)*inner['depth8']
        for newweight in [0,.5,.75,1]:choices.append((mse(y[b],newweight*pn+(1-newweight)*reference_inner),w,newweight))
    score,depth6weight,newweight=min(choices);weights={k:v for k,v in {'depth6':depth6weight,'depth8':1-depth6weight}.items() if v>0};ref=np.load(args.round3/'evaluation.npz');assert np.array_equal(ref['y'],y[n:]);reference=ref['new'];evals={}
    for key in weights:
        r=runs[key];m=CatBoostRegressor(**config(r['depth'],r['iterations']));m.fit(Z[:n],y[:n]);m.save_model(str(args.out/(key+'_evaluation.cbm')));evals[key]=clip(m.predict(Z[n:]))
    raw=sum(weights[k]*v for k,v in evals.items());primary=newweight*raw+(1-newweight)*reference;conservative=.5*primary+.5*reference
    summary={'selected':{'depth_weights':weights,'new_weight':newweight,'inner_MSE':score},'reference_inner_MSE':mse(y[b],reference_inner),'search':search,'runs':runs,'component_selection':selection,'component_selection_rows':len(a),'component_selection_results':controls,'base_features':names,'reference_evaluation_MSE':mse(y[n:],reference),'raw_evaluation_MSE':mse(y[n:],raw),'primary_evaluation_MSE':mse(y[n:],primary),'conservative_evaluation_MSE':mse(y[n:],conservative),'input_manifest':manifest,'meta_selection_rows':n,'final_meta_training_rows':len(y),'reference_public_MSE':.6825445785868727}
    np.savez_compressed(args.out/'evaluation.npz',y=y[n:],reference=reference,raw=raw,primary=primary,conservative=conservative,**evals);(args.out/'results.json').write_text(json.dumps(summary,indent=2));print('LOCAL',summary['primary_evaluation_MSE'],'reference',summary['reference_evaluation_MSE'],flush=True)
    if not args.predict:return
    worker(args,True);FX,_,fn=extended_design(args.data_dir,args.legacy,args.prior,args.round2,args.round3,True);assert names==fn;v=load_components(args.out,True);FZ=np.c_[FX,np.column_stack([v[k] for k in features])].astype(np.float32);preds={}
    for key in weights:
        r=runs[key];m=CatBoostRegressor(**config(r['depth'],r['iterations']));m.fit(Z,y);m.save_model(str(args.out/(key+'_full.cbm')));preds[key]=clip(m.predict(FZ))
    raw=clip(sum(weights[k]*v for k,v in preds.items()));reference=np.load(args.round3/'test_predictions.npz')['primary'];primary=clip(newweight*raw+(1-newweight)*reference);conservative=clip(.5*primary+.5*reference);test=pd.read_csv(args.data_dir/'task_3_test.csv')
    for label,pred in [('primary',primary),('conservative',conservative),('raw',raw)]:
        assert len(pred)==20000 and np.isfinite(pred).all();folder=args.out/label;folder.mkdir(exist_ok=True);frame=test.copy();frame['rating']=pred;frame.to_csv(folder/'task_3_predictions.csv',index=False)
    np.savez_compressed(args.out/'test_predictions.npz',primary=primary,conservative=conservative,raw=raw,reference=reference,**preds)
if __name__=='__main__':main()
