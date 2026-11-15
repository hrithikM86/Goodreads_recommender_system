"""Add genuine time-aware FM predictions to the accepted Task 3 ensemble."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from task3_fm import CONFIGS,library
from task3_round3 import extended_design,dataset,clip,mse

GROUPS=[(0,[0,1]),(2,[2,3,4]),(5,[5,6,7]),(8,[8,9])]
def config(depth,iterations=1800):return dict(iterations=iterations,depth=depth,learning_rate=.025,l2_leaf_reg=20,random_seed=42,thread_count=2,loss_function='RMSE',allow_writing_files=False,verbose=250)
def values(out,full=False):
    result={}
    for part in ([out/'full'] if full else [out/f'validation{k}' for k,_ in GROUPS]):
        for f in sorted(part.glob('*_features.npz')):
            z=np.load(f);result.update({k:z[k] for k in z.files})
    return result
def workers(args,full=False):
    if full:
        part=args.out/'full';part.mkdir(exist_ok=True);library(part);groups=[(0,list(range(5))),(5,list(range(5,10)))]
    else:groups=GROUPS
    jobs=[]
    for index,which in groups:
        part=args.out/('full' if full else f'validation{index}')
        cmd=[sys.executable,str(Path(__file__).with_name('task3_fm.py')),'--data-dir',str(args.data_dir),'--prior',str(args.prior),'--round2',str(args.round2),'--out',str(part),'--which']+list(map(str,which))
        if full:cmd+=['--full','--selection',str(args.out/'selection.json')]
        handle=open(args.out/(f'worker{index}_'+('full' if full else 'validation')+'.log'),'a');jobs.append((subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT),handle))
    for proc,handle in jobs:
        rc=proc.wait();handle.close();assert rc==0,'FM worker failed; inspect logs'

def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','legacy','prior','round2','round3','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--predict',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);workers(a)
    X,anchor,names=extended_design(a.data_dir,a.legacy,a.prior,a.round2,a.round3);p,tr,q,*_=dataset(a.data_dir,False);y=q.rating.to_numpy();n=int(.95*len(p))-int(.9*len(p));idx=np.random.RandomState(17).permutation(n);train=idx[:int(.8*n)];tune=idx[int(.8*n):];v=values(a.out);screen=[];features=[]
    for cfg in CONFIGS:
        options=[(mse(y[train],pred[train]),name) for name,pred in v.items() if name.startswith(cfg['name']+'_e')];score,name=min(options);features.append(name);screen.append({'configuration':cfg,'checkpoint':name,'inner_training_MSE':score,'checkpoint_scores':{k:s for s,k in options}})
    selection={'fm_features':features,'screening':screen};sp=a.out/'selection.json'
    if sp.exists():assert json.loads(sp.read_text())==selection
    sp.write_text(json.dumps(selection,indent=2));G=np.column_stack([v[k] for k in features]);Z=np.c_[X,G].astype(np.float32);manifest={f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['tasks_3_4_train.csv','task_3_test.csv']};assert json.loads((a.round3/'stack_manifest.json').read_text())['input_manifest']==manifest
    expected={'inputs':manifest,'features':names+features,'configs':{str(d):config(d) for d in [6,8]},'inner_seed':17};guard=a.out/'stack_manifest.json'
    if guard.exists():assert json.loads(guard.read_text())==expected
    guard.write_text(json.dumps(expected,indent=2));old=[]
    for depth in [6,8]:
        m=CatBoostRegressor();m.load_model(str(a.round3/f'cat_{depth}_res0_inner.cbm'));old.append(clip(m.predict(X[tune])))
    reference_inner=.5*(old[0]+old[1]);inner={};runs={}
    direct=[]
    for name,pred in list(zip(features,G.T))+[('fm_average',G.mean(1))]:
        for weight in [0,.025,.05,.1,.15,.2]:direct.append((mse(y[tune],weight*pred[tune]+(1-weight)*reference_inner),name,weight))
    print('DIRECT_FM_BLEND',min(direct),flush=True)
    for depth in [6,8]:
        key=f'depth{depth}';path=a.out/(key+'_inner.cbm')
        if path.exists():m=CatBoostRegressor();m.load_model(str(path))
        else:m=CatBoostRegressor(**config(depth));m.fit(Z[train],y[train],eval_set=(Z[tune],y[tune]),early_stopping_rounds=120);m.save_model(str(path))
        inner[key]=clip(m.predict(Z[tune]));runs[key]={'depth':depth,'iterations':m.tree_count_,'inner_MSE':mse(y[tune],inner[key])};print('INNER',runs[key],flush=True)
    score,w=min((mse(y[tune],w*inner['depth6']+(1-w)*inner['depth8']),w) for w in [0,.25,.5,.75,1]);weights={k:v for k,v in {'depth6':w,'depth8':1-w}.items() if v>0};pi=sum(weights[k]*inner[k] for k in weights);selectedscore,shrink=min((mse(y[tune],s*pi+(1-s)*reference_inner),s) for s in [0,.1,.25,.5,.75,1]);evals={}
    for key in weights:
        r=runs[key];m=CatBoostRegressor(**config(r['depth'],r['iterations']));m.fit(Z[:n],y[:n]);m.save_model(str(a.out/(key+'_evaluation.cbm')));evals[key]=clip(m.predict(Z[n:]))
    ref=np.load(a.round3/'evaluation.npz');assert np.array_equal(ref['y'],y[n:]);reference=ref['new'];augmented=clip(sum(weights[k]*v for k,v in evals.items()));blend=clip(.5*augmented+.5*reference);selected=clip(shrink*augmented+(1-shrink)*reference)
    summary={'selection':selection,'selected':{'depth_weights':weights,'new_weight':shrink,'raw_inner_MSE':score,'selected_inner_MSE':selectedscore},'reference_inner_MSE':mse(y[tune],reference_inner),'runs':runs,'direct_FM_blend_inner':min(direct),'reference_evaluation_MSE':mse(y[n:],reference),'augmented_evaluation_MSE':mse(y[n:],augmented),'blend50_evaluation_MSE':mse(y[n:],blend),'selected_evaluation_MSE':mse(y[n:],selected),'FM_selection_MSE':{k:mse(y[:n],v[k][:n]) for k in features},'base_features':names,'input_manifest':manifest,'reference_public_MSE':.6825445785868727,'meta_selection_rows':n,'final_meta_training_rows':len(y)}
    np.savez_compressed(a.out/'evaluation.npz',y=y[n:],reference=reference,augmented=augmented,blend50=blend,selected=selected,**evals);(a.out/'results.json').write_text(json.dumps(summary,indent=2));print('LOCAL',summary['augmented_evaluation_MSE'],summary['blend50_evaluation_MSE'],'reference',summary['reference_evaluation_MSE'],flush=True)
    if not a.predict:return
    workers(a,True);FX,_,fn=extended_design(a.data_dir,a.legacy,a.prior,a.round2,a.round3,True);assert fn==names;fv=values(a.out,True);FZ=np.c_[FX,np.column_stack([fv[k] for k in features])].astype(np.float32);preds={}
    for key in weights:
        r=runs[key];m=CatBoostRegressor(**config(r['depth'],r['iterations']));m.fit(Z,y);m.save_model(str(a.out/(key+'_full.cbm')));preds[key]=clip(m.predict(FZ))
    reference=np.load(a.round3/'test_predictions.npz')['primary'];augmented=clip(sum(weights[k]*v for k,v in preds.items()));blend50=clip(.5*augmented+.5*reference);selected=clip(shrink*augmented+(1-shrink)*reference);test=pd.read_csv(a.data_dir/'task_3_test.csv')
    for label,pred in [('augmented',augmented),('blend50',blend50)]:
        assert len(pred)==20000 and np.isfinite(pred).all();folder=a.out/label;folder.mkdir(exist_ok=True);frame=test.copy();frame['rating']=pred;frame.to_csv(folder/'task_3_predictions.csv',index=False)
    np.savez_compressed(a.out/'test_predictions.npz',augmented=augmented,blend50=blend50,reference=reference,selected=selected,**preds)
if __name__=='__main__':main()
