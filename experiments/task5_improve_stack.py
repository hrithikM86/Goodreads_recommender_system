"""Train-only sequence ensemble selection and chronological evaluation for Task 5."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,hashlib,subprocess,sys
import numpy as np
from catboost import CatBoostClassifier
from task5_improve import data
from task4_stacking import margin,design

def acc(x):return float(np.mean((x>0)+.5*(x==0)))
def cfg(depth,iterations=1000):return dict(iterations=iterations,depth=depth,learning_rate=.04,l2_leaf_reg=15,loss_function='Logloss',eval_metric='Accuracy',random_seed=42,thread_count=2,verbose=200,allow_writing_files=False)
def load(args,full=False):
    old=np.load(args.baseline/('test_pair_features.npz' if full else 'pair_features.npz'))['scores'];part=args.out/('full' if full else 'validation');names=[];cols=[]
    for kind in ['fpmc','fpmc_uniform','neighbor']:
        z=np.load(part/(kind+'_features.npz'));keys=z.files
        if kind=='fpmc':
            # Fixed schema, independent of whether checkpoints were fitted in one
            # run or extended from an earlier completed experiment.
            models=['fpmc32_uniform','fpmc64_mix'];fields=['personal','last','recent3','bias','joint'];keys=[f'{m}_e{e}_{f}' for m in models for e in [15,40] for f in fields]+[f'{m}_e100_{f}' for m in models for f in fields];assert set(keys)==set(z.files)
        names+=keys;cols += [z[k] for k in keys]
    scores=np.c_[old,np.column_stack(cols)].astype(np.float32);N=len(scores)//2;return scores[:N]-scores[N:],scores[:N]+scores[N:],old[:N]-old[N:],old[:N]+old[N:],names
def workers(a,full=False):
    jobs=[]
    for kind in ['fpmc','uniform','neighbors']:
        cmd=[sys.executable,str(Path(__file__).with_name('task5_improve.py')),'--data-dir',str(a.data_dir),'--baseline',str(a.baseline),'--out',str(a.out),'--worker',kind]+(['--full'] if full else []);handle=open(a.out/(kind+('_full' if full else '_validation')+'.log'),'a');jobs.append((subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT),handle))
    for job,h in jobs:
        rc=job.wait();h.close();assert rc==0,'Worker failed; inspect logs'

def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','baseline','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--predict',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);workers(a);history,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline);d,s,od,oscore,names=load(a);inputs={f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['task_5_train.csv','task_5_test.csv']};guard=a.out/'stack_manifest.json';expected={'inputs':inputs,'features':names,'configs':{str(depth):cfg(depth) for depth in [5,7]},'inner_seed':17}
    if guard.exists():assert json.loads(guard.read_text())==expected
    guard.write_text(json.dumps(expected,indent=2));idx=np.random.RandomState(17).permutation(U);tr=idx[:int(.8*U)];va=idx[int(.8*U):];ref=CatBoostClassifier(iterations=364,depth=5,learning_rate=.04,l2_leaf_reg=10,loss_function='Logloss',random_seed=42,thread_count=2,verbose=False,allow_writing_files=False);ref.fit(np.r_[design(od[tr],oscore[tr]),design(-od[tr],oscore[tr])],np.r_[np.ones(len(tr)),np.zeros(len(tr))]);rp=margin(ref,od[va],oscore[va]);inner={};reports={}
    for depth in [5,7]:
        key=f'depth{depth}';path=a.out/(key+'_inner.cbm')
        if path.exists():m=CatBoostClassifier();m.load_model(str(path))
        else:
            m=CatBoostClassifier(**cfg(depth));m.fit(np.r_[design(d[tr],s[tr]),design(-d[tr],s[tr])],np.r_[np.ones(len(tr)),np.zeros(len(tr))],eval_set=(np.r_[design(d[va],s[va]),design(-d[va],s[va])],np.r_[np.ones(len(va)),np.zeros(len(va))]),early_stopping_rounds=100)
            # Tune the actual antisymmetric pair decision, not just row orientation.
            options=[]
            for count in sorted(set(list(range(50,m.tree_count_+1,50))+[m.tree_count_])):
                p=m.predict_proba(design(d[va],s[va]),ntree_end=count)[:,1]-m.predict_proba(design(-d[va],s[va]),ntree_end=count)[:,1];options.append((acc(p),count))
            score,count=max(options,key=lambda r:(r[0],-r[1]));m.shrink(count);m.save_model(str(path))
        inner[key]=margin(m,d[va],s[va]);reports[key]={'depth':depth,'iterations':m.tree_count_,'inner_accuracy':acc(inner[key])};print('INNER',reports[key],flush=True)
    score,w=max(((acc(w*inner['depth5']+(1-w)*inner['depth7']),w) for w in [0,.25,.5,.75,1]),key=lambda r:r[0]);weights={k:v for k,v in {'depth5':w,'depth7':1-w}.items() if v>0};mix=sum(weights[k]*inner[k] for k in weights);selected_score,refweight=max(((acc(v*mix+(1-v)*rp),v) for v in [0,.25,.5,.75,1]),key=lambda r:r[0]);evals={}
    for key in weights:
        r=reports[key];path=a.out/(key+'_evaluation.cbm')
        if path.exists():m=CatBoostClassifier();m.load_model(str(path));assert m.tree_count_==r['iterations']
        else:m=CatBoostClassifier(**cfg(r['depth'],r['iterations']));m.fit(np.r_[design(d[:U],s[:U]),design(-d[:U],s[:U])],np.r_[np.ones(U),np.zeros(U)]);m.save_model(str(path))
        evals[key]=margin(m,d[U:],s[U:])
    old=CatBoostClassifier();old.load_model(str(a.baseline/'stacker.cbm'));reference=margin(old,od[U:],oscore[U:]);raw=sum(weights[k]*v for k,v in evals.items());blend=.5*raw+.5*reference;summary={'selected':{'weights':weights,'inner_accuracy':score,'reference_weighted_accuracy':selected_score,'new_weight':refweight},'reference_inner_accuracy':acc(rp),'runs':reports,'reference_evaluation_accuracy':acc(reference),'new_evaluation_accuracy':acc(raw),'blend50_evaluation_accuracy':acc(blend),'new_component_selection_accuracy':{name:acc(d[:U,22+j]) for j,name in enumerate(names)},'new_features':names,'input_manifest':inputs,'users':U,'items':I,'reference_public_accuracy':.9093402161638974,'all_ratings_retained':True,'ordering_only':True}
    np.savez_compressed(a.out/'evaluation.npz',reference=reference,new=raw,blend50=blend,**evals);(a.out/'results.json').write_text(json.dumps(summary,indent=2));print('LOCAL',summary['new_evaluation_accuracy'],summary['blend50_evaluation_accuracy'],'reference',summary['reference_evaluation_accuracy'],flush=True)
    if not a.predict:return
    workers(a,True);fd,fs,fod,fos,fn=load(a,True);assert fn==names;history,contexts,qu,qi,_,_,test=data(a.data_dir,a.baseline,True);preds={}
    for key in weights:
        # The combiner still trains on exactly the same selection pairs. Reuse
        # that fitted model; only the global component fits/query features change.
        m=CatBoostClassifier();m.load_model(str(a.out/(key+'_evaluation.cbm')));m.save_model(str(a.out/(key+'_full.cbm')));preds[key]=margin(m,fd,fs)
    raw=sum(weights[k]*v for k,v in preds.items());reference=margin(old,fod,fos);blend=.5*raw+.5*reference
    for label,value in [('primary',raw),('blend50',blend)]:
        frame=test.copy();frame['which']=np.where(value>=0,frame.item1,frame.item2);assert ((frame.which==frame.item1)|(frame.which==frame.item2)).all();folder=a.out/label;folder.mkdir(exist_ok=True);frame.to_csv(folder/'task_5_predictions.csv',index=False)
    np.savez_compressed(a.out/'test_margins.npz',primary=raw,reference=reference,blend50=blend,**preds)
if __name__=='__main__':main()
