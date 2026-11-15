"""Check joint FPMC gradients, cached cohort integrity and exported Task 5 CSVs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,ctypes,json,hashlib,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from task5_improve import data,graph
from task4_stacking import margin

def numeric(out):
    out.mkdir(parents=True,exist_ok=True);so=out/'test.so';subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_fpmc_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.train_fpmc.argtypes=[ctypes.c_int]*2+[ip]*6+[ctypes.c_int,ip,ctypes.c_int]+[fp]*5+[ctypes.c_int]*3+[ctypes.c_float]*3+[ctypes.c_int];rng=np.random.RandomState(17)
    for uniform in [0,1]:
        P=rng.normal(0,.1,(1,3)).astype(np.float32);Q=rng.normal(0,.1,(3,3)).astype(np.float32);A=rng.normal(0,.1,(3,3)).astype(np.float32);B=rng.normal(0,.1,(3,3)).astype(np.float32);bias=rng.normal(0,.1,3).astype(np.float32);params=[v.copy() for v in [P,Q,A,B,bias]];ep,eq,ea,eb,ebi=[v.copy() for v in params];diff=bias[1]-bias[2]+P[0]@(Q[1]-Q[2])+A[0]@(B[1]-B[2]);g=1/(1+np.exp(diff));lr=.02;reg=.03;ep[0]+=lr*(g*(Q[1]-Q[2])-reg*P[0]);ea[0]+=lr*(g*(B[1]-B[2])-reg*A[0]);eq[1]+=lr*(g*P[0]-reg*Q[1]);eq[2]+=lr*(-g*P[0]-reg*Q[2]);eb[1]+=lr*(g*A[0]-reg*B[1]);eb[2]+=lr*(-g*A[0]-reg*B[2]);ebi[1]+=lr*(g-reg*bias[1]);ebi[2]+=lr*(-g-reg*bias[2]);arr=lambda x:np.array(x,np.int32)
        lib.train_fpmc(1,3,arr([0]),arr([0]),arr([1]),arr([0,2]),arr([0,1]),arr([0,1,2]),3,arr([0,1,2]),3,P,Q,A,B,bias,0,1,42,lr,reg,0.,uniform)
        assert all(np.allclose(x,y,atol=1e-7) for x,y in zip([P,Q,A,B,bias],[ep,eq,ea,eb,ebi]))
    print('PASS: joint FPMC BPR gradient, both event/user sampling modes.',flush=True)

def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','baseline','out']:ap.add_argument('--'+key,type=Path)
    ap.add_argument('--numerical-only',action='store_true');a=ap.parse_args();numeric(a.out/'checks')
    if a.numerical_only:return
    from task5_improve_stack import load
    train,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline);X=graph(train,U,I);N=len(contexts);positive=qi[:N];assert np.all(np.asarray(X[np.tile(np.arange(U),2),positive]).ravel()==0)
    # Final-event contexts must end at the penultimate event, never at the target.
    for row in range(U):assert positive[U+row] not in contexts[U+row] and positive[row] not in contexts[row]
    history,contexts,qu,qi,_,_,test=data(a.data_dir,a.baseline,True);F=load(a,True);d,s,od,oscore,names=F;summary=json.loads((a.out/'results.json').read_text());assert names==summary['new_features'];preds={}
    old=CatBoostClassifier();old.load_model(str(a.baseline/'stacker.cbm'));ref=margin(old,od,oscore);previous=pd.read_csv(a.baseline/'task_5_predictions.csv');assert np.array_equal(np.where(ref>=0,test.item1,test.item2),previous.which)
    for key in summary['selected']['weights']:
        model=CatBoostClassifier();model.load_model(str(a.out/(key+'_full.cbm')));preds[key]=margin(model,d,s);sample=np.random.RandomState(17).choice(len(d),1024,replace=False);assert np.allclose(margin(model,-d[sample],s[sample]),-preds[key][sample],atol=1e-12)
    raw=sum(summary['selected']['weights'][k]*v for k,v in preds.items());blend=.5*raw+.5*ref;audit={}
    for label,score in [('primary',raw),('blend50',blend)]:
        path=a.out/label/'task_5_predictions.csv';frame=pd.read_csv(path);assert len(frame)==len(test)==86786;pd.testing.assert_frame_equal(frame.drop(columns='which'),test.drop(columns='which'));assert ((frame.which==frame.item1)|(frame.which==frame.item2)).all();assert np.array_equal(frame.which,np.where(score>=0,test.item1,test.item2));audit[label]={'rows':len(frame),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'input_columns_preserved':True,'model_replay':True,'changed_from_accepted':int(np.sum(frame.which!=previous.which))}
    # Independently recalculate full-data final-checkpoint FPMC component scores.
    z={}
    for kind in ['fpmc','fpmc_uniform']:
        f=np.load(a.out/'full'/(kind+'_features.npz'));z.update({k:f[k] for k in f.files})
    N=len(contexts);ctx=np.tile(np.arange(N),2);sample=np.random.RandomState(31).choice(np.flatnonzero(qi>=0),64,replace=False)
    for name in ['fpmc32_uniform','fpmc64_mix','fpmc64_useruniform']:
        m=dict(np.load(a.out/'full'/(name+'_model.npz')));assert int(m['epoch'])==100
        for row in sample:
            user,j=qu[row],qi[row];h=contexts[ctx[row]];known=m['pop'][j]>0;p=float(m['P'][user]@m['Q'][j]) if known else 0.;last=float(m['A'][h[-1]]@m['B'][j]) if known and m['source_count'][h[-1]]>0 else 0.;hh=h[-3:][::-1];w=.7**np.arange(len(hh));w*=m['source_count'][hh]>0;v=w@m['A'][hh]/max(w.sum(),1e-8);recent=float(v@m['B'][j]) if known else 0.;bias=float(m['bias'][j]) if known else 0.
            for key,value in [('personal',p),('last',last),('recent3',recent),('bias',bias),('joint',p+last+bias)]:assert np.allclose(z[name+'_e100_'+key][row],value,atol=2e-5)
    # Independent reader-intersection reconstruction of weighted item similarities.
    X=graph(history,U,I);pop=np.asarray(X.sum(0)).ravel();length=np.array(list(map(len,history)));down=(1/(1+length/length.mean())).astype(np.float32);weighted=X.multiply(down[:,None]).tocsr();wp=np.asarray(weighted.sum(0)).ravel();knn=dict(np.load(a.out/'full/neighbor_features.npz'))
    for row in sample[:12]:
        j=qi[row];h=contexts[ctx[row]];readers=X[:,j].nonzero()[0];shared=np.asarray(X[readers][:,h].T@down[readers]).ravel();shared[h==j]=0
        for alpha in [.25,.5,.75]:
            sim=shared/(max(wp[j],1e-8)**(1-alpha)*np.maximum(wp[h],1e-8)**alpha)
            for shrink in [0,5]:
                val=sim if shrink==0 else sim*shared/(shared+shrink);decay=np.power(.9,np.arange(len(h)-1,-1,-1));expected={'max':val.max(),'mean':val.mean(),'last':val[-1],'recent3':val[-3:].mean(),'recent10':val[-10:].mean(),'decay':val@decay/decay.sum()}
                for key,value in expected.items():assert np.allclose(knn[f'weighted_knn_a{alpha}_s{shrink}_{key}'][row],value,atol=2e-5)
    # Independently retrieve session neighbors on rows without top-k boundary ties.
    idf=np.log1p(U/np.maximum(pop,1)).astype(np.float32);D=X.multiply(idf).tocsr();norm=np.sqrt(np.asarray(D.multiply(D).sum(1)).ravel());checked=0
    for row in range(min(N,256)):
        h=contexts[row][-20:];w=((.9**np.arange(len(h))[::-1])*idf[h]).astype(np.float32);order=np.argsort(h);sim=np.asarray(X[:,h[order]]@(w[order]*idf[h[order]])).ravel()/np.maximum(np.sqrt(np.sum(w*w))*norm,1e-8);sim[qu[row]]=0;nb=np.argsort(-sim,kind='stable')[:201]
        if sim[nb[199]]<=0 or sim[nb[199]]-sim[nb[200]]<1e-7 or sim[nb[49]]-sim[nb[50]]<1e-7:continue
        nb=nb[:200]
        for target in [row,N+row]:
            present=[];tail=[];forward=[]
            for user in nb:
                pos=np.flatnonzero(history[user]==qi[target]);anchor=np.flatnonzero(history[user]==contexts[row][-1]);present.append(float(len(pos)>0));tail.append(float(np.exp(-(len(history[user])-1-pos[0])/5)) if len(pos) else 0.);delta=int(pos[0]-anchor[0]) if len(pos) and len(anchor) else 0;forward.append(1/delta if 1<=delta<=5 else 0.)
            for k in [50,200]:
                for key,val in [('all',present),('tail5',tail),('forward5',forward)]:assert np.allclose(knn[f'sessionnn_{k}_{key}'][target],sim[nb[:k]]@np.asarray(val)[:k]/(sim[nb[:k]].sum()+1e-8),atol=2e-5)
        checked+=1
        if checked==8:break
    assert checked>=2,'Not enough unambiguous session-neighbor audit rows'
    inputs={f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['task_5_train.csv','task_5_test.csv']};assert inputs==summary['input_manifest']
    for stage,full in [('validation',False),('full',True)]:assert json.loads((a.out/stage/'manifest.json').read_text())=={'full':full,'inputs':inputs}
    summary['audit']=audit;summary['checks']={'FPMC_gradient':True,'FPMC_parameter_replay':True,'weighted_item_neighbor_replay':True,'independent_session_neighbor_replay':True,'held_out_targets_absent':True,'chronological_contexts':True,'cached_pair_ids_match':True,'accepted_prediction_reproduced':True,'pair_swap_antisymmetry':True,'input_hashes_verified':True};summary['source_sha256']={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['task5_improve.py','task5_fpmc_kernel.cpp','task5_improve_stack.py','task5_improve_checks.py']};(a.out/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'audit':audit,'checks':summary['checks']},indent=2))
if __name__=='__main__':main()
