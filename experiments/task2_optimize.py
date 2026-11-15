"""Task 2 paired zero-rating ablation with sparse linear and factor ensembles.

Run with PYTHONPATH pointing to an installed implicit package if necessary.
EASE is restricted to the most frequent 2,000 items (dense inversion otherwise
requires quadratic memory in all 51k items). SLIM uses 32 candidate neighbors
per target, nonnegative elastic-net regression, and no self edge.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import argparse,json,time,ctypes,subprocess,hashlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz, load_npz
from scipy.linalg import cho_factor,cho_solve
from implicit.bpr import BayesianPersonalizedRanking
from implicit.als import AlternatingLeastSquares
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression


def log(*args): print(*args,flush=True)
def accuracy(m): return float(np.mean((m>0)+.5*(m==0)))
def matrix(u,i,mask,shape):
    x=coo_matrix((np.ones(np.sum(mask),np.float32),(u[mask],i[mask])),shape=shape).tocsr();x.data[:]=1
    return x


def factor_scores(X,qu,qi,out):
    specs=[('bpr64',BayesianPersonalizedRanking,dict(factors=64,iterations=200,regularization=.01,learning_rate=.03,random_state=17,num_threads=1)),
           ('bpr128',BayesianPersonalizedRanking,dict(factors=128,iterations=350,regularization=.03,learning_rate=.03,random_state=42,num_threads=1)),
           ('bpr96',BayesianPersonalizedRanking,dict(factors=96,iterations=300,regularization=.003,learning_rate=.05,random_state=23,num_threads=1)),
           ('als128',AlternatingLeastSquares,dict(factors=128,iterations=30,regularization=.1,alpha=40,random_state=42,num_threads=2))]
    scores={}
    for name,cls,kw in specs:
        path=out/(name+'.npz')
        if path.exists():
            z=np.load(path);uf,iv=z['user'],z['item']
        else:
            start=time.time();model=cls(**kw);model.fit(X,show_progress=False);uf,iv=model.user_factors,model.item_factors
            np.savez_compressed(path,user=uf,item=iv);log(name,'seconds',round(time.time()-start,1))
        scores[name]=np.einsum('ij,ij->i',uf[qu],iv[qi]); valid=(np.asarray(X.sum(1)).ravel()[qu]>0)&(np.asarray(X.sum(0)).ravel()[qi]>0);scores[name][~valid]=0;del uf,iv
    return scores


def ease_scores(X,qu,qi,out):
    pop=np.asarray(X.sum(0)).ravel();top=np.argsort(pop)[-min(2000,X.shape[1]):];Z=X[:,top].astype(np.float64)
    G=(Z.T@Z).toarray();mapping=np.full(X.shape[1],-1);mapping[top]=np.arange(len(top));q=mapping[qi]
    scores={}
    for reg in [50.,300.,1000.]:
        cache=out/('ease_'+str(int(reg))+'.npy')
        if cache.exists(): B=np.load(cache)
        else:
            A=G.copy();A.flat[::len(top)+1]+=reg
            P=cho_solve(cho_factor(A),np.eye(len(top)));B=-P/np.diag(P)[None,:];np.fill_diagonal(B,0)
            np.save(cache,B.astype(np.float32));del A,P
        s=np.zeros(len(qu),np.float32)
        valid=np.flatnonzero(q>=0)
        for ix in np.array_split(valid,max(1,(len(valid)+1999)//2000)):
            if len(ix):s[ix]=np.asarray(Z[qu[ix]].multiply(B[:,q[ix]].T).sum(1)).ravel()
        scores['ease'+str(int(reg))]=s
    scores['ease_covered']=(q>=0).astype(np.float32)
    return scores


def linear_neighbors(X,qu,qi,out):
    """Fit restricted SLIM with 32 neighbors using coordinate descent.

    For each item j minimize .5||X_j-X_N b||² + 10||b||²/2
    + l1||b||_1 subject to b>=0, N excludes j. l1=.2 and 2.
    Convergence monitored; max 100 sweeps. Scores additionally include
    cosine and Jaccard summaries for direct comparison to the old ensemble.
    """
    cache=out/'linear_scores.npz'
    if cache.exists():return dict(np.load(cache))
    source=Path(__file__).with_name('task2_slim_kernel.cpp');libpath=out/'slim_kernel.so'
    if not libpath.exists():subprocess.run(['g++','-O3','-shared','-fPIC',str(source),'-o',str(libpath)],check=True)
    lib=ctypes.CDLL(str(libpath.resolve()));kernel=lib.slim_fit
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');kernel.argtypes=[ctypes.c_int,ptr,ptr,ptr];kernel.restype=ctypes.c_double
    pop=np.asarray(X.sum(0)).ravel();Xt=X.T.tocsr()
    names=['cosine_max','cosine_top10','jaccard_max','slim02','slim2']
    scores={n:np.zeros(len(qu),np.float32) for n in names}
    order=np.argsort(qi,kind='stable');items,starts=np.unique(qi[order],return_index=True);ends=np.r_[starts[1:],len(order)]
    slim_rows=[[],[]];slim_cols=[[],[]];slim_data=[[],[]]
    max_change=0.;nonconverged=0
    for offset in range(0,len(items),64):
        keys=items[offset:offset+64];co=(Xt[keys]@X).toarray()
        for k,j in enumerate(keys):
            p=offset+k;ix=order[starts[p]:ends[p]];c=co[k];c[j]=0
            cos=c/np.sqrt(np.maximum(pop[j]*pop,1));jac=c/np.maximum(pop[j]+pop-c,1)
            H=X[qu[ix]]
            for r,query in enumerate(ix):
                hist=H.indices[H.indptr[r]:H.indptr[r+1]];v=cos[hist]
                if len(v):
                    scores['cosine_max'][query]=v.max();scores['cosine_top10'][query]=np.sort(v)[-10:].sum();scores['jaccard_max'][query]=jac[hist].max()
            n=min(32,np.count_nonzero(c))
            if not n:continue
            nb=np.argpartition(cos,-n)[-n:];nb=nb[np.argsort(cos[nb])[::-1]]
            gram=np.ascontiguousarray((Xt[nb]@Xt[nb].T).toarray(),dtype=np.float64)
            rhs=np.ascontiguousarray(c[nb],dtype=np.float64);B=np.zeros((n,2))
            change=kernel(n,gram,rhs,B)
            max_change=max(max_change,change);nonconverged+=change>=1e-5
            hs=H[:,nb]
            for v,name in enumerate(['slim02','slim2']):
                scores[name][ix]=np.asarray(hs@B[:,v]).ravel();nz=B[:,v]>0
                slim_rows[v].extend(nb[nz]);slim_cols[v].extend([j]*int(nz.sum()));slim_data[v].extend(B[nz,v])
        if offset%2048==0:log('Neighborhood/SLIM',offset,'/',len(items))
    for v,name in enumerate(['slim02','slim2']):
        W=coo_matrix((slim_data[v],(slim_rows[v],slim_cols[v])),shape=(X.shape[1],X.shape[1])).tocsr()
        assert np.count_nonzero(W.diagonal())==0;save_npz(out/(name+'.npz'),W)
    log('SLIM max final coordinate change',max_change,'not converged',nonconverged)
    (out/'slim_convergence.json').write_text(json.dumps(dict(max_final_change=max_change,nonconverged=int(nonconverged),items=len(items))))
    np.savez_compressed(cache,**scores)
    return scores


def features(X,qu,qi,out):
    cache=out/'features.npz'
    if cache.exists():
        z=np.load(cache);names=list(z.files);f=np.column_stack([z[n] for n in names]);valid=(np.asarray(X.sum(1)).ravel()[qu]>0)&(np.asarray(X.sum(0)).ravel()[qi]>0)
        for k,n in enumerate(names):
            if n.startswith(('bpr','als')):f[~valid,k]=0
        if not valid.all():np.savez_compressed(cache,**{n:f[:,k] for k,n in enumerate(names)})
        return names,f
    pop=np.asarray(X.sum(0)).ravel();up=np.asarray(X.sum(1)).ravel()
    scores=dict(popularity=np.log1p(pop[qi]),already_seen=np.asarray(X[qu,qi]).ravel().astype(np.float32),user_count=np.log1p(up[qu]))
    scores.update(factor_scores(X,qu,qi,out));scores.update(ease_scores(X,qu,qi,out));scores.update(linear_neighbors(X,qu,qi,out))
    assert all(np.isfinite(x).all() for x in scores.values());np.savez_compressed(cache,**scores)
    return list(scores),np.column_stack(list(scores.values()))


def design(d,s):return np.column_stack([d,s])
def predict(m,d,s):
    return m.predict_proba(design(d,s))[:,1]-m.predict_proba(design(-d,s))[:,1]

def blend(f,names,sel,ev,out):
    n=len(f)//2;a,b=f[:n],f[n:];d=a-b;s=a+b
    rng=np.random.RandomState(17);idx=rng.permutation(np.flatnonzero(sel));cut=int(.8*len(idx));tr,va=idx[:cut],idx[cut:]
    reports={};candidates=[]
    # All tuning, including linear vs nonlinear blend weight, stays inside selection split.
    for group,cols in [('old_style',[k for k,n in enumerate(names) if n in ['popularity','already_seen','bpr64','als128','cosine_max','cosine_top10','jaccard_max']]),('expanded',list(range(len(names))))]:
        for C in [.01,.1,1.]:
            scale=np.std(d[tr][:,cols],axis=0);scale[scale<1e-8]=1
            m=LogisticRegression(C=C,fit_intercept=False,max_iter=1000)
            x=d[tr][:,cols]/scale;m.fit(np.r_[x,-x],np.r_[np.ones(len(tr)),np.zeros(len(tr))])
            w=np.zeros(d.shape[1]);w[cols]=m.coef_[0]/scale;score=accuracy(d[va]@w)
            candidates.append((score,group,C,cols));reports[group+'_C'+str(C)]=score
    _,group,C,cols=max(candidates,key=lambda x:x[0]);scale=np.std(d[sel][:,cols],axis=0);scale[scale<1e-8]=1
    m=LogisticRegression(C=C,fit_intercept=False,max_iter=1000);x=d[sel][:,cols]/scale
    m.fit(np.r_[x,-x],np.r_[np.ones(len(x)),np.zeros(len(x))]);w=np.zeros(d.shape[1]);w[cols]=m.coef_[0]/scale
    # Refit linear on inner training to select blend; never use evaluation labels.
    scale0=np.std(d[tr][:,cols],axis=0);scale0[scale0<1e-8]=1
    m0=LogisticRegression(C=C,fit_intercept=False,max_iter=1000);x=d[tr][:,cols]/scale0;m0.fit(np.r_[x,-x],np.r_[np.ones(len(x)),np.zeros(len(x))]);lv=(d[va][:,cols]/scale0)@m0.coef_[0]
    lv=1/(1+np.exp(-np.clip(lv,-50,50)))-.5
    best=(accuracy(lv),0,None,None);reports['selected_linear']=dict(group=group,C=C,inner_accuracy=best[0])
    for depth in [3,5,7]:
        model=CatBoostClassifier(iterations=1200,depth=depth,learning_rate=.035,l2_leaf_reg=10,loss_function='Logloss',eval_metric='Accuracy',random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
        model.fit(np.r_[design(d[tr],s[tr]),design(-d[tr],s[tr])],np.r_[np.ones(len(tr)),np.zeros(len(tr))],eval_set=(np.r_[design(d[va],s[va]),design(-d[va],s[va])],np.r_[np.ones(len(va)),np.zeros(len(va))]),early_stopping_rounds=100)
        nv=predict(model,d[va],s[va])/2;iters=model.get_best_iteration()+1
        for alpha in [0.25,.5,.75,1.]:
            score=accuracy(alpha*nv+(1-alpha)*lv);key=f'cat_depth{depth}_mix{alpha}'
            reports[key]=dict(inner_accuracy=score,iterations=iters)
            if score>best[0]:best=(score,alpha,depth,iters)
        log('Nonlinear depth',depth,'iteration',iters,'inner accuracy',accuracy(nv))
    inner,alpha,depth,iters=best;model=None
    if alpha:
        model=CatBoostClassifier(iterations=iters,depth=depth,learning_rate=.035,l2_leaf_reg=10,loss_function='Logloss',random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
        model.fit(np.r_[design(d[sel],s[sel]),design(-d[sel],s[sel])],np.r_[np.ones(sel.sum()),np.zeros(sel.sum())]);model.save_model(str(out/'stacker.cbm'))
    linear=1/(1+np.exp(-np.clip(d@w,-50,50)))-.5;nonlinear=predict(model,d,s)/2 if model else np.zeros(len(d));margin=(1-alpha)*linear+alpha*nonlinear
    report=dict(feature_names=names,linear_weights=w.tolist(),selected=dict(inner_accuracy=inner,nonlinear_weight=alpha,depth=depth,iterations=iters),search=reports,components={name:accuracy(d[ev,k]) for k,name in enumerate(names)},evaluation_linear=accuracy(linear[ev]),evaluation_nonlinear=accuracy(nonlinear[ev]) if model else None,evaluation_ensemble=accuracy(margin[ev]))
    (out/'results.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'combiner.npz',weights=w,alpha=alpha);np.savez_compressed(out/'validation_margins.npz',margin=margin,selection=sel,evaluation=ev)
    log('SELECTED',report['selected'],'EVALUATION',report['evaluation_ensemble']);return w,alpha,model,report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--variant',choices=['with_zero','without_zero'],required=True);ap.add_argument('--predict',action='store_true');args=ap.parse_args()
    out=args.output_dir/args.variant;out.mkdir(parents=True,exist_ok=True)
    manifest={p:hashlib.sha256((args.data_dir/p).read_bytes()).hexdigest() for p in ['tasks_1_2_train.csv','task_2_test.csv']}
    manifest.update(variant=args.variant,protocol='task2_optimize_v1')
    path=out/'input_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Inputs differ: use a fresh output directory')
    path.write_text(json.dumps(manifest,indent=2))
    df=pd.read_csv(args.data_dir/'tasks_1_2_train.csv');u,users=pd.factorize(df.user);i,items=pd.factorize(df.item);shape=(len(users),len(items));nz=df.rating.to_numpy()>0
    order=np.random.RandomState(42).permutation(len(u));a=int(.9*len(u));b=int(.95*len(u));idx=order[:a];hold=order[a:];train=np.zeros(len(u),bool);train[idx]=True
    # Same positive holdout IDs for both variants; filter positives only for no-zero version.
    allowed=np.ones(len(u),bool) if args.variant=='with_zero' else nz
    train&=allowed;X=matrix(u,i,train,shape);rng=np.random.RandomState(43);eligible=np.flatnonzero(train);neg=i[eligible[rng.randint(len(eligible),size=len(hold))]]
    same=neg==i[hold]
    while same.any():neg[same]=i[eligible[rng.randint(len(eligible),size=same.sum())]];same=neg==i[hold]
    qu=np.r_[u[hold],u[hold]];qi=np.r_[i[hold],neg];part=out/'validation';part.mkdir(exist_ok=True)
    names,f=features(X,qu,qi,part);sel=np.arange(len(hold))<b-a;ev=~sel;sel&=allowed[hold];ev&=allowed[hold]
    w,alpha,model,report=blend(f,names,sel,ev,part)
    z=np.load(part/'validation_margins.npz');report.update(variant=args.variant,training_rows=int(train.sum()),selection_rows=int(sel.sum()),evaluation_rows=int(ev.sum()),evaluation_nonzero=accuracy(z['margin'][(np.arange(len(hold))>=b-a)&nz[hold]]),split_seed=42,negative_sampling='training-popularity weighted; seen allowed',zero_rows=int((~nz).sum()))
    (part/'results.json').write_text(json.dumps(report,indent=2))
    if not args.predict:return
    full=matrix(u,i,allowed,shape);test=pd.read_csv(args.data_dir/'task_2_test.csv');um=dict(zip(users,range(len(users))));im=dict(zip(items,range(len(items))))
    tu=np.array([um.get(v,-1) for v in test.user]);t1=np.array([im.get(v,-1) for v in test.item1]);t2=np.array([im.get(v,-1) for v in test.item2]);assert min(tu.min(),t1.min(),t2.min())>=0
    part=out/'full';part.mkdir(exist_ok=True);names2,ff=features(full,np.r_[tu,tu],np.r_[t1,t2],part);assert names2==names
    n=len(test);d=ff[:n]-ff[n:];s=ff[:n]+ff[n:];linear=1/(1+np.exp(-np.clip(d@w,-50,50)))-.5;nonlinear=predict(model,d,s)/2 if model else np.zeros(n);margin=(1-alpha)*linear+alpha*nonlinear
    # Assert antisymmetry independently, including nonlinear combiner.
    reverse=(1-alpha)*(1/(1+np.exp(np.clip(d@w,-50,50)))-.5)-alpha*nonlinear;assert np.allclose(margin,-reverse,atol=1e-7)
    test['which']=np.where(margin>=0,test.item1,test.item2);assert len(test)==20000 and ((test.which==test.item1)|(test.which==test.item2)).all();assert np.isfinite(margin).all()
    test.to_csv(out/'task_2_predictions.csv',index=False);log('SAVED',out/'task_2_predictions.csv')

if __name__=='__main__':main()
