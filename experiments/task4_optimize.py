"""Task 4: time-aware sparse-linear/factor ensemble. Retains zero ratings."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,ctypes,hashlib,json,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix,load_npz
from scipy.linalg import cho_factor,cho_solve
from implicit.bpr import BayesianPersonalizedRanking
from implicit.als import AlternatingLeastSquares
from catboost import CatBoostClassifier
from task2_optimize import linear_neighbors


def log(*a):print(*a,flush=True)
def acc(m):return float(np.mean((m>0)+.5*(m==0)))
def binary(u,i,shape):
    X=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=shape).tocsr();X.data[:]=1;return X

def fit_factor(name,X,out,kind,params):
    path=out/(name+'.npz');start=time.time()
    if path.exists():
        z=np.load(path);return SimpleNamespace(user_factors=z['user'],item_factors=z['item'])
    cls=BayesianPersonalizedRanking if kind=='bpr' else AlternatingLeastSquares
    m=cls(**params);m.fit(X,show_progress=False)
    np.savez_compressed(path,user=m.user_factors,item=m.item_factors)
    log('Fit',name,'seconds',round(time.time()-start,1));return m

def period_keys(u,t,offset):
    good=(t>=0)&(t<=4102444800)
    p=np.where(good,np.floor((np.clip(t,0,4102444800)/86400-offset)/365),-1000).astype(np.int64)
    return (u+1)*100000+p

def factors_and_scores(X,u,i,t,qu,qi,qt,out):
    periods=[]
    for offset in [0,182]:
        code,index=pd.factorize(period_keys(u,t,offset));B=binary(code,i,(len(index),X.shape[1]));periods.append((offset,code,pd.Index(index),B))
    specs=[('bpr128',X,'bpr',dict(factors=128,iterations=250,regularization=.03,learning_rate=.03,random_state=42,num_threads=1)),
           ('als128new',X,'als',dict(factors=128,iterations=30,regularization=.1,alpha=40,random_state=42,num_threads=1))]
    for offset,code,index,B in periods:
        specs.append(('time_bpr365_'+str(offset),B,'bpr',dict(factors=64,iterations=150,regularization=.03,learning_rate=.03,random_state=42+offset,num_threads=1)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={name:pool.submit(fit_factor,name,A,out,kind,params) for name,A,kind,params in specs}
        models={name:f.result() for name,f in futures.items()}
    pop=np.asarray(X.sum(0)).ravel();up=np.asarray(X.sum(1)).ravel();valid=(pop[qi]>0)&(up[qu]>0);s={}
    for name in ['bpr128','als128new']:
        m=models[name];v=np.einsum('ij,ij->i',m.user_factors[qu],m.item_factors[qi]);v[~valid]=0;s[name]=v
    for offset,code,index,B in periods:
        name='time_bpr365_'+str(offset);m=models[name];keys=period_keys(qu,qt,offset);c=index.get_indexer(keys);ok=(c>=0)&valid
        v=s['bpr128'].copy();v[ok]=np.einsum('ij,ij->i',m.user_factors[c[ok]],m.item_factors[qi[ok]])
        s[name]=v;s[name+'_available']=ok.astype(np.float32)
        counts=np.zeros(len(qu));counts[ok]=np.diff(B.indptr)[c[ok]];s[name+'_history_count']=np.log1p(counts)
    V=models['als128new'].item_factors;V=V/np.maximum(np.linalg.norm(V,axis=1,keepdims=True),1e-8)
    return s,V


def ease(X,qu,qi,out):
    pop=np.asarray(X.sum(0)).ravel();top=np.argsort(pop)[-min(4000,X.shape[1]):];Z=X[:,top].astype(np.float64);G=(Z.T@Z).toarray()
    mapping=np.full(X.shape[1],-1,np.int32);mapping[top]=np.arange(len(top));cq=mapping[qi];scores={};context=None
    for reg in [100,500,2000]:
        path=out/('ease'+str(reg)+'.npy')
        if path.exists():B=np.load(path)
        else:
            A=G.copy();A.flat[::len(top)+1]+=reg;P=cho_solve(cho_factor(A),np.eye(len(top)));B=-P/np.diag(P)[None,:];np.fill_diagonal(B,0);B=B.astype(np.float32);np.save(path,B);del A,P
        v=np.zeros(len(qu),np.float32);valid=np.flatnonzero(cq>=0)
        for ix in np.array_split(valid,max(1,(len(valid)+999)//1000)):
            if len(ix):v[ix]=np.asarray(Z[qu[ix]].multiply(B[:,cq[ix]].T).sum(1)).ravel()
        scores['ease'+str(reg)]=v
        if reg==500:context=B
    scores['ease_covered']=(cq>=0).astype(np.float32);np.save(out/'ease_vocabulary.npy',top)
    return scores,context,mapping


def kde_library(out):
    path=out/'temporal_kernel.so'
    if not path.exists():subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task4_temporal_kernel.cpp')),'-o',str(path)],check=True)
    lib=ctypes.CDLL(str(path.resolve()));fn=lib.temporal_kde;ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');fn.argtypes=[ctypes.c_int,ptr,ctypes.c_int,ptr,ctypes.c_double,ptr];fn.restype=None
    def evaluate(x,q,h):
        x=np.ascontiguousarray(x,dtype=np.float64);q=np.ascontiguousarray(q,dtype=np.float64);v=np.zeros(len(q));fn(len(x),x,len(q),q,h,v);return v
    return evaluate


def time_features(u,i,t,qu,qi,qt,out):
    """All dates below come from fitted training rows, never held-out targets."""
    kde=kde_library(out);valid=(t>=0)&(t<=4102444800);global_dates=np.sort(t[valid]/86400);q=qt/86400;good=(qt>=0)&(qt<=4102444800)
    result={'query_valid_time':good.astype(float),'query_year':np.clip(q/365.25,0,130),'query_year_sin':np.sin(2*np.pi*q/365.25),'query_year_cos':np.cos(2*np.pi*q/365.25)}
    for name in ['book_date_count','book_time_position','book_time_z','book_last_age','book_prev_gap','book_next_gap','book_has_prev','book_has_next']:
        result[name]=np.zeros(len(q))
    for h in [7,30,90,365]:
        result['book_kde'+str(h)]=np.full(len(q),np.log(1e-10));result['book_kde_lift'+str(h)]=np.zeros(len(q))
    windows=[1,7,14,60,180,730]
    global_counts={h:np.searchsorted(global_dates,q+h,side='right')-np.searchsorted(global_dates,q-h,side='left') for h in windows}
    global_density={h:kde(global_dates,q,h) for h in [7,30,90,365]}
    for h in windows:
        for n in ['book_count','book_fraction','book_lift']:result[n+str(h)]=np.zeros(len(q))
    sort=np.argsort(i[valid],kind='stable');it=i[valid][sort];ts=t[valid][sort]/86400;keys,starts=np.unique(it,return_index=True);ends=np.r_[starts[1:],len(it)];dates={k:np.sort(ts[a:b]) for k,a,b in zip(keys,starts,ends)}
    order=np.argsort(qi,kind='stable');keys,starts=np.unique(qi[order],return_index=True);ends=np.r_[starts[1:],len(order)]
    for j,a,b in zip(keys,starts,ends):
        ix=order[a:b];x=dates.get(j,np.array([]));N=len(x)
        if not N:continue
        y=q[ix];p=np.searchsorted(x,y);hasprev=p>0;hasnext=p<N;prev=y-x[np.maximum(p-1,0)];nxt=x[np.minimum(p,N-1)]-y
        result['book_date_count'][ix]=np.log1p(N);result['book_time_position'][ix]=p/N
        q25,med,q75=np.quantile(x,[.25,.5,.75]);result['book_time_z'][ix]=np.clip((y-med)/(q75-q25+30),-20,20)
        age=y-x[-1];result['book_last_age'][ix]=np.sign(age)*np.log1p(np.abs(age))
        result['book_prev_gap'][ix]=np.log1p(np.where(hasprev,prev,100000));result['book_next_gap'][ix]=np.log1p(np.where(hasnext,nxt,100000));result['book_has_prev'][ix]=hasprev;result['book_has_next'][ix]=hasnext
        for h in windows:
            count=np.searchsorted(x,y+h,side='right')-np.searchsorted(x,y-h,side='left')
            result['book_count'+str(h)][ix]=np.log1p(count);result['book_fraction'+str(h)][ix]=count/N
            result['book_lift'+str(h)][ix]=np.log((count+.5)/(N+1))-np.log((global_counts[h][ix]+.5)/(len(global_dates)+1))
        for h in [7,30,90,365]:
            density=kde(x,y,h);result['book_kde'+str(h)][ix]=np.log(density+1e-10);result['book_kde_lift'+str(h)][ix]=np.log(density+1e-10)-np.log(global_density[h][ix]+1e-10)
    for name,value in result.items():
        if name!='query_valid_time':value[~good]=0
    log('Fine-grained book activity and KDE complete');return result


def history_features(X,Times,qu,qi,qt,W,E,mapping,V,out):
    assert np.array_equal(X.indptr,Times.indptr) and np.array_equal(X.indices,Times.indices)
    score_names=['history_count','history_valid_count','history_prev_gap','history_next_gap','history_has_prev','history_has_next','history_position','als_prev','als_next','slim_prev','slim_next','ease_prev','ease_next']
    for h in [30,180,365]:score_names += ['history_local'+str(h),'slim_time'+str(h),'slim_mean'+str(h),'ease_time'+str(h),'als_time'+str(h),'als_max_time'+str(h)]
    result={n:np.zeros(len(qu),np.float32) for n in score_names};C=W.tocsc();order=np.argsort(qi,kind='stable');keys,starts=np.unique(qi[order],return_index=True);ends=np.r_[starts[1:],len(order)]
    for z,(j,a,b) in enumerate(zip(keys,starts,ends)):
        weight=np.zeros(X.shape[1],np.float32);lo,hi=C.indptr[j:j+2];weight[C.indices[lo:hi]]=C.data[lo:hi];ej=mapping[j]
        for ix in order[a:b]:
            lo,hi=X.indptr[qu[ix]:qu[ix]+2];hist=X.indices[lo:hi];ht=Times.data[lo:hi]/86400;good=(ht>=0)&(ht<=4102444800/86400)
            result['history_count'][ix]=np.log1p(len(hist));hist=hist[good];ht=ht[good];result['history_valid_count'][ix]=np.log1p(len(hist))
            if not len(hist) or qt[ix]<0 or qt[ix]>4102444800:continue
            date=qt[ix]/86400;distance=np.abs(ht-date);sw=weight[hist];vh=V[hist]@V[j];vh[hist==j]=0
            ew=np.zeros(len(hist),np.float32);ehi=mapping[hist];ok=ehi>=0
            if ej>=0:ew[ok]=E[ehi[ok],ej]
            before=np.flatnonzero(ht<date);after=np.flatnonzero(ht>=date);result['history_position'][ix]=len(before)/len(hist)
            for side,ids in [('prev',before),('next',after)]:
                if len(ids):
                    k=ids[np.argmin(distance[ids])];result['history_'+side+'_gap'][ix]=np.log1p(distance[k]);result['history_has_'+side][ix]=1
                    result['als_'+side][ix]=vh[k];result['slim_'+side][ix]=sw[k];result['ease_'+side][ix]=ew[k]
                else:result['history_'+side+'_gap'][ix]=np.log1p(100000)
            for h in [30,180,365]:
                w=np.exp(-distance/h);den=max(w.sum(),1e-8);result['history_local'+str(h)][ix]=np.log1p(w.sum());s=sw@w
                result['slim_time'+str(h)][ix]=s;result['slim_mean'+str(h)][ix]=s/den;result['ease_time'+str(h)][ix]=ew@w;result['als_time'+str(h)][ix]=vh@w/den;result['als_max_time'+str(h)][ix]=np.max(vh*w)
        if z%4096==0:log('Temporal history candidates',z,'/',len(keys))
    return result


def extended_features(X,u,i,t,qu,qi,qt,out):
    path=out/'extended_features.npz'
    if path.exists():
        z=np.load(path);return z.files,np.column_stack([z[n] for n in z.files])
    result,V=factors_and_scores(X,u,i,t,qu,qi,qt,out)
    es,E,mapping=ease(X,qu,qi,out);result.update(es);log('EASE complete')
    linear=linear_neighbors(X,qu,qi,out);result.update({k:v for k,v in linear.items() if k.startswith('slim')})
    result.update(time_features(u,i,t,qu,qi,qt,out))
    Times=coo_matrix((t,(u,i)),shape=X.shape).tocsr();W=load_npz(out/'slim02.npz')
    result.update(history_features(X,Times,qu,qi,qt,W,E,mapping,V,out))
    assert all(np.isfinite(v).all() for v in result.values());np.savez_compressed(path,**result)
    return list(result),np.column_stack(list(result.values()))


def design(d,s):return np.column_stack([d,s])
def margin(m,d,s):return (m.predict_proba(design(d,s))[:,1]-m.predict_proba(design(-d,s))[:,1])/2

def fit_tree(d,s,rows,depth,iters,learning_rate=.035):
    m=CatBoostClassifier(iterations=iters,depth=depth,learning_rate=learning_rate,l2_leaf_reg=10,random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
    m.fit(np.r_[design(d[rows],s[rows]),design(-d[rows],s[rows])],np.r_[np.ones(len(rows)),np.zeros(len(rows))]);return m

def select_combiner(base,extra,names,selection_count,out):
    F=np.c_[base,extra];N=len(F)//2;d=F[:N]-F[N:];s=F[:N]+F[N:];v=d[:selection_count];vs=s[:selection_count];t=d[selection_count:];ts=s[selection_count:]
    idx=np.random.RandomState(17).permutation(selection_count);cut=int(.8*len(idx));tr,va=idx[:cut],idx[cut:];old_dim=base.shape[1]
    # Refit accepted architecture inside selection, so its inner predictions are not in-sample.
    old=fit_tree(v[:,:old_dim],vs[:,:old_dim],tr,5,698,learning_rate=.04);old_inner=margin(old,v[va,:old_dim],vs[va,:old_dim]);best=(acc(old_inner),0.,5,698);reports={'accepted_architecture_inner':best[0]}
    for depth in [5,7,9]:
        m=CatBoostClassifier(iterations=1600,depth=depth,learning_rate=.035,l2_leaf_reg=10,random_seed=42,thread_count=2,verbose=False,allow_writing_files=False,eval_metric='Accuracy')
        m.fit(np.r_[design(v[tr],vs[tr]),design(-v[tr],vs[tr])],np.r_[np.ones(len(tr)),np.zeros(len(tr))],eval_set=(np.r_[design(v[va],vs[va]),design(-v[va],vs[va])],np.r_[np.ones(len(va)),np.zeros(len(va))]),early_stopping_rounds=120)
        p=margin(m,v[va],vs[va]);iterations=m.get_best_iteration()+1
        for alpha in [.25,.5,.75,1.]:
            score=acc(alpha*p+(1-alpha)*old_inner);reports[f'depth{depth}_mix{alpha}']=dict(inner_accuracy=score,iterations=iterations)
            if score>best[0]:best=(score,alpha,depth,iterations)
        log('New stacker depth',depth,'iters',iterations,'inner',acc(p))
    score,alpha,depth,iters=best;fullrows=np.arange(selection_count);old=fit_tree(v[:,:old_dim],vs[:,:old_dim],fullrows,5,698,learning_rate=.04);old.save_model(str(out/'baseline_refit.cbm'))
    model=fit_tree(v,vs,fullrows,depth,iters) if alpha else None
    if model:model.save_model(str(out/'extended_stacker.cbm'))
    original_eval=margin(old,t[:,:old_dim],ts[:,:old_dim]);new_eval=margin(model,t,ts) if model else np.zeros(len(t));prediction=alpha*new_eval+(1-alpha)*original_eval
    r=dict(feature_names=names,baseline_features=old_dim,selected=dict(inner_accuracy=score,new_weight=alpha,depth=depth,iterations=iters),search=reports,baseline_evaluation_accuracy=acc(original_eval),extended_evaluation_accuracy=acc(new_eval) if model else None,selected_evaluation_accuracy=acc(prediction),component_evaluation={name:acc(t[:,j]) for j,name in enumerate(names)})
    (out/'results.json').write_text(json.dumps(r,indent=2));np.savez_compressed(out/'evaluation.npz',baseline=original_eval,extended=new_eval,selected=prediction)
    log('SELECTED',r['selected'],'EVALUATION',r['selected_evaluation_accuracy']);return old,model,alpha,r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--baseline-dir',type=Path);ap.add_argument('--predict',action='store_true');args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={p:hashlib.sha256((args.data_dir/p).read_bytes()).hexdigest() for p in ['tasks_3_4_train.csv','task_4_test.csv']};manifest['protocol']='task4_optimize_v1'
    mf=args.output_dir/'input_manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest:raise ValueError('Input files differ: choose a fresh output directory')
    mf.write_text(json.dumps(manifest,indent=2));baseline=args.baseline_dir or args.output_dir/'baseline';baseline.mkdir(parents=True,exist_ok=True);bm=baseline/'baseline_manifest.json'
    if not bm.exists() or json.loads(bm.read_text())!=manifest:
        subprocess.run([sys.executable,str(Path(__file__).with_name('task4_ensemble.py')),'--data-dir',str(args.data_dir),'--output-dir',str(baseline),'--predict','--stack'],check=True);bm.write_text(json.dumps(manifest,indent=2))
    data=pd.read_csv(args.data_dir/'tasks_3_4_train.csv');u,users=pd.factorize(data.user);i,items=pd.factorize(data.item);t=data.timestamp.to_numpy(float);shape=(len(users),len(items));assert not data.duplicated(['user','item']).any()
    order=np.random.RandomState(42).permutation(len(u));a=int(.9*len(u));b=int(.95*len(u));idx=order[:a];hold=order[a:];rng=np.random.RandomState(43);negative=[]
    for p in i[hold]:
        j=i[idx[rng.randint(len(idx))]]
        while j==p:j=i[idx[rng.randint(len(idx))]]
        negative.append(j)
    qu=np.r_[u[hold],u[hold]];qi=np.r_[i[hold],negative];qt=np.r_[t[hold],t[hold]];base=np.load(baseline/'pair_features.npz');assert np.array_equal(base['users'],qu) and np.array_equal(base['items'],qi)
    report=json.loads((baseline/'results.json').read_text());base_names=report['feature_names'];part=args.output_dir/'validation';part.mkdir(exist_ok=True);X=binary(u[idx],i[idx],shape)
    extra_names,extra=extended_features(X,u[idx],i[idx],t[idx],qu,qi,qt,part);names=base_names+extra_names;assert len(names)==len(set(names))
    old,model,alpha,r=select_combiner(base['scores'],extra,names,b-a,part);r.update(training_rows=a,selection_rows=b-a,evaluation_rows=len(u)-b,zero_rows_retained=int((data.rating==0).sum()),input_manifest=manifest);(part/'results.json').write_text(json.dumps(r,indent=2))
    if not args.predict:return
    test=pd.read_csv(args.data_dir/'task_4_test.csv');um=dict(zip(users,range(len(users))));im=dict(zip(items,range(len(items))));tu=np.array([um[v] for v in test.user]);i1=np.array([im[v] for v in test.item1]);i2=np.array([im[v] for v in test.item2]);qu=np.r_[tu,tu];qi=np.r_[i1,i2];qt=np.r_[test.timestamp.to_numpy(float),test.timestamp.to_numpy(float)];base=np.load(baseline/'test_pair_features.npz');assert np.array_equal(base['users'],qu) and np.array_equal(base['items'],qi)
    part=args.output_dir/'full';part.mkdir(exist_ok=True);X=binary(u,i,shape);en,ef=extended_features(X,u,i,t,qu,qi,qt,part);assert en==extra_names
    F=np.c_[base['scores'],ef];n=len(test);d=F[:n]-F[n:];s=F[:n]+F[n:];pold=margin(old,d[:,:len(base_names)],s[:,:len(base_names)]);pnew=margin(model,d,s) if model else np.zeros(n);p=alpha*pnew+(1-alpha)*pold
    assert np.allclose(margin(old,-d[:,:len(base_names)],s[:,:len(base_names)]),-pold)
    if model:assert np.allclose(margin(model,-d,s),-pnew)
    np.savez_compressed(args.output_dir/'test_margins.npz',selected=p,extended=pnew,baseline=pold)
    variants={'selected':p}
    if model and alpha not in [0.,1.]:variants['extended']=pnew
    elif model:variants['conservative']=.5*pnew+.5*pold
    for label,pred in variants.items():
        out=test.copy();out['which']=np.where(pred>=0,test.item1,test.item2);assert len(out)==20000 and ((out.which==out.item1)|(out.which==out.item2)).all();assert np.isfinite(pred).all()
        dest=args.output_dir/('task_4_predictions.csv' if label=='selected' else 'task_4_'+label+'_predictions.csv');out.to_csv(dest,index=False);log('SAVED',dest)

if __name__=='__main__':main()
