"""Task 3 rating ensemble: SVD++, temporal residual MF and local context.
All rating-zero training rows are excluded. Use fresh output after source changes.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,ctypes,hashlib,json,pickle,subprocess,sys,time
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from catboost import CatBoostRegressor
from task3_temporal_model import TemporalBias

CONFIG={'user':5,'item':10,'month':100,'drift':40,'user90':30,'item90':60,'userday':20}
def log(*a):print(*a,flush=True)
def mse(y,p):return float(np.mean((y-p)**2))
def clip(p):return np.clip(p,1,5)
def sparse(v,u,i,shape):return coo_matrix((v,(u,i)),shape=shape).tocsr()

def dataset(data,full):
    p=pd.read_csv(data/'tasks_3_4_train.csv');p=p[p.rating>0].reset_index(drop=True)
    u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);t=p.timestamp.to_numpy(float);y=p.rating.to_numpy(float)
    order=np.random.RandomState(42).permutation(len(p));a=int(.9*len(p));b=int(.95*len(p));idx=np.arange(len(p)) if full else order[:a]
    query=pd.read_csv(data/'task_3_test.csv') if full else p.iloc[order[a:]].reset_index(drop=True)
    qu=pd.Index(users).get_indexer(query.user);qi=pd.Index(items).get_indexer(query.item)
    return p,p.iloc[idx],query,u[idx].astype(np.int32),i[idx].astype(np.int32),t[idx],y[idx],qu.astype(np.int32),qi.astype(np.int32),users,items,a,b


def smooth(train,query,residual,key,tau,penalty):
    groups={k:ix for k,ix in train.groupby(key,sort=False).indices.items()};out=np.zeros(len(query))
    t=train.timestamp.to_numpy(float)/86400;qt=query.timestamp.to_numpy(float)/86400
    for k,qs in query.groupby(key,sort=False).indices.items():
        rows=groups.get(k)
        if rows is None:continue
        rows=rows[(t[rows]>=0)&(t[rows]<=4102444800/86400)]
        if not len(rows):continue
        # Bound peak allocation for highly active users/items.
        for ix in np.array_split(qs,max(1,(len(qs)+127)//128)):
            w=np.exp(-np.abs(qt[ix,None]-t[rows][None,:])/tau);out[ix]=w@residual[rows]/(w.sum(1)+penalty)
    out[(qt<0)|(qt>4102444800/86400)]=0;return out


def library_worker(data,out,full):
    from surprise import SVDpp,Dataset,Reader
    target=out/'library_features.npz'
    if target.exists():return
    p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);start=time.time()
    training=Dataset.load_from_df(tr[['user','item','rating']],Reader(rating_scale=(1,5))).build_full_trainset()
    m=SVDpp(n_factors=24,n_epochs=25,lr_all=.007,reg_all=.08,random_state=42,cache_ratings=True);m.fit(training)
    U=pd.Index([training.to_raw_uid(k) for k in range(training.n_users)]);I=pd.Index([training.to_raw_iid(k) for k in range(training.n_items)])
    cu=U.get_indexer(tr.user);ci=I.get_indexer(tr.item);H=sparse(np.ones(len(tr)),cu,ci,(len(U),len(I)));count=np.asarray(H.sum(1)).ravel()
    effective=m.pu+(H@m.yj)/np.sqrt(np.maximum(count,1))[:,None]
    def predict(frame):
        uu=U.get_indexer(frame.user);ii=I.get_indexer(frame.item);v=np.full(len(frame),training.global_mean);vu=uu>=0;vi=ii>=0;ok=vu&vi
        v[vu]+=m.bu[uu[vu]];v[vi]+=m.bi[ii[vi]];v[ok]+=np.einsum('ij,ij->i',effective[uu[ok]],m.qi[ii[ok]]);return clip(v)
    pred=predict(q);tp=predict(tr)
    sample=np.random.RandomState(42).choice(len(q),min(128,len(q)),replace=False);reference=np.array([m.predict(q.user.iloc[k],q.item.iloc[k]).est for k in sample]);assert np.allclose(pred[sample],reference,atol=1e-10)
    r=y-tp;du=smooth(tr,q,r,'user',30,3);di=smooth(tr,q,r,'item',90,20)
    np.savez_compressed(target,svdpp24=pred,svdpp_user30=clip(pred+du),svdpp_temporal=clip(pred+du+di))
    np.savez_compressed(out/'svdpp_model.npz',users=U.to_numpy(dtype=str),items=I.to_numpy(dtype=str),effective=effective,qi=m.qi,bu=m.bu,bi=m.bi,mu=training.global_mean)
    log('Library SVD++ + temporal corrections complete; seconds',round(time.time()-start,1))


def mf_library(out):
    so=out/'mf_kernel.so'
    if not so.exists():subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task3_mf_kernel.cpp')),'-o',str(so)],check=True)
    lib=ctypes.CDLL(str(so.resolve()));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.train_mf.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,ip,fp,fp,fp,fp,fp,fp,fp,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float,ctypes.c_float,ctypes.c_float,ctypes.c_float];lib.train_mf.restype=None
    lib.predict_mf.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ip,ip,ip,fp,fp,ip,ip,fp,fp,fp,fp,ctypes.c_int,fp];lib.predict_mf.restype=None
    return lib


def context_features(train,query,base_train,base_query):
    result={};qt=query.timestamp.to_numpy(float)/86400;tt=train.timestamp.to_numpy(float)/86400;y=train.rating.to_numpy(float);good=(tt>=0)&(tt<=4102444800/86400);res=y-base_train
    result['valid_query_date']=((qt>=0)&(qt<=4102444800/86400)).astype(float);result['query_year']=np.clip(qt/365.25,0,130);result['year_sin']=np.sin(2*np.pi*qt/365.25);result['year_cos']=np.cos(2*np.pi*qt/365.25)
    for key in ['user','item']:
        groups={k:ix for k,ix in train.groupby(key,sort=False).indices.items()}
        names=['prev_rating','next_rating','prev_gap','next_gap','has_prev','has_next','rating_entropy']
        for day in [0,1,7,30,90,365]:names += ['mean'+str(day),'std'+str(day),'count'+str(day),'residual'+str(day)]
        for value in range(1,6):names.append('fraction'+str(value))
        for n in names:result[key+'_'+n]=np.zeros(len(query))
        for name,qs in query.groupby(key,sort=False).indices.items():
            rows=groups.get(name)
            if rows is None:continue
            vals=y[rows];freq=np.bincount(vals.astype(int),minlength=6)[1:6]/len(vals)
            for value in range(1,6):result[key+'_fraction'+str(value)][qs]=freq[value-1]
            result[key+'_rating_entropy'][qs]=-np.sum(freq*np.log(np.maximum(freq,1e-12)))
            rows=rows[good[rows]];rows=rows[np.argsort(tt[rows],kind='stable')]
            if not len(rows):continue
            dates=tt[rows];v=y[rows];rr=res[rows];c=np.r_[0,np.cumsum(v)];c2=np.r_[0,np.cumsum(v*v)];cr=np.r_[0,np.cumsum(rr)];q=qt[qs];pos=np.searchsorted(dates,q)
            for side,ids,ok in [('prev',np.maximum(pos-1,0),pos>0),('next',np.minimum(pos,len(rows)-1),pos<len(rows))]:
                result[key+'_'+side+'_rating'][qs]=np.where(ok,v[ids],0);result[key+'_'+side+'_gap'][qs]=np.log1p(np.where(ok,np.abs(q-dates[ids]),100000));result[key+'_has_'+side][qs]=ok
            for h in [0,1,7,30,90,365]:
                if h==0:
                    lo=np.searchsorted(dates,np.floor(q),side='left');hi=np.searchsorted(dates,np.floor(q)+1,side='left')
                else:lo=np.searchsorted(dates,q-h);hi=np.searchsorted(dates,q+h,side='right')
                n=hi-lo;total=c[hi]-c[lo];mean=np.divide(total,n,out=np.zeros(len(n)),where=n>0);var=np.maximum(0,np.divide(c2[hi]-c2[lo],n,out=np.zeros(len(n)),where=n>0)-mean**2)
                result[key+'_mean'+str(h)][qs]=mean;result[key+'_std'+str(h)][qs]=np.sqrt(var);result[key+'_count'+str(h)][qs]=np.log1p(n)
                result[key+'_residual'+str(h)][qs]=(cr[hi]-cr[lo])/(n+(3 if key=='user' else 20))
    return result


def user_neighbors(u,i,t,y,qu,qi,qt,base_train,U,I):
    shape=(U,I);X=sparse(np.ones(len(u),np.float32),u,i,shape);up=np.asarray(X.sum(1)).ravel();mean=np.bincount(u,weights=y,minlength=U)/np.maximum(up,1)
    C=sparse(y-mean[u],u,i,shape);R=sparse(y-base_train,u,i,shape).tocsc();Times=sparse(t,u,i,shape).tocsc();norm=np.sqrt(np.asarray(C.multiply(C).sum(1)).ravel());result={}
    for sim in ['binary','pearson']:
        for tau in [0,365]:
            for k in [20,50]:result[f'usernn_{sim}_{tau}_{k}']=np.zeros(len(qu))
        result['usernn_'+sim+'_support']=np.zeros(len(qu))
    groups={}
    for a,user in enumerate(qu):
        if user>=0 and qi[a]>=0:groups.setdefault(user,[]).append(a)
    keys=np.array(list(groups));XT=X.T.tocsr();CT=C.T.tocsr()
    for offset in range(0,len(keys),64):
        batch=keys[offset:offset+64];co=(X[batch]@XT).toarray();dot=(C[batch]@CT).toarray()
        for row,user in enumerate(batch):
            binary=co[row]/np.sqrt(np.maximum(up[user]*up,1));pearson=np.maximum(0,dot[row]/np.maximum(norm[user]*norm,1e-8));shrink=co[row]/(co[row]+10);binary*=shrink;pearson*=shrink;binary[user]=0;pearson[user]=0
            for a in groups[user]:
                j=qi[a];lo,hi=R.indptr[j:j+2];other=R.indices[lo:hi];res=R.data[lo:hi];distance=np.abs(Times.data[lo:hi]-qt[a])/86400;valid=(Times.data[lo:hi]>=0)&(Times.data[lo:hi]<=4102444800)
                for name,sim in [('binary',binary),('pearson',pearson)]:
                    w0=sim[other];result['usernn_'+name+'_support'][a]=w0.sum()
                    for tau in [0,365]:
                        w=w0 if tau==0 else w0*np.exp(-distance/tau)*valid
                        for k in [20,50]:
                            ix=np.argpartition(w,-k)[-k:] if len(w)>k else np.arange(len(w));result[f'usernn_{name}_{tau}_{k}'][a]=(w[ix]@res[ix])/(w[ix].sum()+.25)
        if offset%4096==0:log('User-neighborhood rows',offset,'/',len(keys))
    return result


def temporal_worker(data,out,full):
    target=out/'temporal_features.npz'
    if target.exists():return
    p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);U,I=len(users),len(items);start=time.time();base=TemporalBias(CONFIG).fit(u.astype(np.int64),i.astype(np.int64),t,y,U);bq=base.predict(qu.astype(np.int64),qi.astype(np.int64),q.timestamp.to_numpy(float));features={'temporal_base':clip(bq)}
    def dev(uu,tt):
        valid=(tt>=0)&(tt<=4102444800)&(uu>=0);d=(tt-base.mean_time[np.maximum(uu,0)])/(86400*365.25);return np.ascontiguousarray(np.where(valid,np.clip(np.sign(d)*np.abs(d)**.4,-5,5),0),dtype=np.float32)
    def keys(uu,tt):return (uu.astype(np.int64)+1)*1000000000+np.where((tt>=0)&(tt<=4102444800),np.floor(np.clip(tt,0,4102444800)/86400/180),-1).astype(np.int64)
    period,index=pd.factorize(keys(u,t));period=np.ascontiguousarray(period,np.int32);qp=np.ascontiguousarray(pd.Index(index).get_indexer(keys(qu,q.timestamp.to_numpy(float))),np.int32);count=np.bincount(period);scale=np.ascontiguousarray(np.sqrt(count[period]/(count[period]+10)),np.float32);qs=np.zeros(len(q),np.float32);ok=qp>=0;qs[ok]=np.sqrt(count[qp[ok]]/(count[qp[ok]]+10));dv=dev(u,t);qd=dev(qu,q.timestamp.to_numpy(float));res=np.ascontiguousarray(y-base.training_prediction,np.float32);uc=np.ascontiguousarray(np.bincount(u,minlength=U),np.int32);ic=np.ascontiguousarray(np.bincount(i,minlength=I),np.int32);lib=mf_library(out)
    for name,k,mode,epochs,lr,reg,seed in [('residual_mf32',32,0,[30,60],.01,.06,42),('time_mf32',32,1,[30,60],.008,.08,17),('time_mf64',64,1,[30,50],.007,.1,23)]:
        rng=np.random.RandomState(seed);P=(rng.normal(0,.05,(U,k))).astype(np.float32);Q=(rng.normal(0,.05,(I,k))).astype(np.float32);A=np.zeros_like(P);S=np.zeros((len(index),k),np.float32);last=0
        for epoch in epochs:
            lib.train_mf(len(u),k,u,i,period,dv,scale,res,P,A,S,Q,mode,last,epoch-last,seed,lr,reg,reg,.3,.2);last=epoch
            assert all(np.isfinite(v).all() for v in [P,Q,A,S]);pred=np.zeros(len(q),np.float32);lib.predict_mf(len(q),k,U,I,qu,qi,qp,qd,qs,uc,ic,P,A,S,Q,mode,pred);features[name+'_e'+str(epoch)]=clip(bq+pred)
            np.savez_compressed(out/(name+'_e'+str(epoch)+'.npz'),P=P,Q=Q,A=A,S=S,period_keys=np.asarray(index),mean_time=base.mean_time,uc=uc,ic=ic,mode=mode)
            log(name,'checkpoint',epoch,'seconds elapsed',round(time.time()-start,1))
    features.update(context_features(tr,q,base.training_prediction,bq));features.update(user_neighbors(u,i,t,y,qu,qi,q.timestamp.to_numpy(float),base.training_prediction,U,I))
    assert all(np.isfinite(v).all() for v in features.values());np.savez_compressed(target,**features)
    with open(out/'temporal_bias.pkl','wb') as f:pickle.dump(base,f)
    log('Temporal MF, rating context and user neighborhoods complete')


def baseline_validation(data,legacy):
    from task3_components import extra_features
    p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,False);split=np.load(legacy/'split.npz');assert np.array_equal(split['y'],p.rating.to_numpy()) and np.array_equal(split['time'],p.timestamp.to_numpy(float));assert np.array_equal(split['order'],np.random.RandomState(42).permutation(len(p)))
    parts={}
    for prefix,file in [('bias','predictions'),('grid','grid_predictions'),('neighbor','neighbors_predictions'),('kernel','kernel_predictions')]:
        z=np.load(legacy/(file+'.npz'))
        for key in z.files:
            if key.startswith('v_'):parts[prefix+'_'+key[2:]]=np.r_[z[key],z['t_'+key[2:]]]
    names=json.loads((legacy/'meta_results.json').read_text())['base_columns'];F=np.column_stack([parts[k] for k in names]);F=np.c_[F,extra_features(tr,q)];return F,q.rating.to_numpy(),b-a


def baseline_full(data,out):
    target=out/'baseline_features.npz'
    if target.exists():return
    from task3_components import generate,extra_features
    p,tr,q,*rest=dataset(data,True);configs=json.loads(Path(__file__).with_name('task3_configs.json').read_text());components=generate(tr,q,configs)
    # Ordering matches the original meta model, regardless of dictionary generation order.
    names=['bias_static','bias_drift_item','bias_temporal180','bias_temporal90','bias_temporal90_strong','bias_svd16','bias_svd_global_month','bias_svd32']+[k for k in configs if k.startswith('grid_')]
    names += [f'neighbor_{kind}_20_{power}_{tau}' for kind in ['static','temporal'] for tau in [0,365,90] for power in [1,2]]
    names += [f'kernel_{kind}_{tau}_{reg}' for kind in ['static','temporal'] for tau in [7,30,90,365] for reg in [1,5]]
    F=np.column_stack([components[k] for k in names]);F=np.c_[F,extra_features(tr,q)];np.savez_compressed(target,features=F)
    log('Full-data baseline component cache complete')


def train_tree(X,y,depth,iters,seed=42,lr=.03):
    m=CatBoostRegressor(loss_function='RMSE',iterations=iters,depth=depth,learning_rate=lr,l2_leaf_reg=10,random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False);m.fit(X,y);return m

def select(base,extra,y,n,out):
    X=np.c_[base,extra];idx=np.random.RandomState(17).permutation(n);cut=int(.8*n);tr,va=idx[:cut],idx[cut:];old=train_tree(base[tr],y[tr],6,698);oldp=clip(old.predict(base[va]));best=(mse(y[va],oldp),0,6,698);reports={'baseline_inner_MSE':best[0]}
    for depth in [4,6,8]:
        m=CatBoostRegressor(loss_function='RMSE',iterations=1600,depth=depth,learning_rate=.03,l2_leaf_reg=10,random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
        m.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=100);iters=m.get_best_iteration()+1;p=clip(m.predict(X[va]))
        for alpha in [.5,.75,1.]:
            score=mse(y[va],alpha*p+(1-alpha)*oldp);reports[f'depth{depth}_mix{alpha}']=dict(inner_MSE=score,iterations=iters)
            if score<best[0]:best=(score,alpha,depth,iters)
        log('Stacker depth',depth,'iters',iters,'inner MSE',mse(y[va],p))
    score,alpha,depth,iters=best;old=train_tree(base[:n],y[:n],6,698);old.save_model(str(out/'baseline_meta.cbm'));m=train_tree(X[:n],y[:n],depth,iters) if alpha else None
    if m:m.save_model(str(out/'extended_meta.cbm'))
    po=clip(old.predict(base[n:]));pn=clip(m.predict(X[n:])) if m else po;pred=alpha*pn+(1-alpha)*po
    r=dict(selected=dict(inner_MSE=score,new_weight=alpha,depth=depth,iterations=iters),search=reports,baseline_evaluation_MSE=mse(y[n:],po),extended_evaluation_MSE=mse(y[n:],pn),selected_evaluation_MSE=mse(y[n:],pred),conservative_evaluation_MSE=mse(y[n:],.5*pn+.5*po))
    np.savez_compressed(out/'evaluation.npz',baseline=po,extended=pn,selected=pred,y=y[n:]);(out/'results.json').write_text(json.dumps(r,indent=2));log('SELECTED',r['selected'],'EVALUATION',r['selected_evaluation_MSE']);return old,m,alpha,r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--baseline-dir',type=Path);ap.add_argument('--predict',action='store_true');ap.add_argument('--worker',choices=['library','temporal','baseline']);ap.add_argument('--full',action='store_true');args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);os.environ.setdefault('SURPRISE_DATA_FOLDER',str(args.output_dir/'surprise_data'))
    if args.worker:
        {'library':library_worker,'temporal':temporal_worker,'baseline':baseline_full}[args.worker](args.data_dir,args.output_dir,args.full) if args.worker!='baseline' else baseline_full(args.data_dir,args.output_dir)
        return
    legacy=args.baseline_dir or args.output_dir/'baseline'
    if not (legacy/'split.npz').exists():
        subprocess.run([sys.executable,str(Path(__file__).with_name('task3_pipeline.py')),'--data-dir',str(args.data_dir),'--output-dir',str(legacy),'--predict'],check=True)
        legacy=Path(json.loads((legacy/'run.json').read_text())['cache'])
    args.baseline_dir=legacy.resolve()
    manifest={name:hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest() for name in ['tasks_3_4_train.csv','task_3_test.csv']};mf=args.output_dir/'input_manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest:raise ValueError('Data changed; use a fresh output directory')
    mf.write_text(json.dumps(manifest,indent=2))
    def workers(part,full,kinds):
        jobs=[]
        for kind in kinds:
            h=open(part/(kind+'.log'),'a');cmd=[sys.executable,str(Path(__file__).resolve()),'--data-dir',str(args.data_dir),'--output-dir',str(part),'--baseline-dir',str(args.baseline_dir),'--worker',kind]+(['--full'] if full else []);jobs.append((subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT),h,kind))
        for p,h,kind in jobs:
            rc=p.wait();h.close()
            if rc:raise RuntimeError(f'{kind} worker failed; inspect {part/(kind+".log")}')
    part=args.output_dir/'validation';part.mkdir(exist_ok=True);workers(part,False,['library','temporal']);base,y,n=baseline_validation(args.data_dir,args.baseline_dir);zs=[np.load(part/(kind+'_features.npz')) for kind in ['library','temporal']];names=[k for z in zs for k in z.files];extra=np.column_stack([z[k] for z in zs for k in z.files]);assert np.isfinite(extra).all();old,model,alpha,r=select(base,extra,y,n,part);r['extra_feature_names']=names;r['base_feature_count']=base.shape[1];(part/'results.json').write_text(json.dumps(r,indent=2))
    if not args.predict:return
    part=args.output_dir/'full';part.mkdir(exist_ok=True);workers(part,True,['library','temporal','baseline']);base=np.load(part/'baseline_features.npz')['features'];zs=[np.load(part/(kind+'_features.npz')) for kind in ['library','temporal']];assert names==[k for z in zs for k in z.files];extra=np.column_stack([z[k] for z in zs for k in z.files]);X=np.c_[base,extra];po=clip(old.predict(base));pn=clip(model.predict(X)) if model else po;pred=clip(alpha*pn+(1-alpha)*po);test=pd.read_csv(args.data_dir/'task_3_test.csv');test['rating']=pred;assert len(test)==20000 and np.isfinite(pred).all();test.to_csv(args.output_dir/'task_3_predictions.csv',index=False);np.savez_compressed(args.output_dir/'test_predictions.npz',baseline=po,extended=pn,selected=pred)
    if model:
        weight=.5 if alpha!=.5 else 1.
        alternate=test.copy();alternate['rating']=clip(weight*pn+(1-weight)*po);alternate.to_csv(args.output_dir/'task_3_alternative_predictions.csv',index=False)
    log('SAVED',args.output_dir/'task_3_predictions.csv')

if __name__=='__main__':main()
