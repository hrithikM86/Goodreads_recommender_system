"""Task 5: joint FPMC and sequence/recency neighbors; no timestamp/rating values."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,json,hashlib,subprocess,time,fcntl
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix,save_npz

def data(data,baseline,full=False):
    p=pd.read_csv(data/'task_5_train.csv',usecols=['user','item']);u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);assert np.all(np.diff(u)>=0);U,I=len(users),len(items);ptr=np.searchsorted(u,np.arange(U+1));history=[i[ptr[j]:ptr[j+1]].astype(np.int32) for j in range(U)];assert min(map(len,history))>=4
    if full:
        q=pd.read_csv(data/'task_5_test.csv');qu=pd.Index(users).get_indexer(q.user);assert np.all(qu>=0),'Unknown test user';qi=np.r_[pd.Index(items).get_indexer(q.item1),pd.Index(items).get_indexer(q.item2)];contexts=[history[j] for j in qu];qu=np.r_[qu,qu];train=history
    else:
        train=[h[:-2] for h in history];contexts=train+[h[:-1] for h in history];positive=np.r_[[h[-2] for h in history],[h[-1] for h in history]];pool=np.concatenate(train);rng=np.random.RandomState(43);negative=np.empty(len(positive),np.int32)
        for z,pos in enumerate(positive):
            choice=pool[rng.randint(len(pool))]
            while choice==pos:choice=pool[rng.randint(len(pool))]
            negative[z]=choice
        qu=np.tile(np.arange(U),4);qi=np.r_[positive,negative];q=None
    cached=np.load(baseline/('test_pair_features.npz' if full else 'pair_features.npz'));assert np.array_equal(qu,cached['users']) and np.array_equal(qi,cached['items']);return train,contexts,qu.astype(np.int32),qi.astype(np.int32),U,I,q

def stage(args):
    part=args.out/('full' if args.full else 'validation');part.mkdir(parents=True,exist_ok=True);manifest={'full':args.full,'inputs':{f:hashlib.sha256((args.data_dir/f).read_bytes()).hexdigest() for f in ['task_5_train.csv','task_5_test.csv']}};guard=part/'manifest.json'
    if guard.exists():assert json.loads(guard.read_text())==manifest
    temp=part/f'manifest_{os.getpid()}.partial';temp.write_text(json.dumps(manifest,indent=2));temp.replace(guard);return part

def graph(history,U,I):
    row=np.repeat(np.arange(U),list(map(len,history)));items=np.concatenate(history);X=coo_matrix((np.ones(len(items),np.float32),(row,items)),shape=(U,I)).tocsr();X.sort_indices();assert np.max(X.data)==1;return X

def fpmc(args,uniform_only=False):
    out=stage(args)
    target=out/('fpmc_uniform_features.npz' if uniform_only else 'fpmc_features.npz');configs=[('fpmc64_useruniform',64,.03,.01,.5,23,1)] if uniform_only else [('fpmc32_uniform',32,.02,.02,0.,42,0),('fpmc64_mix',64,.02,.03,.5,17,0)]
    result=dict(np.load(target)) if target.exists() else {}
    if all(name+'_e100_joint' in result for name,*_ in configs):return
    history,contexts,qu,qi,U,I,q=data(args.data_dir,args.baseline,args.full);X=graph(history,U,I);pop=np.asarray(X.sum(0)).ravel();active=np.flatnonzero(pop>0).astype(np.int32);pool=np.concatenate(history).astype(np.int32);tu=np.repeat(np.arange(U),[len(h)-1 for h in history]).astype(np.int32);prev=np.concatenate([h[:-1] for h in history]);nxt=np.concatenate([h[1:] for h in history]);ptr=X.indptr.astype(np.int32);seen=X.indices.astype(np.int32)
    source=Path(__file__).with_name('task5_fpmc_kernel.cpp');so=out/('fpmc_uniform.so' if uniform_only else 'fpmc_v2.so');subprocess.run(['g++','-O3','-shared','-fPIC',str(source),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so.resolve()));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.train_fpmc.argtypes=[ctypes.c_int]*2+[ip]*6+[ctypes.c_int,ip,ctypes.c_int]+[fp]*5+[ctypes.c_int]*3+[ctypes.c_float]*3+[ctypes.c_int]
    ctx=np.tile(np.arange(len(contexts)),2);valid=qi>=0;known=valid.copy();known[valid]&=pop[qi[valid]]>0;last=np.array([h[-1] for h in contexts]);source_count=np.bincount(prev,minlength=I);start=time.time()
    for name,k,lr,reg,mix,seed,uniform in configs:
        if name+'_e100_joint' in result:continue
        rng=np.random.RandomState(seed);P=rng.normal(0,.05,(U,k)).astype(np.float32);Q=rng.normal(0,.05,(I,k)).astype(np.float32);A=rng.normal(0,.05,(I,k)).astype(np.float32);B=rng.normal(0,.05,(I,k)).astype(np.float32);bias=np.zeros(I,np.float32);epoch0=0
        if (out/(name+'_model.npz')).exists() and name+'_e40_joint' in result:
            saved=np.load(out/(name+'_model.npz'));P=saved['P'];Q=saved['Q'];A=saved['A'];B=saved['B'];bias=saved['bias'];epoch0=int(saved['epoch']) if 'epoch' in saved else 40
        for epoch in [15,40,100]:
            if epoch<=epoch0:continue
            lib.train_fpmc(len(tu),k,tu,prev,nxt,ptr,seen,pool,len(pool),active,len(active),P,Q,A,B,bias,epoch0,epoch-epoch0,seed,lr,reg,mix,uniform);epoch0=epoch;assert all(np.isfinite(z).all() for z in [P,Q,A,B,bias]);context=np.zeros((len(contexts),k),np.float32)
            for z,h in enumerate(contexts):
                hh=h[-3:][::-1];w=.7**np.arange(len(hh));w*=source_count[hh]>0;context[z]=w@A[hh]/max(w.sum(),1e-8)
            personal=np.zeros(len(qi));markov=np.zeros(len(qi));recent=np.zeros(len(qi));ib=np.zeros(len(qi));personal[known]=np.einsum('ij,ij->i',P[qu[known]],Q[qi[known]]);ok=known&(source_count[last[ctx]]>0);markov[ok]=np.einsum('ij,ij->i',A[last[ctx[ok]]],B[qi[ok]]);recent[known]=np.einsum('ij,ij->i',context[ctx[known]],B[qi[known]]);ib[known]=bias[qi[known]]
            for label,z in [('personal',personal),('last',markov),('recent3',recent),('bias',ib),('joint',personal+markov+ib)]:result[f'{name}_e{epoch}_{label}']=z
            # Save final checkpoint only; earlier validation predictions remain cached.
            if epoch==100:np.savez_compressed(out/(name+'_model.npz'),P=P,Q=Q,A=A,B=B,bias=bias,source_count=source_count,pop=pop,epoch=epoch)
            print(name,epoch,'seconds',round(time.time()-start),flush=True)
    np.savez_compressed(target,**result)

def neighbors(args):
    out=stage(args)
    if (out/'neighbor_features.npz').exists():return
    history,contexts,qu,qi,U,I,q=data(args.data_dir,args.baseline,args.full);X=graph(history,U,I);N=len(contexts);ctx=np.tile(np.arange(N),2);pop=np.asarray(X.sum(0)).ravel();length=np.array(list(map(len,history)));down=(1/(1+length/max(length.mean(),1))).astype(np.float32);weighted=X.multiply(down[:,None]).tocsr();wp=np.asarray(weighted.sum(0)).ravel();G=X.T.tocsr();groups={}
    for z,j in enumerate(qi):
        if j>=0 and pop[j]>0:groups.setdefault(j,[]).append(z)
    result={f'weighted_knn_a{alpha}_s{shrink}_{agg}':np.zeros(len(qi),np.float32) for alpha in [.25,.5,.75] for shrink in [0,5] for agg in ['max','mean','last','recent3','recent10','decay']};keys=np.array(list(groups));start=time.time()
    for off in range(0,len(keys),64):
        batch=keys[off:off+64];co=(G[batch]@weighted).tocsr();co.sort_indices()
        for z,j in enumerate(batch):
            lo,hi=co.indptr[z:z+2];nb=co.indices[lo:hi];count=co.data[lo:hi]
            for row in groups[j]:
                h=contexts[ctx[row]];idx=np.searchsorted(nb,h);ok=idx<len(nb);ok[ok]&=nb[idx[ok]]==h[ok];shared=np.zeros(len(h));shared[ok]=count[idx[ok]];shared[h==j]=0
                for alpha in [.25,.5,.75]:
                    similarity=shared/(max(wp[j],1e-8)**(1-alpha)*np.maximum(wp[h],1e-8)**alpha)
                    for shrink in [0,5]:
                        value=similarity if shrink==0 else similarity*shared/(shared+shrink);prefix=f'weighted_knn_a{alpha}_s{shrink}_';result[prefix+'max'][row]=value.max();result[prefix+'mean'][row]=value.mean();result[prefix+'last'][row]=value[-1];result[prefix+'recent3'][row]=value[-3:].mean();result[prefix+'recent10'][row]=value[-10:].mean();w=.9**np.arange(len(h))[::-1];result[prefix+'decay'][row]=value@w/w.sum()
        if off%10000<64:print('Weighted item neighbors',off,len(keys),'seconds',round(time.time()-start),flush=True)
    # IDF-weighted session-neighbor retrieval. All candidate histories are fitted prefixes.
    positions=coo_matrix((np.concatenate([np.arange(len(h)) for h in history]),(np.repeat(np.arange(U),length),np.concatenate(history))),shape=X.shape).tocsc();XC=X.tocsc();assert np.array_equal(XC.indptr,positions.indptr) and np.array_equal(XC.indices,positions.indices)
    idf=np.log1p(U/np.maximum(pop,1)).astype(np.float32);D=X.multiply(idf).tocsr();norm=np.sqrt(np.asarray(D.multiply(D).sum(1)).ravel());recent=[h[-20:] for h in contexts];rr=np.repeat(np.arange(N),list(map(len,recent)));ii=np.concatenate(recent);ww=np.concatenate([(.9**np.arange(len(h))[::-1])*idf[h] for h in recent]).astype(np.float32);C=coo_matrix((ww,(rr,ii)),shape=(N,I)).tocsr();cnorm=np.sqrt(np.asarray(C.multiply(C).sum(1)).ravel());DT=D.T.tocsr()
    for k in [50,200]:
        for name in ['all','tail5','forward5']:result[f'sessionnn_{k}_{name}']=np.zeros(len(qi),np.float32)
    result['sessionnn_support']=np.zeros(len(qi),np.float32)
    def matches(j,neighbors):
        present=np.zeros(len(neighbors),bool);pos=np.zeros(len(neighbors))
        if j<0:return present,pos
        lo,hi=XC.indptr[j:j+2];other=XC.indices[lo:hi];where=np.searchsorted(other,neighbors);ok=where<len(other);ok[ok]&=other[where[ok]]==neighbors[ok];present[ok]=True;pos[ok]=positions.data[lo:hi][where[ok]];return present,pos
    for off in range(0,N,64):
        co=(C[off:off+64]@DT).tocsr()
        for z in range(co.shape[0]):
            row=off+z;lo,hi=co.indptr[z:z+2];nb=co.indices[lo:hi];sim=co.data[lo:hi]/np.maximum(cnorm[row]*norm[nb],1e-8);ok=nb!=qu[row];nb=nb[ok];sim=sim[ok];take=np.argpartition(sim,-200)[-200:] if len(sim)>200 else np.arange(len(sim));take=take[np.lexsort((nb[take],-sim[take]))];nb=nb[take];sim=sim[take];haslast,lastpos=matches(contexts[row][-1],nb)
            for target in [row,N+row]:
                present,pos=matches(qi[target],nb);tail=np.exp(-(length[nb]-1-pos)/5)*present;delta=pos-lastpos;future=present&haslast&(delta>=1)&(delta<=5);forward=np.where(future,1/np.maximum(delta,1),0);result['sessionnn_support'][target]=np.log1p(present.sum())
                for k in [50,200]:
                    w=sim[:k];den=w.sum()+1e-8;result[f'sessionnn_{k}_all'][target]=w@present[:k]/den;result[f'sessionnn_{k}_tail5'][target]=w@tail[:k]/den;result[f'sessionnn_{k}_forward5'][target]=w@forward[:k]/den
        if off%10000<64:print('Session neighbors',off,N,'seconds',round(time.time()-start),flush=True)
    assert all(np.isfinite(v).all() for v in result.values());np.savez_compressed(out/'neighbor_features.npz',**result)

def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','baseline','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--full',action='store_true');ap.add_argument('--worker',choices=['fpmc','uniform','neighbors'],required=True);args=ap.parse_args()
    part=args.out/('full' if args.full else 'validation');part.mkdir(parents=True,exist_ok=True)
    with open(part/(args.worker+'.lock'),'w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.worker=='fpmc':fpmc(args)
        elif args.worker=='uniform':fpmc(args,True)
        else:neighbors(args)
if __name__=='__main__':main()
