"""Positive-only Task 1: reproducible SVD/neighborhood bank plus SVD++."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,hashlib,time,ctypes,subprocess,fcntl
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix,save_npz

WEIGHTS=np.array([.1541547817989482,.16261770829034805,.1196487441069676,.1062846995076872,.31124904671666187,.1460450195793872])
def clip(p):return np.clip(p,1,5)
def dataset(data,full):
    p=pd.read_csv(data/'tasks_1_2_train.csv');p=p[p.rating>0].reset_index(drop=True);u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);order=np.random.RandomState(42).permutation(len(p));a=int(.9*len(p));b=int(.95*len(p));ix=np.arange(len(p)) if full else order[:a];q=pd.read_csv(data/'task_1_test.csv') if full else p.iloc[order[a:]].reset_index(drop=True);qu=pd.Index(users).get_indexer(q.user);qi=pd.Index(items).get_indexer(q.item)
    return p,p.iloc[ix],q,u[ix],i[ix],qu,qi,users,items,b-a
def guard(a):
    part=a.out/('full' if a.full else 'validation');part.mkdir(parents=True,exist_ok=True);m={'full':a.full,'inputs':{f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['tasks_1_2_train.csv','task_1_test.csv']}};path=part/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==m
    temp=part/f'manifest_{os.getpid()}.json';temp.write_text(json.dumps(m,indent=2));temp.replace(path);return part
def library(a):
    out=guard(a);os.environ.setdefault('SURPRISE_DATA_FOLDER',str(a.out/'surprise'))
    if (out/'library.npz').exists():return
    p,tr,q,u,i,qu,qi,users,items,n=dataset(a.data_dir,a.full);U,I=len(users),len(items);order=np.argsort(u,kind='stable');uu=u[order];ii=np.ascontiguousarray(i[order],np.int32);y=np.ascontiguousarray(tr.rating.to_numpy(float)[order],np.float32);count=np.bincount(uu,minlength=U);ic=np.ascontiguousarray(np.bincount(ii,minlength=I),np.int32);ptr=np.ascontiguousarray(np.r_[0,np.cumsum(count)],np.int32);H=coo_matrix((np.ones(len(y)),(uu,ii)),shape=(U,I)).tocsr();assert np.max(H.data)==1;mu=float(y.mean());start=time.time();result={};so=out/'svdpp.so';subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task1_svdpp_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.svdpp_train.argtypes=[ctypes.c_int]*2+[ip,ip,fp,ip,ctypes.c_float]+[fp]*5+[ctypes.c_int]*3+[ctypes.c_float]*2+[ctypes.c_int]
    for k,reg,seed in [(24,.1,42),(48,.15,17)]:
        rng=np.random.RandomState(seed);P=rng.normal(0,.05,(U,k)).astype(np.float32);Q=rng.normal(0,.05,(I,k)).astype(np.float32);Y=np.zeros_like(Q);bu=np.zeros(U,np.float32);bi=np.zeros(I,np.float32);last=0
        for epoch in [20,50,100]:
            lib.svdpp_train(U,k,ptr,ii,y,ic,mu,P,Q,Y,bu,bi,last,epoch-last,seed,.007,reg,1);last=epoch;effective=P+(H@Y)/np.sqrt(np.maximum(count,1))[:,None];effective[count==0]=0;Q[ic==0]=0;pred=np.full(len(q),mu);knownu=qu>=0;knowni=qi>=0;both=knownu&knowni;pred[knownu]+=bu[qu[knownu]];pred[knowni]+=bi[qi[knowni]];pred[both]+=np.einsum('ij,ij->i',effective[qu[both]],Q[qi[both]]);name=f'svdpp{k}_e{epoch}';result[name]=clip(pred);assert np.isfinite(pred).all();np.savez_compressed(out/(name+'_model.npz'),effective=effective,Q=Q,bu=bu,bi=bi,mu=mu,cu=count,ci=ic);print('Fast SVD++',k,epoch,'seconds',round(time.time()-start),flush=True)
    if not a.full:
        z=np.load(a.legacy/'task1_round1_predictions.npz')
        for k in [16,32,64]:result[f'svd{k}']=np.r_[z[f'v_svd{k}_reg.1'],z[f't_svd{k}_reg.1']]
        split=np.load(a.legacy/'task1_split.npz');assert np.array_equal(split['order'],np.random.RandomState(42).permutation(len(p)));assert np.array_equal(q.rating.to_numpy(),np.r_[split['validation_y'],split['evaluation_y']])
    np.savez_compressed(out/'library.npz',**result)
def neighbors(a):
    out=guard(a)
    if (out/'neighbors.npz').exists():return
    p,tr,q,u,i,qu,qi,users,items,n=dataset(a.data_dir,a.full);U,I=len(users),len(items);y=tr.rating.to_numpy(float);mu=y.mean();cu=np.bincount(u,minlength=U);ci=np.bincount(i,minlength=I);bu=np.zeros(U);bi=np.zeros(I)
    for _ in range(20):bu=np.bincount(u,weights=y-mu-bi[i],minlength=U)/(cu+10);bi=np.bincount(i,weights=y-mu-bu[u],minlength=I)/(ci+10)
    residual=y-mu-bu[u]-bi[i];base=np.full(len(q),mu);knownu=qu>=0;knowni=qi>=0;base[knownu]+=bu[qu[knownu]];base[knowni]+=bi[qi[knowni]];X=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=(U,I)).tocsr();R=coo_matrix((residual,(u,i)),shape=(U,I)).tocsr();G=X.T.tocsr();RT=R.T.tocsr();RC=R.tocsc();norm=np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel());settings=[(ratio,reg) for ratio in [.25,1,4] for reg in [10,50]];result={f'joint_r{r}_p{pen}':base.copy() for r,pen in settings};result.update({f'cos{k}_{power}':base.copy() for k,power in [(10,1),(20,2),(10,2)]});groups={}
    for z,j in enumerate(qi):
        if j>=0 and qu[z]>=0:groups.setdefault(j,[]).append(z)
    rows=[[] for _ in settings];cols=[[] for _ in settings];values=[[] for _ in settings];start=time.time();keys=np.array(list(groups))
    for offset in range(0,len(keys),64):
        batch=keys[offset:offset+64];co=(G[batch]@X).toarray();dot=(RT[batch]@R).toarray()
        for z,j in enumerate(batch):
            sim=co[z]/np.sqrt(np.maximum(ci[j]*ci,1));sim[j]=0;corr=np.abs(dot[z])/np.maximum(norm[j]*norm,1e-8)*co[z]/(co[z]+5);corr[j]=0;nb1=np.argpartition(sim,-32)[-32:];nb2=np.argpartition(corr,-32)[-32:];nb=np.unique(np.r_[nb1[sim[nb1]>0],nb2[corr[nb2]>0]]);nb=nb[nb!=j];queries=np.array(groups[j]);lo,hi=RC.indptr[j:j+2];us=RC.indices[lo:hi];yy=RC.data[lo:hi]
            for row in queries:
                lo,hi=R.indptr[qu[row]:qu[row]+2];w=sim[R.indices[lo:hi]];order=np.argsort(w)[::-1][:20];w=w[order];err=R.data[lo:hi][order]
                for k,power in [(10,1),(20,2),(10,2)]:ww=w[:k]**power;result[f'cos{k}_{power}'][row]+=(ww@err[:k])/(ww.sum()+1e-12)
            if not len(nb):continue
            A=np.c_[X[us][:,nb].toarray(),R[us][:,nb].toarray()];B=np.c_[X[qu[queries]][:,nb].toarray(),R[qu[queries]][:,nb].toarray()]
            for ratio in [.25,1,4]:
                scale=np.r_[np.ones(len(nb)),np.full(len(nb),1/np.sqrt(ratio))];scaled=A*scale;dual=len(us)<scaled.shape[1];eig,V=np.linalg.eigh(scaled@scaled.T if dual else scaled.T@scaled);left=scaled.T@V if dual else V;rhs=V.T@(yy if dual else scaled.T@yy)
                for index,(r,pen) in enumerate(settings):
                    if r!=ratio:continue
                    w=((left@(rhs/(np.maximum(eig,0)+pen)))*scale).astype(np.float32);result[f'joint_r{r}_p{pen}'][queries]+=B@w
                    if a.full:rows[index].extend(np.r_[nb,nb+I]);cols[index].extend([j]*len(w));values[index].extend(w)
        if offset%5000<64:print('Task1 neighborhoods',offset,len(keys),'seconds',round(time.time()-start),flush=True)
    for j,(r,pen) in enumerate(settings):
        if a.full:save_npz(out/f'joint_r{r}_p{pen}.npz',coo_matrix((values[j],(rows[j],cols[j])),shape=(2*I,I)).tocsr())
    np.savez_compressed(out/'neighbors.npz',**{k:clip(v) for k,v in result.items()});np.savez_compressed(out/'bias.npz',mu=mu,bu=bu,bi=bi,cu=cu,ci=ci)
def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','legacy','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--full',action='store_true');ap.add_argument('--worker',choices=['library','neighbors'],required=True);a=ap.parse_args();part=a.out/('full' if a.full else 'validation');part.mkdir(parents=True,exist_ok=True)
    with open(part/(a.worker+'.lock'),'w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);(library if a.worker=='library' else neighbors)(a)
if __name__=='__main__':main()
