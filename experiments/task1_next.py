"""Expanded positive-rating collaborative models; same frozen split as accepted."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,subprocess,time,json,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from task1_improve import dataset,clip

def run(a):
 out=a.out/('full' if a.full else 'validation');out.mkdir(parents=True,exist_ok=True)
 p,tr,q,u,i,qu,qi,users,items,n=dataset(a.data_dir,a.full);U,I=len(users),len(items);y=tr.rating.to_numpy(float);mu=y.mean();cu=np.bincount(u,minlength=U);ci=np.bincount(i,minlength=I);bu=np.zeros(U);bi=np.zeros(I)
 for _ in range(20):bu=np.bincount(u,weights=y-mu-bi[i],minlength=U)/(cu+10);bi=np.bincount(i,weights=y-mu-bu[u],minlength=I)/(ci+10)
 X=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=(U,I)).tocsr();R=coo_matrix((y-mu-bu[u]-bi[i],(u,i)),shape=(U,I)).tocsr();base=np.full(len(q),mu);ku=qu>=0;ki=qi>=0;base[ku]+=bu[qu[ku]];base[ki]+=bi[qi[ki]]
 if a.worker in ['svdpp','svdppstrong','itemspp']:
  if a.worker=='itemspp':u,i,qu,qi,U,I,cu,ci=i,u,qi,qu,I,U,ci,cu;X=X.T.tocsr();ku=qu>=0;ki=qi>=0
  filename=a.worker+'.npz'
  if (out/filename).exists():return
  order=np.argsort(u,kind='stable');ii=np.ascontiguousarray(i[order],np.int32);yy=np.ascontiguousarray(y[order],np.float32);ptr=np.ascontiguousarray(np.r_[0,np.cumsum(cu)],np.int32);ic=np.ascontiguousarray(ci,np.int32);so=out/'svd.so';subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task1_svdpp_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.svdpp_train.argtypes=[ctypes.c_int]*2+[ip,ip,fp,ip,ctypes.c_float]+[fp]*5+[ctypes.c_int]*3+[ctypes.c_float]*2+[ctypes.c_int]
  k=96 if a.worker=='svdppstrong' else (32 if a.worker=='itemspp' else 64);reg=.3 if a.worker=='svdppstrong' else (.1 if a.worker=='itemspp' else .04);rng=np.random.RandomState(71);P=rng.normal(0,.05,(U,k)).astype('f');Q=rng.normal(0,.05,(I,k)).astype('f');Y=np.zeros_like(Q);b=np.zeros(U,'f');c=np.zeros(I,'f');last=0;result={};start=time.time()
  for epoch in ([30,70,140] if a.worker=='svdppstrong' else ([20,50,100] if a.worker=='itemspp' else [50,100,180])):
   lib.svdpp_train(U,k,ptr,ii,yy,ic,mu,P,Q,Y,b,c,last,epoch-last,71,.007,reg,1);last=epoch;effective=P+(X@Y)/np.sqrt(np.maximum(cu,1))[:,None];effective[cu==0]=0;Q[ci==0]=0;pred=np.full(len(q),mu);pred[ku]+=b[qu[ku]];pred[ki]+=c[qi[ki]];ok=ku&ki;pred[ok]+=np.einsum('ij,ij->i',effective[qu[ok]],Q[qi[ok]]);result[('svdppstrong96_e' if a.worker=='svdppstrong' else ('itemspp32_e' if a.worker=='itemspp' else 'svdpp64low_e'))+str(epoch)]=clip(pred);print('svdpp',epoch,round(time.time()-start),flush=True)
  np.savez_compressed(out/filename,**result);np.savez_compressed(out/(a.worker+'_final.npz'),effective=effective,Q=Q,bu=b,bi=c,mu=mu);return
 if (out/'neighbors.npz').exists():return
 result={};start=time.time()
 for axis in ['item','user']:
  A=X if axis=='item' else X.T.tocsr();E=R if axis=='item' else R.T.tocsr();target=qi if axis=='item' else qu;context=qu if axis=='item' else qi;G=A.T.tocsr();ET=E.T.tocsr();EC=E.tocsc();count=np.asarray(A.sum(0)).ravel();norm=np.sqrt(np.asarray(E.multiply(E).sum(0)).ravel());groups={}
  for z,j in enumerate(target):
   if j>=0 and context[z]>=0:groups.setdefault(j,[]).append(z)
  keys=np.array(list(groups));settings=[(r,pen) for r in [1,4] for pen in [20,100]]
  for r,pen in settings:result[f'{axis}_ridge_r{r}_p{pen}']=base.copy()
  for power in [1,2]:result[f'{axis}_pearson{power}']=base.copy()
  for off in range(0,len(keys),32):
   batch=keys[off:off+32];co=(G[batch]@A).toarray();dot=(ET[batch]@E).toarray()
   for z,j in enumerate(batch):
    sim=co[z]/np.sqrt(np.maximum(count[j]*count,1));sim[j]=0;corr=dot[z]/np.maximum(norm[j]*norm,1e-8)*co[z]/(co[z]+10);corr[j]=0;top=lambda arr:np.argpartition(arr,-min(64,len(arr)))[-min(64,len(arr)):];nb=np.unique(np.r_[top(sim),top(abs(corr))]);nb=nb[(nb!=j)&(sim[nb]>0)];queries=np.array(groups[j]);lo,hi=EC.indptr[j:j+2];trainrows=EC.indices[lo:hi];yy=EC.data[lo:hi]
    for row in queries:
     lo,hi=E.indptr[context[row]:context[row]+2];ids=E.indices[lo:hi];w=corr[ids];order=np.argsort(abs(w))[::-1][:40];w=w[order];err=E.data[lo:hi][order]
     for power in [1,2]:ww=np.sign(w)*abs(w)**power;result[f'{axis}_pearson{power}'][row]+=(ww@err)/(abs(ww).sum()+1e-12)
    if not len(nb):continue
    T=np.c_[A[trainrows][:,nb].toarray(),E[trainrows][:,nb].toarray()];B=np.c_[A[context[queries]][:,nb].toarray(),E[context[queries]][:,nb].toarray()]
    for ratio in [1,4]:
     scale=np.r_[np.ones(len(nb)),np.full(len(nb),1/np.sqrt(ratio))];S=T*scale;dual=len(trainrows)<S.shape[1];eig,V=np.linalg.eigh(S@S.T if dual else S.T@S);left=S.T@V if dual else V;rhs=V.T@(yy if dual else S.T@yy)
     for pen in [20,100]:w=(left@(rhs/(np.maximum(eig,0)+pen)))*scale;result[f'{axis}_ridge_r{ratio}_p{pen}'][queries]+=B@w
   if off%5000<32:print(axis,off,len(keys),'seconds',round(time.time()-start),flush=True)
 np.savez_compressed(out/'neighbors.npz',**{k:clip(v) for k,v in result.items()})
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for key in ['data-dir','out']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--worker',choices=['svdpp','svdppstrong','itemspp','neighbors'],required=True);ap.add_argument('--full',action='store_true');run(ap.parse_args())
