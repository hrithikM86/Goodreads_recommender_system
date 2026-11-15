"""FOSSIL variant: last-20 history + personalized last-three transition weights."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,subprocess,ctypes,time
from pathlib import Path
import numpy as np
from task5_improve import data,graph

def run(a):
 out=a.out/('full' if a.full else 'validation');out.mkdir(exist_ok=True,parents=True)
 if (out/'fossil.npz').exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);X=graph(h,U,I);ptr=np.ascontiguousarray(np.r_[0,np.cumsum(list(map(len,h)))],np.int32);seq=np.concatenate(h).astype(np.int32);sp=X.indptr.astype(np.int32);seen=X.indices.astype(np.int32);pop=np.asarray(X.sum(0)).ravel();active=np.flatnonzero(pop>0).astype(np.int32);so=out/'fossil.so';subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_fossil_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.fossil_train.argtypes=[ctypes.c_int]*2+[ip]*5+[ctypes.c_int]*4+[ctypes.c_float]*2+[fp]*5
 k=24;rng=np.random.RandomState(71);A=rng.normal(0,.05,(I,k)).astype('f');B=rng.normal(0,.05,(I,k)).astype('f');wu=np.zeros((U,3),'f');wg=np.array([.5,.25,.1],'f');bias=np.zeros(I,'f');old=0;result={};known=qi>=0;known[known]&=pop[qi[known]]>0;ctx=np.tile(np.arange(len(contexts)),2);start=time.time()
 for epoch in [5,15,35]:
  lib.fossil_train(U,k,ptr,seq,sp,seen,active,len(active),old,epoch-old,71,.015,.03,A,B,wu,wg,bias);old=epoch;A[pop==0]=0;long=np.zeros((len(contexts),k),'f');recent=np.zeros_like(long)
  for z,hist in enumerate(contexts):
   hh=hist[-20:];long[z]=A[hh].sum(0)/np.sqrt(len(hh));last=hist[-3:][::-1];user=qu[z];recent[z]=(wg[:len(last)]+wu[user,:len(last)])@A[last]
  for name,values in [('long',long),('recent',recent),('joint',long+recent)]:
   pred=np.zeros(len(qi),'f');pred[known]=np.einsum('ij,ij->i',values[ctx[known]],B[qi[known]])+(bias[qi[known]] if name=='joint' else 0);result[f'fossil_e{epoch}_{name}']=pred
  print('FOSSIL',a.full,epoch,'seconds',round(time.time()-start),flush=True)
 assert all(np.isfinite(x).all() for x in [A,B,wu,wg,bias]);np.savez_compressed(out/'fossil.npz',**result);np.savez_compressed(out/'fossil_model.npz',A=A,B=B,wu=wu,wg=wg,bias=bias,pop=pop)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for key in ['data-dir','baseline','out']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
