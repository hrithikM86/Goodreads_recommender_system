"""Metric sequential models from temporal lecture: PRME-style and TransRec."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,subprocess,time
from pathlib import Path
import numpy as np
from task5_improve import data,graph

def run(a):
 out=a.out/('full' if a.full else 'validation');out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.npz').exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);X=graph(h,U,I);ptr=np.ascontiguousarray(np.r_[0,np.cumsum(list(map(len,h)))],np.int32);seq=np.concatenate(h).astype(np.int32);sp=X.indptr.astype(np.int32);seen=X.indices.astype(np.int32);pop=np.asarray(X.sum(0)).ravel();active=np.flatnonzero(pop>0).astype(np.int32);so=out/'metric.so';subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_metric_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.metric_train.argtypes=[ctypes.c_int]*2+[ip]*5+[ctypes.c_int,ip,ctypes.c_int]+[fp]*4+[ctypes.c_int]*4+[ctypes.c_float]*2
 ctx=np.tile(np.arange(len(contexts)),2);last=np.array([x[-1] for x in contexts]);known=qi>=0;known[known]&=pop[qi[known]]>0;result={};start=time.time()
 for kind,name in [(0,'prme'),(1,'transrec')]:
  k=32;rng=np.random.RandomState(92+kind);P=rng.normal(0,.05,(U,k)).astype('f');Q=rng.normal(0,.05,(I,k)).astype('f');globalv=np.zeros(k,'f');bias=np.zeros(I,'f');old=0
  for epoch in [20,60,120]:
   lib.metric_train(U,k,ptr,seq,sp,seen,seq,len(seq),active,len(active),P,Q,globalv,bias,kind,old,epoch-old,92+kind,.02,.01);old=epoch;assert all(np.isfinite(x).all() for x in [P,Q,globalv,bias]);qq=qi[known];us=qu[known];prev=last[ctx[known]];pers=-np.sum((P[us]-Q[qq])**2,axis=1);transition=-np.sum((Q[prev]-Q[qq])**2,axis=1);joint=.5*(pers+transition) if kind==0 else -np.sum((P[us]+Q[prev]+globalv-Q[qq])**2,axis=1)
   source_known=pop[prev]>0;transition[~source_known]=0;joint[~source_known]=0
   for label,val in [('personal',pers),('last',transition),('bias',bias[qq]),('joint',joint+bias[qq])]:
    z=np.zeros(len(qi),'f');z[known]=val;result[f'{name}_e{epoch}_{label}']=z
   print(name,epoch,'seconds',round(time.time()-start),flush=True)
  np.savez_compressed(out/(name+'_model.npz'),P=P,Q=Q,globalv=globalv,bias=bias,pop=pop)
 np.savez_compressed(out/'metrics.npz',**result)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for key in ['data-dir','baseline','out']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
