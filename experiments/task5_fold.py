"""Query-history recency adaptation with frozen FPMC preference item factors."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,subprocess,time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from task5_improve import data

def run(a):
 part=a.out/('full' if a.full else 'validation');part.mkdir(exist_ok=True,parents=True)
 if (part/'fold.npz').exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);N=len(contexts);w=dict(np.load(a.previous/('full' if a.full else 'validation')/'fpmc64_useruniform_model.npz'));P=w['P'];Q=w['Q'];bias=w['bias'];pop=w['pop'];active=np.flatnonzero(pop>0).astype('i');known=(pop>0).astype('i');lengths=np.array(list(map(len,contexts)));seq=np.concatenate(contexts).astype('i');ptr=np.ascontiguousarray(np.r_[0,np.cumsum(lengths)],np.int32);X=coo_matrix((np.ones(len(seq)),(np.repeat(np.arange(N),lengths),seq)),shape=(N,I)).tocsr();X.sort_indices();sp=X.indptr.astype('i');seen=X.indices.astype('i');users=qu[:N].astype('i');ctx=np.tile(np.arange(N),2);valid=qi>=0;valid[valid]&=pop[qi[valid]]>0;k=P.shape[1];normq=np.linalg.norm(Q,axis=1);so=part/'fold.so';subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_fold_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.fold_users.argtypes=[ctypes.c_int]*2+[ip]*6+[ctypes.c_int,ip]+[fp]*4+[ctypes.c_int]*3;result={};sample={};start=time.time()
 for cap in [3,10,0]:
  factors=P[users].copy();old=0
  for step in [25,100]:
   lib.fold_users(N,k,ptr,seq,sp,seen,users,active,len(active),known,P,Q,bias,factors,cap,old,step-old);old=step;score=np.zeros(len(qi),'f');score[valid]=np.einsum('ij,ij->i',factors[ctx[valid]],Q[qi[valid]]);result[f'fold{cap}_e{step}_score']=score;cos=np.zeros(len(qi),'f');cos[valid]=score[valid]/np.maximum(np.linalg.norm(factors,axis=1)[ctx[valid]]*normq[qi[valid]],1e-9);result[f'fold{cap}_e{step}_cos']=cos
   if step==100:
    ix=np.random.RandomState(91).choice(np.flatnonzero(valid),256,replace=False);sample[f'fold{cap}_rows']=ix;sample[f'fold{cap}_user']=factors[ctx[ix]];sample[f'fold{cap}_item']=Q[qi[ix]];sample[f'fold{cap}_score']=score[ix]
   print('QUERY_FOLD',a.full,cap,step,'seconds',round(time.time()-start),flush=True)
 temp=part/'fold.tmp.npz';np.savez_compressed(temp,**result);temp.replace(part/'fold.npz');np.savez_compressed(part/'fold_replay_sample.npz',**sample)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','previous','out']:ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
