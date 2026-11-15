"""EASE closed-form implicit neighborhoods and positional-context scoring."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse import coo_matrix
from task5_improve import data,graph

def run(a):
 out=a.out/('full' if a.full else 'validation');out.mkdir(exist_ok=True,parents=True)
 if (out/'ease.npz').exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);X=graph(h,U,I);pop=np.asarray(X.sum(0)).ravel();top=np.argsort(pop)[-4000:];mapping=np.full(I,-1);mapping[top]=np.arange(len(top));Z=X[:,top].astype(float);G=(Z.T@Z).toarray();ct=[];N=len(contexts)
 for mode in ['whole','last','last3','decay']:
  rr=[];cc=[];vv=[]
  for z,hist in enumerate(contexts):
   hist=hist[-1:] if mode=='last' else (hist[-3:] if mode=='last3' else hist);ids=mapping[hist];weight=.9**np.arange(len(hist))[::-1] if mode=='decay' else np.ones(len(hist));ok=ids>=0;rr.extend([z]*int(ok.sum()));cc.extend(ids[ok]);vv.extend(weight[ok])
  ct.append(coo_matrix((vv,(rr,cc)),shape=(N,len(top))).tocsr())
 ctx=np.tile(np.arange(N),2);target=np.full(len(qi),-1);valid=qi>=0;target[valid]=mapping[qi[valid]];valid=np.flatnonzero(target>=0);result={};start=time.time()
 for reg in [50,300,1000]:
  A=G.copy();A.flat[::len(top)+1]+=reg;P=cho_solve(cho_factor(A),np.eye(len(top)));B=(-P/np.diag(P)[None,:]).astype('f');np.fill_diagonal(B,0)
  for mode,C in zip(['whole','last','last3','decay'],ct):
   pred=np.zeros(len(qi),'f')
   for ix in np.array_split(valid,max(1,(len(valid)+511)//512)):pred[ix]=np.asarray(C[ctx[ix]].multiply(B[:,target[ix]].T).sum(1)).ravel()
   result[f'ease{reg}_{mode}']=pred
  print('EASE',a.full,reg,'seconds',round(time.time()-start),flush=True)
 np.savez_compressed(out/'ease.npz',**result,covered=(target>=0).astype('f'))
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for key in ['data-dir','baseline','out']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
