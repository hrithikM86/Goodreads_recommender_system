"""Infer a scalar community reading order from known sequence precedences."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg,LinearOperator
from task5_order import transitions
from task5_improve import data

def ordering(H,pen):
 incoming=np.asarray(H.sum(0)).ravel();outgoing=np.asarray(H.sum(1)).ravel();degree=incoming+outgoing;A=diags(degree+pen)-H-H.T;M=LinearOperator(A.shape,matvec=lambda x:x/(degree+pen));theta,info=cg(A.astype(float),incoming-outgoing,M=M,rtol=1e-6,maxiter=400);assert info==0;return theta

def era_scores(theta,contexts,qi):
 N=len(contexts);features={k:np.zeros(len(qi),'f') for k in ['level','gap_last','gap_mean3','gap_mean','gap_max','distance_trend5','distance_trend20','percentile']}
 for row,h in enumerate(contexts):
  hist=theta[h];last=hist[-1];mean3=hist[-3:].mean();mean=hist.mean();maximum=hist.max();future=[]
  for cap in [5,20]:
   v=hist[-cap:];x=np.arange(len(v),dtype=float);xc=x-x.mean();slope=(xc@(v-v.mean()))/max(xc@xc,1e-9);future.append(v.mean()+slope*(len(v)-x.mean()))
  for side in [0,1]:
   z=row+side*N;j=qi[z]
   if j<0:continue
   value=theta[j];vals=[value,value-last,value-mean3,value-mean,value-maximum,abs(value-future[0]),abs(value-future[1]),np.mean(hist<value)]
   for key,val in zip(features,vals):features[key][z]=val
 return features

def run(a):
 part=a.out/('full' if a.full else 'validation');part.mkdir(exist_ok=True,parents=True)
 if (part/'era.npz').exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);H=None;start=time.time()
 for window,H in transitions(h,I):
  if window==3:break
 result={};params={}
 for pen in [.1,1.,10.]:
  theta=ordering(H,pen);params['theta_'+str(pen)]=theta;result.update({f'era{pen}_{k}':v for k,v in era_scores(theta,contexts,qi).items()});print('COMMUNITY_ORDER',a.full,pen,'seconds',round(time.time()-start),flush=True)
 np.savez_compressed(part/'era_parameters.npz',**params);temp=part/'era.tmp.npz';np.savez_compressed(temp,**result);temp.replace(part/'era.npz')
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','out']:ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
