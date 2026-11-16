"""Training-only reader neighborhoods, including local-time voting for Task 4."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
W=Path('/workspace/work');DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026'
def inputs(task,full):
 if task==5:
  from task5_improve import data,graph
  h,ctx,qu,qi,U,I,_=data(DATA,W/'task5/results',full);X=graph(h,U,I);return X,X.copy(),qu,qi,np.zeros(len(qu))
 df=pd.read_csv(DATA/('tasks_1_2_train.csv' if task==2 else 'tasks_3_4_train.csv'));u,users=pd.factorize(df.user);i,items=pd.factorize(df.item);t=df.timestamp.to_numpy(float)
 if full:
  ix=np.arange(len(df));test=pd.read_csv(DATA/f'task_{task}_test.csv');um=dict(zip(users,range(len(users))));im=dict(zip(items,range(len(items))));qu=np.tile(test.user.map(um).to_numpy(),2);qi=np.r_[test.item1.map(im),test.item2.map(im)];qt=np.tile(test.timestamp.to_numpy(float),2) if task==4 else np.zeros(len(qu))
 else:
  order=np.random.RandomState(42).permutation(len(df));a=int(.9*len(df));ix=order[:a];hold=order[a:];rng=np.random.RandomState(43)
  if task==2:
   neg=i[ix[rng.randint(len(ix),size=len(hold))]];same=neg==i[hold]
   while same.any():neg[same]=i[ix[rng.randint(len(ix),size=same.sum())]];same=neg==i[hold]
  else:
   neg=[]
   for p in i[hold]:
    j=i[ix[rng.randint(len(ix))]]
    while j==p:j=i[ix[rng.randint(len(ix))]]
    neg.append(j)
  qu=np.tile(u[hold],2);qi=np.r_[i[hold],neg];qt=np.tile(t[hold],2)
 shape=(len(users),len(items));X=csr_matrix((np.ones(len(ix),dtype='f'),(u[ix],i[ix])),shape=shape);X.data[:]=1
 T=csr_matrix((t[ix]/86400,(u[ix],i[ix])),shape=shape)
 return X,T,qu,qi,qt/86400

def compute(X,T,qu,qi,qt,temporal):
 pop=np.asarray(X.sum(0)).ravel();cnt=np.asarray(X.sum(1)).ravel();idf=1/np.sqrt(pop+10);Z=X.multiply(idf).tocsr();norm=np.sqrt(np.asarray(Z.multiply(Z).sum(1)).ravel());Z=Z.multiply(1/np.maximum(norm,1e-8)[:,None]).tocsr();ZT=Z.T.tocsr();C=X.tocsc();TC=T.tocsc();assert np.array_equal(C.indices,TC.indices) and np.array_equal(C.indptr,TC.indptr)
 names=[f'readers{k}_{s}' for k in [50,200,800] for s in ['sum','max','count']]
 if temporal:names += [f'readers{k}_time{h}' for k in [50,200,800] for h in [30,180,730]]
 result={k:np.zeros(len(qu),dtype='f') for k in names};order=np.argsort(qu,kind='stable');keys,starts=np.unique(qu[order],return_index=True);ends=np.r_[starts[1:],len(order)];start=time.time()
 for offset in range(0,len(keys),64):
  block=keys[offset:offset+64];sim=(Z[block]@ZT).tocsr()
  for local,user in enumerate(block):
   lo,hi=sim.indptr[local:local+2];ids=sim.indices[lo:hi];val=sim.data[lo:hi];good=(ids!=user)&(val>0);ids=ids[good];val=val[good]
   top=np.argpartition(val,-min(800,len(val)))[-800:] if len(val) else np.array([],int);top=top[np.argsort(val[top])[::-1]];ids=ids[top];val=val[top];rank=np.full(X.shape[0],100000,dtype=np.int32);weight=np.zeros(X.shape[0],dtype='f');rank[ids]=np.arange(len(ids));weight[ids]=val
   pos=offset+local
   for q in order[starts[pos]:ends[pos]]:
    if qi[q]<0:continue
    a,b=C.indptr[qi[q]:qi[q]+2];readers=C.indices[a:b];ranks=rank[readers];w=weight[readers]
    for k in [50,200,800]:
     mask=ranks<k;v=w[mask];result[f'readers{k}_sum'][q]=v.sum();result[f'readers{k}_max'][q]=v.max(initial=0);result[f'readers{k}_count'][q]=mask.sum()
     if temporal:
      times=TC.data[a:b][mask];valid=(times>=0)&(times<4102444800/86400)&(qt[q]>=0)&(qt[q]<4102444800/86400);dist=np.abs(times[valid]-qt[q])
      for h in [30,180,730]:result[f'readers{k}_time{h}'][q]=np.dot(v[valid],np.exp(-dist/h))
  if offset%2048==0:print('READERS',offset,len(keys),'seconds',round(time.time()-start),flush=True)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);ap.add_argument('--full',action='store_true');a=ap.parse_args();out=W/f'task{a.task}_round4';out.mkdir(exist_ok=True);path=out/('readers_full.npz' if a.full else 'readers_validation.npz');X,T,qu,qi,qt=inputs(a.task,a.full);f=compute(X,T,qu,qi,qt,a.task==4);assert all(np.isfinite(v).all() for v in f.values());np.savez_compressed(path,**f);print('SAVED',path,flush=True)
if __name__=='__main__':main()
