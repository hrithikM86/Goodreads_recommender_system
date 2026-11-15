"""Candidate-specific neural attention, warm-started from fitted-prefix FPMC."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,subprocess,time,json,hashlib
from pathlib import Path
import numpy as np
from task5_improve import data,graph

def scores(model,contexts,qu,qi,personal,query):
 E,V,bias,pos,pop=[model[k] for k in ['E','V','bias','position','pop']];N=len(contexts);result={k:np.zeros(len(qi),'f') for k in ['attention','joint','last','recent3','max10']};k=E.shape[1]
 # Each candidate gets its own normalized relevance weights over the same history.
 for side in [0,1]:
  for row,h in enumerate(contexts):
   index=side*N+row;j=qi[index]
   if j<0 or pop[j]==0:continue
   hh=h[-10:][::-1];values=E[hh]@V[j];logits=values/np.sqrt(k)+pos[:len(hh)];weight=np.exp(logits-logits.max());weight/=weight.sum();value=weight@values;result['attention'][index]=value;result['joint'][index]=value+bias[j]+personal[qu[index]]@query[j];result['last'][index]=values[0];result['recent3'][index]=values[:3].mean();result['max10'][index]=values.max()
 return result

def run(a):
 part=a.out/('full' if a.full else 'validation');part.mkdir(exist_ok=True,parents=True);target=part/'attention.npz'
 if target.exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);X=graph(h,U,I);ptr=np.ascontiguousarray(np.r_[0,np.cumsum(list(map(len,h)))],np.int32);seq=np.concatenate(h).astype('i');sp=X.indptr.astype('i');seen=X.indices.astype('i');warm=dict(np.load(a.previous/('full' if a.full else 'validation')/'fpmc32_uniform_model.npz'));P=warm['P'];Q=warm['Q'];E=warm['A'];V=warm['B'];bias=warm['bias'];pop=np.asarray(X.sum(0)).ravel();E[(pop==0)|(warm['source_count']==0)]=0;V[pop==0]=0;active=np.flatnonzero(pop>0).astype('i');position=(-np.log1p(np.arange(10))).astype('f');k=E.shape[1];so=part/'attention.so';subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_attention_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.train_attention.argtypes=[ctypes.c_int]*2+[ip]*5+[ctypes.c_int]+[fp]*6+[ctypes.c_int]*3+[ctypes.c_float];old=0;result={};start=time.time()
 for epoch in [0,3,10,20]:
  if epoch:lib.train_attention(U,k,ptr,seq,sp,seen,active,len(active),P,Q,E,V,bias,position,old,epoch-old,53,.005)
  old=epoch;assert all(np.isfinite(x).all() for x in [E,V,bias,position]);model=dict(E=E,V=V,bias=bias,position=position,pop=pop);values=scores(model,contexts,qu,qi,P,Q);result.update({f'attention_e{epoch}_{name}':v for name,v in values.items()});print('ATTENTION',a.full,epoch,'seconds',round(time.time()-start),flush=True)
 np.savez_compressed(part/'attention_model.npz',E=E,V=V,bias=bias,position=position,pop=pop);temp=part/'attention.tmp.npz';np.savez_compressed(temp,**result);temp.replace(target)
 (part/'attention_manifest.json').write_text(json.dumps({'full':a.full,'history_limit':10,'seed':53,'checkpoints':[0,3,10,20],'learning_rate':.005,'inputs':{f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['task_5_train.csv','task_5_test.csv']}},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','previous','out']:ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
