"""User-balanced skip transitions and exact two-hop sequence diffusion."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,ctypes,subprocess,json,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
from task5_improve import data

def transitions(history,I):
 lengths=np.array(list(map(len,history)));users=np.repeat(np.arange(len(history)),lengths);seq=np.concatenate(history);down=1/np.sqrt(lengths);H=csr_matrix((I,I),dtype=np.float32)
 for distance in range(1,11):
  ok=users[:-distance]==users[distance:];s=seq[:-distance][ok];t=seq[distance:][ok];w=(down[users[:-distance][ok]]/distance).astype('f');H=H+coo_matrix((w,(s,t)),shape=(I,I)).tocsr();H.sort_indices()
  if distance in [1,3,10]:yield distance,H.copy()

def score_graph(H,contexts,qi):
 N=len(contexts);recent=[h[-20:][::-1] for h in contexts];lengths=np.array(list(map(len,recent)));owner=np.repeat(np.arange(N),lengths);src=np.concatenate(recent);pos=np.concatenate([np.arange(n) for n in lengths]);weight=.8**pos;rowsum=np.asarray(H.sum(1)).ravel();pop=np.asarray(H.sum(0)).ravel();out={};last=np.array([h[-1] for h in contexts]);P=H.multiply((1/np.maximum(rowsum,1e-8))[:,None]).tocsr()
 for side in [0,1]:
  target=np.repeat(qi[side*N:(side+1)*N],lengths);good=target>=0;fwd=np.zeros(len(src));rev=np.zeros(len(src));fwd[good]=np.asarray(H[src[good],target[good]]).ravel();rev[good]=np.asarray(H[target[good],src[good]]).ravel();p=fwd/np.maximum(rowsum[src],1e-8);r=np.zeros_like(rev);r[good]=rev[good]/np.maximum(rowsum[target[good]],1e-8);lift=np.zeros_like(p);lift[good]=p[good]/np.sqrt(np.maximum(pop[target[good]]/max(pop.sum(),1),1e-12));den=np.bincount(owner,weights=weight,minlength=N)
  vals={'last':np.bincount(owner[pos==0],weights=p[pos==0],minlength=N),'recent3':np.bincount(owner[pos<3],weights=p[pos<3],minlength=N)/np.minimum(lengths,3),'decay':np.bincount(owner,weights=p*weight,minlength=N)/den,'reverse_decay':np.bincount(owner,weights=r*weight,minlength=N)/den,'order_evidence':np.bincount(owner,weights=weight*np.log((fwd+.1)/(rev+.1)),minlength=N)/den,'lift_decay':np.bincount(owner,weights=lift*weight,minlength=N)/den};mx=np.zeros(N);np.maximum.at(mx,owner,fwd);vals['count_max']=np.log1p(mx)
  for name,v in vals.items():
   if name not in out:out[name]=np.zeros(len(qi),'f')
   out[name][side*N:(side+1)*N]=v
 return out,P

def diffusion(P,contexts,qi,so):
 lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.paths.argtypes=[ctypes.c_int]+[ip]*4+[fp]+[ip]*2+[fp]*2;C=P.tocsc();rp=P.indptr.astype('i');ri=P.indices.astype('i');rv=P.data.astype('f');cp=C.indptr.astype('i');ci=C.indices.astype('i');cv=C.data.astype('f');N=len(contexts);recent=[h[-3:][::-1] for h in contexts];lens=np.array(list(map(len,recent)));src=np.concatenate(recent).astype('i');owner=np.repeat(np.arange(N),lens);out={k:np.zeros(len(qi),'f') for k in ['twohop_last','twohop_recent3','twohop_reverse']};first=np.r_[0,np.cumsum(lens)[:-1]]
 for side in [0,1]:
  dst=np.repeat(qi[side*N:(side+1)*N],lens).astype('i');value=np.zeros(len(src),'f');lib.paths(len(src),src,dst,rp,ri,rv,cp,ci,cv,value);out['twohop_last'][side*N:(side+1)*N]=value[first];out['twohop_recent3'][side*N:(side+1)*N]=np.bincount(owner,weights=value,minlength=N)/lens;back=np.zeros(len(src),'f');lib.paths(len(src),dst,src,rp,ri,rv,cp,ci,cv,back);out['twohop_reverse'][side*N:(side+1)*N]=np.bincount(owner,weights=back,minlength=N)/lens
 return out

def run(a):
 part=a.out/('full' if a.full else 'validation');part.mkdir(exist_ok=True,parents=True);target=part/'order.npz'
 if target.exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);source=Path(__file__).with_name('task5_order_kernel.cpp');so=part/'paths.so';subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(source),'-o',str(so)],check=True);result={};start=time.time()
 for window,H in transitions(h,I):
  vals,P=score_graph(H,contexts,qi);result.update({f'skip{window}_{k}':v for k,v in vals.items()})
  if window==3:result.update(diffusion(P,contexts,qi,so))
  print('ORDER_GRAPH',a.full,window,'edges',H.nnz,'seconds',round(time.time()-start),flush=True)
 temp=part/'order.tmp.npz';np.savez_compressed(temp,**result);temp.replace(target);(part/'order_manifest.json').write_text(json.dumps({'full':a.full,'sequence_only':True,'all_ratings_retained':True,'inputs':{f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['task_5_train.csv','task_5_test.csv']}},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','out']:ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
