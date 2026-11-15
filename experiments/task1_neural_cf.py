"""Five-class neural CF, with categorical CE and optional expected-rating MSE."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,subprocess,time,json,hashlib
from pathlib import Path
import numpy as np
from task1_improve import dataset

def probabilities(model,qu,qi):
 P,Q,bu,bi,W,b=[model[x] for x in ['P','Q','bu','bi','W','b']];out=np.empty((len(qu),5),'f')
 for ix in np.array_split(np.arange(len(qu)),max(1,(len(qu)+8191)//8192)):
  ku=qu[ix]>=0;ki=qi[ix]>=0;h=np.zeros((len(ix),P.shape[1]),'f');ok=ku&ki;h[ok]=np.tanh(P[qu[ix[ok]]]*Q[qi[ix[ok]]]);logits=h@W.T+b;logits[ku]+=bu[qu[ix[ku]]];logits[ki]+=bi[qi[ix[ki]]];logits-=logits.max(1,keepdims=True);prob=np.exp(logits);out[ix]=prob/prob.sum(1,keepdims=True)
 assert np.isfinite(out).all() and np.allclose(out.sum(1),1,atol=1e-6);return out

def run(a):
 out=a.out/('full' if a.full else 'validation');out.mkdir(exist_ok=True,parents=True);target=out/'cf_probabilities.npz'
 if target.exists():return
 p,tr,q,u,i,qu,qi,users,items,n=dataset(a.data_dir,a.full);u=np.ascontiguousarray(u,np.int32);i=np.ascontiguousarray(i,np.int32);y=np.ascontiguousarray(tr.rating.to_numpy()-1,np.int32);cu=np.ascontiguousarray(np.bincount(u,minlength=len(users)),np.int32);ci=np.ascontiguousarray(np.bincount(i,minlength=len(items)),np.int32);warm=dict(np.load(a.previous/('full' if a.full else 'validation')/'svdpp24_e50_model.npz'));k=24;globalp=np.bincount(y,minlength=5)/len(y);userfreq=np.zeros((len(users),5));itemfreq=np.zeros((len(items),5));np.add.at(userfreq,(u,y),1);np.add.at(itemfreq,(i,y),1)
 so=out/'neural.so';subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task1_neural_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.neural_train.argtypes=[ctypes.c_int]*2+[ip]*5+[fp]*12+[ctypes.c_int]*3+[ctypes.c_float];result={};start=time.time()
 for lam in [0.,2.]:
  P=np.array(warm['effective'],dtype='f',order='C',copy=True);Q=np.array(warm['Q'],dtype='f',order='C',copy=True);bu=(.5*np.log(((userfreq+20*globalp)/(cu[:,None]+20))/globalp)).astype('f');bi=(.5*np.log(((itemfreq+20*globalp)/(ci[:,None]+20))/globalp)).astype('f');W=np.tile((np.arange(5)-2)[:,None]*.3,(1,k)).astype('f');b=np.log(globalp).astype('f');parameters=[P,Q,bu,bi,W,b];history=[np.zeros_like(z) for z in parameters];last=0;name='ce' if lam==0 else 'ce_mse'
  for epoch in [3,8,15,25]:
   lib.neural_train(len(u),k,u,i,y,cu,ci,*parameters,*history,last,epoch-last,133,lam);last=epoch;assert all(np.isfinite(z).all() for z in parameters);result[name+'_e'+str(epoch)]=probabilities(dict(zip(['P','Q','bu','bi','W','b'],parameters)),qu,qi);print('NEURAL_CF',a.full,name,epoch,'seconds',round(time.time()-start),flush=True)
  np.savez_compressed(out/(name+'_model.npz'),**dict(zip(['P','Q','bu','bi','W','b'],parameters)))
 temp=out/'cf_probabilities.tmp.npz';np.savez_compressed(temp,**result);temp.replace(target);(out/'manifest.json').write_text(json.dumps({'full':a.full,'positive_train_rows':len(u),'selection_rows':n,'inputs':{f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['tasks_1_2_train.csv','task_1_test.csv']}},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for key in ['data-dir','previous','out']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
