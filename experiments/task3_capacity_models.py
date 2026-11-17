"""Higher-capacity temporal SVD++ workers, same audited native SGD kernel.
New holdout is nested INSIDE original base-training90%, disjoint from all
cached meta targets. Only scoring arrays/configuration are persisted;
large experimental parameter snapshots are round-tripped in memory.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import ctypes,io,json,time,subprocess,gc
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix
from task3_round2 import time_arrays
from task3_optimize import dataset,clip,mse
W=Path('/workspace/work');DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026';OUT=W/'task3_capacity_extended';OUT.mkdir(exist_ok=True)
def compile_kernel():
 path=OUT/'joint.so'
 if not path.exists():subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task3_joint_kernel.cpp')),'-o',str(path)],check=True)
 return path

def make_data(stage):
 p,_,oldq,*_=dataset(DATA,False);u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);N=len(p);oldorder=np.random.RandomState(42).permutation(N);a=int(.9*N);pool=oldorder[:a]
 if stage=='fresh':
  shuffled=np.random.RandomState(91).permutation(pool);cut=int(.9*len(shuffled));rows=shuffled[:cut];qr=shuffled[cut:];query=p.iloc[qr].reset_index(drop=True);tune=len(query)//2
  assert not np.intersect1d(qr,oldorder[a:]).size and not np.intersect1d(rows,qr).size
 elif stage=='validation':rows=pool;query=p.iloc[oldorder[a:]].reset_index(drop=True);tune=0
 else:rows=np.arange(N);query=pd.read_csv(DATA/'task_3_test.csv');tune=0
 qu=pd.Index(users).get_indexer(query.user).astype('int32');qi=pd.Index(items).get_indexer(query.item).astype('int32');U,I=len(users),len(items)
 uu=u[rows].astype('int32');ii=i[rows].astype('int32');tt=p.timestamp.to_numpy(float)[rows];yy=p.rating.to_numpy(float)[rows];assert (yy>0).all();sort=np.argsort(uu,kind='stable');uu=uu[sort];ii=np.ascontiguousarray(ii[sort]);tt=tt[sort];yy=np.ascontiguousarray(yy[sort],dtype='f');count=np.bincount(uu,minlength=U);ic=np.ascontiguousarray(np.bincount(ii,minlength=I),dtype='int32');ptr=np.ascontiguousarray(np.r_[0,np.cumsum(count)],dtype='int32')
 fields,dev,qdev,center=time_arrays(uu,ii,tt,qu,qi,query.timestamp.to_numpy(float),U);codes=[c for c,_,_ in fields];counts=[np.ascontiguousarray(np.bincount(c),dtype='int32') for c in codes];H=coo_matrix((np.ones(len(uu),dtype='f'),(uu,ii)),shape=(U,I)).tocsr()
 if stage!='full':assert np.asarray(H[qu,qi]).max()==0,'Held-out positive found in training history'
 return dict(U=U,I=I,u=uu,i=ii,t=tt,y=yy,ptr=ptr,ic=ic,uc=count,fields=fields,codes=codes,counts=counts,dev=dev,center=center,H=H,qu=qu,qi=qi,qt=query.timestamp.to_numpy(float),target=query.rating.to_numpy(float) if stage!='full' else None,tune=tune,rows=len(rows),query=query)

def predict_state(s,u,i,t):
 u=np.asarray(u,dtype='int64');i=np.asarray(i,dtype='int64');good=(t>=0)&(t<=4102444800);day=np.where(good,np.floor(np.clip(t,0,4102444800)/86400),-1).astype('int64');d=(t-s['center'][np.maximum(u,0)])/(86400*365.25);dev=np.where(good&(u>=0),np.clip(np.sign(d)*abs(d)**.4,-5,5),0).astype('f')
 out=np.full(len(u),float(s['mu']));vu=(u>=0)&(s['uc'][np.maximum(u,0)]>0);vi=(i>=0)&(s['ic'][np.maximum(i,0)]>0);ok=vu&vi;out[vu]+=s['bu'][u[vu]]+s['alpha'][u[vu]]*dev[vu];out[vi]+=s['bi'][i[vi]];out[ok]+=np.einsum('ij,ij->i',s['effective'][u[ok]]+dev[ok,None]*s['A'][u[ok]],s['Q'][i[ok]])
 keys=[(u+1)*1000000000+day,(i+1)*1000000000+np.where(good,day//90,-1),(u+1)*1000000000+np.where(good,day//90,-1),np.where(good,day//30,-1)]
 for j,labels in enumerate(keys):
  ix=pd.Index(s['keys'+str(j)]).get_indexer(labels);valid=ix>=0;out[valid]+=s['term'+str(j)][ix[valid]]
 return clip(out)

def run_worker(k,stage,epochs=None):
 folder=OUT/str(k);folder.mkdir(exist_ok=True);cache=folder/(stage+'_scores.npz');report_path=folder/(stage+'_results.json')
 if cache.exists() and report_path.exists():return json.loads(report_path.read_text())
 d=make_data(stage);U,I=d['U'],d['I'];rng=np.random.RandomState(47);std=.1*(48/k)**.25;P=rng.normal(0,std,(U,k)).astype('f');Q=rng.normal(0,std,(I,k)).astype('f');Y=np.zeros((I,k),dtype='f');A=np.zeros_like(P);bu=np.zeros(U,dtype='f');bi=np.zeros(I,dtype='f');terms=[np.zeros(len(c),dtype='f') for c in d['counts']];alpha=np.zeros(U,dtype='f');mu=float(d['y'].mean())
 lib=ctypes.CDLL(str((OUT/'joint.so').resolve()));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');fn=lib.joint_train;fn.argtypes=[ctypes.c_int,ctypes.c_int]+[ip]*6+[fp]*2+[ip]*5+[ctypes.c_float]+[fp]*11+[ctypes.c_int]*3+[ctypes.c_float]*5;fn.restype=None
 checkpoints=[15,30,50,75,100,150,200] if stage=='fresh' else list(range(10,epochs,10))+[epochs];last=0;best=None;stale=0;history=[];started=time.time();state=None
 for stop in checkpoints:
  fn(U,k,d['ptr'],d['i'],*d['codes'],d['dev'],d['y'],d['ic'],*d['counts'],mu,P,Q,Y,A,bu,bi,*terms,alpha,last,stop-last,47,.005,.08,30.,40.,10.);last=stop
  assert all(np.isfinite(v).all() for v in [P,Q,Y,A,bu,bi,alpha,*terms])
  print('CAPACITY',k,stage,'epoch',stop,'elapsed',round(time.time()-started),flush=True)
  if stage=='fresh' or stop==epochs:
   effective=np.asarray(P+(d['H']@Y)/np.sqrt(np.maximum(d['uc'],1))[:,None],dtype='f');state=dict(effective=effective,Q=Q,A=A,bu=bu,bi=bi,alpha=alpha,mu=np.array(mu),center=d['center'],uc=d['uc'],ic=d['ic'],**{'keys'+str(j):f[2] for j,f in enumerate(d['fields'])},**{'term'+str(j):v for j,v in enumerate(terms)});pred=predict_state(state,d['qu'],d['qi'],d['qt'])
   if stage=='fresh':
    score=mse(d['target'][:d['tune']],pred[:d['tune']]);row=dict(epochs=stop,selection_MSE=score);history.append(row);print('FRESH_SELECTION',k,row,flush=True)
    if best is None or score<best[0]-1e-6:best=(score,stop,pred.copy());stale=0
    else:stale+=1
    if stale>=2:break
 if stage=='fresh':
  score,chosen,pred=best;r=dict(factors=k,stage=stage,selected_epochs=chosen,selection_MSE=score,evaluation_MSE=mse(d['target'][d['tune']:],pred[d['tune']:]),history=history,selection_rows=d['tune'],evaluation_rows=len(pred)-d['tune'],train_rows=d['rows'],original_meta_target_overlap=0);np.savez_compressed(cache,prediction=pred,y=d['target'],tune=d['tune'])
 else:
  # Round trip full parameter state through a compressed in-memory archive.
  # Persist predictions/config rather than two additional~100MB checkpoints.
  if stage=='full':
   ix=np.random.RandomState(11).choice(len(pred),min(256,len(pred)),replace=False);buf=io.BytesIO();np.savez_compressed(buf,**state);buf.seek(0);z=np.load(buf);restored={name:z[name] for name in z.files};rp=predict_state(restored,d['qu'][ix],d['qi'][ix],d['qt'][ix]);assert np.array_equal(rp,pred[ix]);replay=True
  else:replay=None
  np.savez_compressed(cache,prediction=pred);r=dict(factors=k,stage=stage,epochs=epochs,train_rows=d['rows'],scored_rows=len(pred),in_memory_checkpoint_replay=replay)
 r.update(parameters=dict(seed=47,learning_rate=.005,factor_regularization=.08,drift_factor_regularization=.15,day_penalty=30,item_period_penalty=40,user_period_penalty=10,initial_std=std),seconds=round(time.time()-started,2));report_path.write_text(json.dumps(r,indent=2));return r
