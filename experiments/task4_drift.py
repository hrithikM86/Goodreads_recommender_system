"""BPR residual model: continuous reader drift and calendar-specific book bias.
Frozen base BPR factors, train-only dates; selection-only meta-model fitting.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import ctypes,subprocess,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tasks24_reader_neighbors import inputs
from tasks245_round4 import W,DATA,load,acc
from catboost import CatBoostClassifier
OUT=W/'task4_drift';OUT.mkdir(exist_ok=True)
def component(full=False):
 X,T,qu,qi,qt=inputs(4,full);df=pd.read_csv(DATA/'tasks_3_4_train.csv');u,_=pd.factorize(df.user);i,_=pd.factorize(df.item);days=df.timestamp.to_numpy(float)/86400;order=np.random.RandomState(42).permutation(len(u));idx=np.arange(len(u)) if full else order[:int(.9*len(u))];root=W/'task4_optimized'/('full' if full else 'validation');z=np.load(root/'bpr128.npz');P=np.ascontiguousarray(z['user'],dtype='f');Q=np.ascontiguousarray(z['item'],dtype='f');N,I=X.shape;K=P.shape[1];B=16
 valid=(days[idx]>=0)&(days[idx]<4102444800/86400);ix=idx[valid];lo,hi=np.quantile(days[ix],[.01,.99]);width=max((hi-lo)/B,1);bins=np.clip((days[ix]-lo)/width,0,B-1).astype('i');qbins=np.clip((qt-lo)/width,0,B-1).astype('i');counts=np.bincount(u[ix],minlength=N);means=np.bincount(u[ix],weights=days[ix],minlength=N)/np.maximum(counts,1);means[counts==0]=np.median(days[ix]);t=np.clip((days[ix]-means[u[ix]])/365,-3,3).astype('f');tq=np.clip((qt-means[qu])/365,-3,3).astype('f');uu=u[ix].astype('i');ii=i[ix].astype('i');D=np.zeros((N,16),'f');C=np.zeros((I,B),'f');lib=ctypes.CDLL(str(OUT/'drift.so'));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.train.argtypes=[ctypes.c_int]*5+[ip]*3+[fp]*5+[ctypes.c_int];result={};old=0
 base=np.einsum('ij,ij->i',P[qu],Q[qi]);qvalid=(qt>=0)&(qt<4102444800/86400)
 for ep in [3,10,25]:
  lib.train(len(ix),I,K,B,ep-old,uu,ii,bins,t,P,Q,D,C,47+old);old=ep;drift=tq*np.einsum('ij,ij->i',D[qu],Q[qi,:16]);bias=C[qi,qbins];drift[~qvalid]=0;bias[~qvalid]=0
  for name,v in [('drift',drift),('calendar',bias),('joint',base+drift+bias)]:result[f'{name}{ep}']=v.astype('f')
  print('FIT',full,ep,flush=True)
 assert all(np.isfinite(v).all() for v in result.values());np.savez_compressed(OUT/('full.npz' if full else 'validation.npz'),**result);return result

def ref(d,s):
 root=W/'task4_optimized/validation';v=0
 for name,cols in [('extended_stacker',d.shape[1]),('baseline_refit',23)]:
  m=CatBoostClassifier();m.load_model(str(root/(name+'.cbm')));v+=.5*(m.predict_proba(np.c_[d[:,:cols],s[:,:cols]])[:,1]-m.predict_proba(np.c_[-d[:,:cols],s[:,:cols]])[:,1])
 return v

def main():
 subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_suffix('.cpp')),'-o',str(OUT/'drift.so')],check=True)
 z=component();d,s,n=load(4);N=len(d);f=np.column_stack(list(z.values()));m=N;dd=f[:m]-f[m:];ss=f[:m]+f[m:];reference=ref(d,s);features=np.c_[reference,dd,ss];tr=np.random.RandomState(17).permutation(n);cut=int(.8*n);fit,va=tr[:cut],tr[cut:];best=(-1,None,None)
 # Differences are odd, sums even. Include products to gate drift by baseline confidence.
 def design(f):return np.c_[f[:,:10],f[:,1:10]*f[:,10:]]
 features[va,0]=np.load(W/'task4_round4/reference.npz')['inner'];xx=design(features);models=[]
 for c in [.001,.01,.1,1]:
  scale=StandardScaler(with_mean=False).fit(xx[fit]);a=scale.transform(xx[fit]);model=LogisticRegression(C=c,fit_intercept=False,max_iter=500);model.fit(np.r_[a,-a],np.r_[np.ones(len(a)),np.zeros(len(a))]);pred=model.decision_function(scale.transform(xx[va]));score=acc(pred);print('INNER',c,score,flush=True)
  if score>best[0]:best=(score,c,None)
 scale=StandardScaler(with_mean=False).fit(xx[:n]);a=scale.transform(xx[:n]);model=LogisticRegression(C=best[1],fit_intercept=False,max_iter=500);model.fit(np.r_[a,-a],np.r_[np.ones(n),np.zeros(n)]);pred=model.decision_function(scale.transform(xx[n:]));r={'inner_accuracy':best[0],'C':best[1],'reference_evaluation':acc(reference[n:]),'new_evaluation':acc(pred),'note':'Frozen BPR with learned continuous reader drift and calendar book biases; no evaluation labels used for tuning.'};print('RESULT',r,flush=True);(OUT/'results.json').write_text(json.dumps(r,indent=2))
 fz=component(True);fd,fs,_=load(4,True);rtest=ref(fd,fs);f=np.column_stack(list(fz.values()));m=len(fd);features=np.c_[rtest,f[:m]-f[m:],f[:m]+f[m:]];value=model.decision_function(scale.transform(design(features)));test=pd.read_csv(DATA/'task_4_test.csv');dest=Path('/workspace/output/task4_time_bpr');dest.mkdir(exist_ok=True);test['which']=np.where(value>=0,test.item1,test.item2);assert len(test)==20000 and np.isfinite(value).all();test.to_csv(dest/'task_4_predictions.csv',index=False)
if __name__=='__main__':main()
