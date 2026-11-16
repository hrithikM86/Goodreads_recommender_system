"""Temporal FM addition and fresh retraining of every accepted Task4 component.
All selection stays inside selection split, evaluation labels are not used to tune.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,subprocess,json,gc,pickle,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from tasks245_round4 import load,W,DATA,acc
from tasks24_reader_neighbors import inputs
from task4_drift import ref
OUT=W/'task4_two_approaches';OUT.mkdir(exist_ok=True)
NSEL=71462

def split():
 df=pd.read_csv(DATA/'tasks_3_4_train.csv');u,users=pd.factorize(df.user);i,items=pd.factorize(df.item);t=df.timestamp.to_numpy(float);order=np.random.RandomState(42).permutation(len(df));a=int(.9*len(df));idx=order[:a];hold=order[a:];rng=np.random.RandomState(43);neg=[]
 for pos in i[hold]:
  j=i[idx[rng.randint(len(idx))]]
  while j==pos:j=i[idx[rng.randint(len(idx))]]
  neg.append(j)
 return df,u,i,t,users,items,idx,hold,np.array(neg)
def queries(full):
 df,u,i,t,users,items,idx,hold,neg=split()
 if not full:return u[idx],i[idx],t[idx],np.tile(u[hold],2),np.r_[i[hold],neg],np.tile(t[hold],2),len(users),len(items)
 q=pd.read_csv(DATA/'task_4_test.csv');qu=pd.Index(users).get_indexer(q.user);qi=np.r_[pd.Index(items).get_indexer(q.item1),pd.Index(items).get_indexer(q.item2)];return u,i,t,np.tile(qu,2),qi,np.tile(q.timestamp.to_numpy(float),2),len(users),len(items)
def parts():
 idx=np.random.RandomState(17).permutation(NSEL);c=int(.8*NSEL);return idx[:c],idx[c:]

def fm(full=False):
 path=OUT/('fm_full.npz' if full else 'fm_validation.npz')
 if path.exists():return dict(np.load(path))
 u,i,t,qu,qi,qt,U,I=queries(full);days=t/86400;good=(days>=0)&(days<4102444800/86400);lo,hi=np.quantile(days[good],[.01,.99]);width=max((hi-lo)/24,1);b=np.clip((days-lo)/width,0,23).astype('i');qb=np.clip((qt/86400-lo)/width,0,23).astype('i');b[~good]=24;qb[(qt<0)|(qt>=4102444800)]=24
 # user-period effect uses coarse, fitted calendar slices; unknown slices have no ID.
 period=b//4;keys=u.astype('int64')*10+period;code,index=pd.factorize(keys);qcode=pd.Index(index).get_indexer(qu.astype('int64')*10+qb//4);off=U+I+25;ids=np.ascontiguousarray(np.c_[u,U+i,U+I+b,off+code],dtype='i');qids=np.ascontiguousarray(np.c_[qu,U+qi,U+I+qb,np.where(qcode>=0,off+qcode,-1)],dtype='i');nfeatures=off+len(index);k=16;rng=np.random.RandomState(419);V=rng.normal(0,.03,(nfeatures,k)).astype('f');w=np.zeros(nfeatures,'f');pool=i.astype('i');lib=ctypes.CDLL(str(OUT/'pair_fm.so'));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.train.argtypes=[ctypes.c_int]*5+[ip]*2+[fp]*2+[ctypes.c_int]*3;lib.predict.argtypes=[ctypes.c_int]*3+[ip]+[fp]*3
 result={};old=0
 for epoch in [5,15,30]:
  lib.train(len(ids),4,k,I,U,ids,pool,w,V,old,epoch-old,419);old=epoch;value=np.empty(len(qids),'f');lib.predict(len(qids),4,k,qids,w,V,value);assert np.isfinite(value).all();result['fm_e'+str(epoch)]=value
  print('FM',full,epoch,flush=True)
 np.savez_compressed(path,**result);return result

def drift_design(f):return np.c_[f[:,:10],f[:,1:10]*f[:,10:]]
def accepted():
 path=OUT/'accepted.npz'
 if path.exists():return dict(np.load(path))
 d,s,n=load(4);z=dict(np.load(W/'task4_drift/validation.npz'));F=np.column_stack(list(z.values()));N=len(d);reference=ref(d,s);f=np.c_[reference,F[:N]-F[N:],F[:N]+F[N:]];tr,va=parts();f[va,0]=np.load(W/'task4_round4/reference.npz')['inner'];x=drift_design(f)
 def fitted(rows):
  scale=StandardScaler(with_mean=False).fit(x[rows]);xx=scale.transform(x[rows]);m=LogisticRegression(C=.001,fit_intercept=False,max_iter=500);m.fit(np.r_[xx,-xx],np.r_[np.ones(len(xx)),np.zeros(len(xx))]);return m,scale
 m0,sc0=fitted(tr);inner=expit(m0.decision_function(sc0.transform(x[va])))-.5;m,sc=fitted(slice(n));ev=expit(m.decision_function(sc.transform(x[n:])))-.5;sel=expit(m.decision_function(sc.transform(x[:n])))-.5
 fd,fs,_=load(4,True);r=ref(fd,fs);F=np.column_stack(list(dict(np.load(W/'task4_drift/full.npz')).values()));N=len(fd);test=expit(m.decision_function(sc.transform(drift_design(np.c_[r,F[:N]-F[N:],F[:N]+F[N:]]))))-.5
 q=pd.read_csv(DATA/'task_4_test.csv');csv=pd.read_csv('/workspace/output/task4_time_bpr/task_4_predictions.csv');assert np.array_equal(csv.which,np.where(test>=0,q.item1,q.item2));out={'inner':inner,'selection':sel,'evaluation':ev,'test':test};np.savez_compressed(path,**out);print('ACCEPTED_REPLAY',acc(ev),flush=True);return out

def pair(f):n=len(f)//2;return f[:n]-f[n:],f[:n]+f[n:]
def calibrate(d,tr,va,fd,epochs):
 # Positive candidate first on historical pairs; mirrored differences calibrate probability.
 best=(-1,None,None)
 for epoch in epochs:
  xx=d[:,epoch:epoch+1];m=LogisticRegression(C=1,fit_intercept=False,max_iter=300);m.fit(np.r_[xx[tr],-xx[tr]],np.r_[np.ones(len(tr)),np.zeros(len(tr))]);p=m.predict_proba(xx[va])[:,1]-.5
  if acc(p)>best[0]:best=(acc(p),epoch,p)
 return best

def blend():
 accs=accepted();z=fm();fz=fm(True);d,_=pair(np.column_stack(list(z.values())));fd,_=pair(np.column_stack(list(fz.values())));tr,va=parts();n=NSEL;options=[]
 for c in range(d.shape[1]):
  m=LogisticRegression(C=1,fit_intercept=False,max_iter=300);xx=d[:,c:c+1];m.fit(np.r_[xx[tr],-xx[tr]],np.r_[np.ones(len(tr)),np.zeros(len(tr))]);p=m.predict_proba(xx[va])[:,1]-.5
  for w in [.1,.25,.5,.75,1]:options.append((acc((1-w)*accs['inner']+w*p),c,w))
 score,c,w=max(options,key=lambda r:r[0]);m=LogisticRegression(C=1,fit_intercept=False,max_iter=300);xx=d[:n,c:c+1];m.fit(np.r_[xx,-xx],np.r_[np.ones(n),np.zeros(n)]);ev=(1-w)*accs['evaluation']+w*(m.predict_proba(d[n:,c:c+1])[:,1]-.5);test=(1-w)*accs['test']+w*(m.predict_proba(fd[:,c:c+1])[:,1]-.5);report={'approach':'accepted plus FM','fm_checkpoint':list(z)[c],'FM_weight':w,'inner_accuracy':score,'accepted_inner_accuracy':acc(accs['inner']),'accepted_evaluation_accuracy':acc(accs['evaluation']),'evaluation_accuracy':acc(ev)};print('BLEND',report,flush=True);(OUT/'blend_results.json').write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/'blend_margins.npz',evaluation=ev,test=test);export('task4_accepted_plus_fm',test)

def export(tag,value):
 test=pd.read_csv(DATA/'task_4_test.csv');assert len(value)==len(test) and np.isfinite(value).all();test['which']=np.where(value>=0,test.item1,test.item2);dest=Path('/workspace/output')/tag;dest.mkdir(exist_ok=True);test.to_csv(dest/'task_4_predictions.csv',index=False)

def fresh_stage(full):
 """Every learned component is refitted in a new directory, without loading old factors."""
 dest=OUT/('fresh_full' if full else 'fresh_validation');dest.mkdir(exist_ok=True);final=dest/'scores.npz'
 if final.exists():return np.load(final)['scores'],json.loads((dest/'names.json').read_text())
 import task4_ensemble as base,task4_optimize as ext,task4_drift as drift
 from scipy.sparse import coo_matrix
 u,i,t,qu,qi,qt,U,I=queries(full);X=ext.binary(u,i,(U,I));Times=coo_matrix((t,(u,i)),shape=X.shape).tocsr()
 # Persist only query-score artifacts; fitted factors are consumed in memory.
 oldsavez=np.savez_compressed;oldsave=np.save
 def save_npz(path,*args,**kw):
  name=Path(path).name
  if ('user_factors' in kw or 'user' in kw) and name!='bpr128.npz':return
  return oldsavez(path,*args,**kw)
 def save_npy(path,*args,**kw):
  if Path(path).name.startswith('ease') and not 'vocabulary' in Path(path).name:return
  return oldsave(path,*args,**kw)
 np.savez_compressed=save_npz;np.save=save_npy
 try:
  models=base.train_models(X,dest/'factors');tm=base.train_time_models(u,i,t,X.shape,dest/'factors');bn,bs=base.features(X,models,qu,qi,Times,qt,tm);del models,tm;gc.collect();en,es=ext.extended_features(X,u,i,t,qu,qi,qt,dest)
 finally:np.savez_compressed=oldsavez;np.save=oldsave
 # Refit drift on freshly fitted BPR, using the same source algorithm in an isolated namespace.
 src=Path(drift.__file__).read_text();src=src[src.index('def component('):src.index('\ndef ref(')];src=src.replace("W/'task4_optimized'/('full' if full else 'validation')",'dest');ns=dict(drift.__dict__);ns.update(OUT=OUT,dest=dest);exec(src,ns);dz=ns['component'](full);ds=np.column_stack(list(dz.values()));scores=np.c_[bs,es,ds].astype('f');names=bn+en+['drift_'+n for n in dz];assert len(names)==scores.shape[1];
 for scratch in ['extended_features.npz','linear_scores.npz']:
  p=dest/scratch
  if p.exists():p.unlink()
 np.savez_compressed(final,scores=scores);(dest/'names.json').write_text(json.dumps(names));
 # Full factors are deterministic, reproducible and deliberately not archived on this disk.
 for p in dest.rglob('bpr128.npz'):p.unlink()
 print('ALL_COMPONENTS_REFIT',full,len(names),flush=True);return scores,names

def retrain():
 z,names=fresh_stage(False);F=fm();z=np.c_[z,np.column_stack(list(F.values()))];names+=list(F);d,s=pair(z);tr,va=parts();n=NSEL
 # One fresh combiner over every refitted component, no inherited nested blend.
 best=(-1,None,None);reports={}
 for depth in [4,6]:
  m=CatBoostClassifier(iterations=1000,depth=depth,learning_rate=.035,l2_leaf_reg=20,random_seed=529,thread_count=2,verbose=False,allow_writing_files=False)
  m.fit(np.r_[np.c_[d[tr],s[tr]],np.c_[-d[tr],s[tr]]],np.r_[np.ones(len(tr)),np.zeros(len(tr))],eval_set=(np.r_[np.c_[d[va],s[va]],np.c_[-d[va],s[va]]],np.r_[np.ones(len(va)),np.zeros(len(va))]),early_stopping_rounds=100)
  def margin(m,d,s):return m.predict_proba(np.c_[d,s])[:,1]-m.predict_proba(np.c_[-d,s])[:,1]
  choices=[(acc(margin_prefix(m,d[va],s[va],count)),count) for count in sorted(set([m.tree_count_]+list(range(50,m.tree_count_+1,50))))];score,count=max(choices,key=lambda x:(x[0],-x[1]));reports[str(depth)]={'inner_accuracy':score,'iterations':count};print('FRESH_INNER',depth,reports[str(depth)],flush=True)
  if score>best[0]:best=(score,depth,count)
 score,depth,count=best;m=CatBoostClassifier(iterations=count,depth=depth,learning_rate=.035,l2_leaf_reg=20,random_seed=529,thread_count=2,verbose=False,allow_writing_files=False);m.fit(np.r_[np.c_[d[:n],s[:n]],np.c_[-d[:n],s[:n]]],np.r_[np.ones(n),np.zeros(n)]);ev=margin(m,d[n:],s[n:]);m.save_model(str(OUT/'fresh_combiner.cbm'));r={'approach':'all components individually retrained plus FM; one fresh CatBoost combiner','inner_accuracy':score,'evaluation_accuracy':acc(ev),'runs':reports,'features':names};(OUT/'fresh_results.json').write_text(json.dumps(r,indent=2));print('FRESH_RESULT',r['evaluation_accuracy'],flush=True);del z,d,s;gc.collect();f,fn=fresh_stage(True);assert fn==names[:-len(F)];f=np.c_[f,np.column_stack(list(fm(True).values()))];fd,fs=pair(f);test=margin(m,fd,fs);np.savez_compressed(OUT/'fresh_margins.npz',evaluation=ev,test=test);export('task4_all_models_retrained_fm',test)

def margin_prefix(m,d,s,count):return m.predict_proba(np.c_[d,s],ntree_end=count)[:,1]-m.predict_proba(np.c_[-d,s],ntree_end=count)[:,1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['blend','retrain'],required=True);a=ap.parse_args();subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task4_pair_fm.cpp')),'-o',str(OUT/'pair_fm.so')],check=True);subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task4_drift.cpp')),'-o',str(OUT/'drift.so')],check=True);(blend if a.mode=='blend' else retrain)()
if __name__=='__main__':main()
