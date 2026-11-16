"""Train-only pair ranking: complementary neural/leaf-wise boosted ensembles.
Inner selection is split before pair mirroring; evaluation labels never tune weights.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import sys,json,pickle,argparse,gc,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
sys.path.insert(0,'/workspace/work/task3_lgb')
import lightgbm as lgb
W=Path('/workspace/work'); DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026'
def acc(p):return float(np.mean((p>0)+.5*(p==0)))
def pair(f):
 n=len(f)//2;return (f[:n]-f[n:]).astype('f'),(f[:n]+f[n:]).astype('f')
def load(task,full=False):
 stage='full' if full else 'validation'
 if task==2:
  z=np.load(W/f'task2_optimized/with_zero/{stage}/features.npz');d,s=pair(np.column_stack([z[k] for k in z.files]));n=93082
 elif task==4:
  base=np.load(W/'task4/results'/('test_pair_features.npz' if full else 'pair_features.npz'))['scores'];z=np.load(W/f'task4_optimized/{stage}/extended_features.npz');d,s=pair(np.c_[base,np.column_stack([z[k] for k in z.files])]);n=71462
 else:
  from task5_preference_round3 import load as loader
  a=SimpleNamespace(data_dir=DATA,baseline=W/'task5/results',previous=W/'task5_improved',reference=W/'task5_next',out=W/'task5_round3')
  d,s,_,_=loader(a,full);n=86786
 assert np.isfinite(d).all() and np.isfinite(s).all();return d,s,n

def catmargin(m,d,s):return m.predict_proba(np.c_[d,s],thread_count=2)[:,1]-m.predict_proba(np.c_[-d,s],thread_count=2)[:,1]
def reference(task,d,s,n,tr,va,full=False):
 if task in [2,4]:
  root=W/('task2_optimized/with_zero' if task==2 else 'task4_optimized');size=len(json.loads((root/'validation/results.json').read_text())['feature_names']);d,s=d[:,:size],s[:,:size]
 if task==5:
  _,oldscale=pickle.loads((W/'task5_round3/preference/small_evaluation.pkl').read_bytes());size=oldscale.n_features_in_//2;d,s=d[:,:size],s[:,:size]
  from task5_preference_round3 import current
  from task5_round3 import load as core,nn_margin
  a=SimpleNamespace(data_dir=DATA,baseline=W/'task5/results',previous=W/'task5_improved',reference=W/'task5_next',out=W/'task5_round3');parts=core(a,full)
  def one(stage,ix):
   m,scale=pickle.loads((a.out/f'preference/small_{stage}.pkl').read_bytes());return .5*nn_margin(m,scale,d[ix],s[ix])+.5*current(a,parts,stage,ix)
  if full:return one('evaluation',slice(None))
  return one('inner',va),one('evaluation',slice(n,None))
 root=W/('task2_optimized/with_zero' if task==2 else 'task4_optimized');r=json.loads((root/'validation/results.json').read_text())['selected']
 def fitted(path):
  m=CatBoostClassifier();m.load_model(str(path));return m
 if full:
  if task==2:return catmargin(fitted(root/'validation/stacker.cbm'),d,s)
  return .5*catmargin(fitted(root/'validation/extended_stacker.cbm'),d,s)+.5*catmargin(fitted(root/'validation/baseline_refit.cbm'),d[:,:23],s[:,:23])
 def fit(cols,depth,iters,lr):
  m=CatBoostClassifier(iterations=iters,depth=depth,learning_rate=lr,l2_leaf_reg=10,random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
  m.fit(np.r_[np.c_[d[tr,:cols],s[tr,:cols]],np.c_[-d[tr,:cols],s[tr,:cols]]],np.r_[np.ones(len(tr)),np.zeros(len(tr))]);return catmargin(m,d[va,:cols],s[va,:cols])
 print('REFERENCE_INNER_FIT',task,flush=True)
 ri=fit(d.shape[1],r['depth'],r['iterations'],.035)
 if task==2:re=2*np.load(root/'validation/validation_margins.npz')['margin'][n:]
 else:
  ri=.5*ri+.5*fit(23,5,698,.04);z=np.load(root/'validation/evaluation.npz');re=z['extended']+z['baseline']
 return ri,re

def transform(d,s,scale):
 # Signed log compression bounds extreme sparse counts without discarding signs.
 return np.c_[np.arcsinh(d/scale),np.arcsinh(s/scale)].astype('f')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);ap.add_argument('--readers',action='store_true');a=ap.parse_args();task=a.task;tag=f'task{task}_round4'+('_readers' if a.readers else '');out=W/tag;out.mkdir(exist_ok=True)
 d,s,n=load(task)
 if a.readers:
  z=np.load(W/f'task{task}_round4/readers_validation.npz');rd,rs=pair(np.column_stack([z[k] for k in z.files]));d,s=np.c_[d,rd],np.c_[s,rs]
 idx=np.random.RandomState(17).permutation(n);tr=idx[:int(.8*n)];va=idx[int(.8*n):]
 cache=out/'reference.npz'
 if a.readers and not cache.exists():
  import shutil
  shutil.copyfile(W/f'task{task}_round4/reference.npz',cache)
 if cache.exists():z=np.load(cache);ri,re=z['inner'],z['evaluation']
 else:ri,re=reference(task,d,s,n,tr,va);np.savez_compressed(cache,inner=ri,evaluation=re)
 print('REFERENCE',task,acc(ri),acc(re),d.shape,flush=True)
 inner={'reference':ri};configs={};sc=np.maximum(np.median(np.abs(np.r_[d[tr],s[tr]]),axis=0),1e-4).astype('f');X=transform(d,s,sc);Xm=transform(-d,s,sc);y=np.r_[np.ones(len(tr)),np.zeros(len(tr))]
 for leaves in ([] if (a.readers and task==5) else [31] if a.readers else [15,31,63]):
  key=f'lgb{leaves}';path=out/(key+'_inner.pkl')
  if path.exists():model=pickle.loads(path.read_bytes())
  else:
   model=lgb.LGBMClassifier(n_estimators=1000,num_leaves=leaves,learning_rate=.035,min_child_samples=150,reg_lambda=20,colsample_bytree=.9,n_jobs=2,verbosity=-1,random_state=211)
   model.fit(np.r_[X[tr],Xm[tr]],y,eval_set=[(np.r_[X[va],Xm[va]],np.r_[np.ones(len(va)),np.zeros(len(va))])],callbacks=[lgb.early_stopping(100,verbose=False)])
   path.write_bytes(pickle.dumps(model,protocol=5))
  inner[key]=model.predict_proba(X[va])[:,1]-model.predict_proba(Xm[va])[:,1];configs[key]={'leaves':leaves,'iterations':model.best_iteration_};print('INNER',task,key,configs[key],acc(inner[key]),flush=True)
 for seed,hidden,alpha in ([(131,(64,32),1.)] if a.readers else [(131,(64,32),1.),(227,(128,64),5.)]):
  key=f'nn{seed}';path=out/(key+'_inner.pkl');cfgpath=out/(key+'.json')
  if path.exists() and cfgpath.exists():model,scale=pickle.loads(path.read_bytes());cfg=json.loads(cfgpath.read_text())
  else:
   scale=StandardScaler().fit(np.r_[X[tr],Xm[tr]]);xx=scale.transform(np.r_[X[tr],Xm[tr]]);v=scale.transform(X[va]);vm=scale.transform(Xm[va]);model=MLPClassifier(hidden_layer_sizes=hidden,alpha=alpha,batch_size=512,learning_rate_init=.0007,random_state=seed);best=-1;epochbest=0
   for epoch in range(1,41):
    model.partial_fit(xx,y,classes=[0,1]);p=model.predict_proba(v)[:,1]-model.predict_proba(vm)[:,1];score=max(acc(w*p+(1-w)*ri) for w in [.25,.5,.75,1])
    if score>best:
     best=score;epochbest=epoch;path.write_bytes(pickle.dumps((model,scale),protocol=5));cfg={'seed':seed,'hidden':hidden,'alpha':alpha,'epochs':epoch};cfgpath.write_text(json.dumps(cfg))
    if epoch%5==0:print('NN',task,key,epoch,acc(p),best,flush=True)
    if epoch-epochbest>=8:break
   model,scale=pickle.loads(path.read_bytes())
  inner[key]=model.predict_proba(scale.transform(X[va]))[:,1]-model.predict_proba(scale.transform(Xm[va]))[:,1];configs[key]=cfg
 # Greedy convex blend selection exclusively on inner holdout, limited 2 additions.
 weights={'reference':1.};p=ri.copy()
 for _ in range(2):
  best=(acc(p),None,0)
  for k,v in inner.items():
   for w in [.25,.5,.75,1.]:
    score=acc((1-w)*p+w*v)
    if score>best[0]:best=(score,k,w)
  if best[1] is None:break
  _,k,w=best;weights={j:v*(1-w) for j,v in weights.items()};weights[k]=weights.get(k,0)+w;p=sum(weights[k]*inner[k] for k in weights)
 weights={k:v for k,v in weights.items() if v>0};print('CHOSEN',task,weights,acc(p),flush=True)
 # Fit only selection rows. Final event/evaluation rows remain untouched.
 evals={'reference':re};models={};scfull=np.maximum(np.median(np.abs(np.r_[d[:n],s[:n]]),axis=0),1e-4).astype('f');X=transform(d,s,scfull);Xm=transform(-d,s,scfull);yy=np.r_[np.ones(n),np.zeros(n)]
 for k,w in weights.items():
  if k=='reference':continue
  cfg=configs[k];path=out/(k+'_evaluation.pkl')
  if path.exists():obj=pickle.loads(path.read_bytes())
  elif k.startswith('lgb'):
   obj=lgb.LGBMClassifier(n_estimators=cfg['iterations'],num_leaves=cfg['leaves'],learning_rate=.035,min_child_samples=150,reg_lambda=20,colsample_bytree=.9,n_jobs=2,verbosity=-1,random_state=211);obj.fit(np.r_[X[:n],Xm[:n]],yy);path.write_bytes(pickle.dumps(obj,protocol=5))
  else:
   scale=StandardScaler().fit(np.r_[X[:n],Xm[:n]]);xx=scale.transform(np.r_[X[:n],Xm[:n]]);m=MLPClassifier(hidden_layer_sizes=tuple(cfg['hidden']),alpha=cfg['alpha'],batch_size=512,learning_rate_init=.0007,random_state=cfg['seed'])
   for epoch in range(cfg['epochs']):m.partial_fit(xx,yy,classes=[0,1])
   obj=(m,scale);path.write_bytes(pickle.dumps(obj,protocol=5))
  def predict(obj,x,xm):
   if isinstance(obj,tuple):m,scale=obj;return m.predict_proba(scale.transform(x))[:,1]-m.predict_proba(scale.transform(xm))[:,1]
   return obj.predict_proba(x)[:,1]-obj.predict_proba(xm)[:,1]
  models[k]=obj;evals[k]=predict(obj,X[n:],Xm[n:]);print('EVALUATION',task,k,acc(evals[k]),flush=True)
 primary=sum(weights[k]*evals[k] for k in weights);r={'task':task,'weights':weights,'configs':configs,'inner':{k:acc(v) for k,v in inner.items()},'selected_inner':acc(p),'reference_evaluation':acc(re),'primary_evaluation':acc(primary),'blend50_evaluation':acc(.5*primary+.5*re),'evaluation_components':{k:acc(v) for k,v in evals.items()},'protocol':'inner-only selection; fit selection rows; evaluation excluded; zeros retained'};print('RESULT',r,flush=True);(out/'results.json').write_text(json.dumps(r,indent=2));np.savez_compressed(out/'evaluation.npz',primary=primary,reference=re);np.save(out/'transform_scale.npy',scfull)
 del d,s,X,Xm;gc.collect();fd,fs,_=load(task,True)
 if a.readers:
  z=np.load(W/f'task{task}_round4/readers_full.npz');rd,rs=pair(np.column_stack([z[k] for k in z.files]));fd,fs=np.c_[fd,rd],np.c_[fs,rs]
 ref=reference(task,fd,fs,n,tr,va,True);xx=transform(fd,fs,scfull);xm=transform(-fd,fs,scfull);pp=weights.get('reference',0)*ref
 for k,obj in models.items():pp+=weights[k]*predict(obj,xx,xm)
 assert np.isfinite(pp).all();test=pd.read_csv(DATA/f'task_{task}_test.csv');assert len(test)==len(pp)
 for label,values in [('primary',pp),('conservative',.5*pp+.5*ref)]:
  dest=Path('/workspace/output')/f'{tag}_{label}';dest.mkdir(exist_ok=True,parents=True);f=test.copy();f['which']=np.where(values>=0,f.item1,f.item2);f.to_csv(dest/f'task_{task}_predictions.csv',index=False)
 np.savez_compressed(out/'test_margins.npz',primary=pp,reference=ref)
if __name__=='__main__':main()
