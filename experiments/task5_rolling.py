"""Refresh every accepted Task5 component on H[:-1], then refit fixed neural stack.
Only supplied Task5 user/item IDs and CSV order are used. No test labels.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,subprocess,sys,json,pickle,hashlib,gc,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
W=Path('/workspace/work');ROOT=Path(__file__).parent;OUT=W/'task5_rolling';OUT.mkdir(exist_ok=True)
DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026'
SEEDS=[43,441,997]

def cohort(*unused,**kw):
 if hasattr(cohort,'cache'):return cohort.cache
 p=pd.read_csv(DATA/'task_5_train.csv',usecols=['user','item']);u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);U,I=len(users),len(items);ptr=np.searchsorted(u,np.arange(U+1));H=[i[ptr[v]:ptr[v+1]].astype('i') for v in range(U)];h=[x[:-1] for x in H];pos=np.array([x[-1] for x in H],'i');pool=np.concatenate(h);neg=[]
 for seed in SEEDS:
  rng=np.random.RandomState(seed);n=np.empty(U,'i')
  for z,target in enumerate(pos):
   j=pool[rng.randint(len(pool))]
   while j==target:j=pool[rng.randint(len(pool))]
   n[z]=j
  neg.append(n)
 contexts=h*len(SEEDS);N=len(contexts);qu=np.tile(np.arange(U,dtype='i'),2*len(SEEDS));qi=np.r_[np.tile(pos,len(SEEDS)),np.concatenate(neg)].astype('i')
 assert all(target not in hh for target,hh in zip(pos,h));pop=np.bincount(pool,minlength=I);assert np.all(pop[[hh[-1] for hh in h]]>0)
 cohort.cache=(h,contexts,qu,qi,U,I,None);return cohort.cache

def _worker(kind):
 import task5_improve as imp
 imp.data=cohort
 # Parameter caches are dispensable except two warm starts used below.
 original=np.savez_compressed
 def compact(path,*args,**kw):
  name=Path(path).name
  if name.endswith('_model.npz') and name not in ['fpmc32_uniform_model.npz','fpmc64_useruniform_model.npz']:return
  original(path,*args,**{k:v.astype('f') if hasattr(v,'dtype') and v.dtype.kind=='f' else v for k,v in kw.items()})
 np.savez_compressed=compact
 a=SimpleNamespace(data_dir=DATA,baseline=W/'task5/results',out=OUT,previous=OUT,full=False)
 if kind=='base':
  import task5_ensemble as m
  h,contexts,qu,qi,U,I,_=cohort();X,T,models=m.prepare(h,I,OUT/'factors');models['tail_popularity']=m.tail_pop(h,I);names,scores=m.features(X,T,models,contexts,qu,qi);original(OUT/'validation/base.npz',scores=scores.astype('f'));(OUT/'base_names.json').write_text(json.dumps(names))
  for f in (OUT/'factors').glob('*.npz'):f.unlink()
 elif kind in ['fpmc','uniform','neighbors']:
  imp.fpmc(a,kind=='uniform') if kind!='neighbors' else imp.neighbors(a)
 else:
  module={'metrics':'task5_next','ease':'task5_ease_next','fossil':'task5_fossil_next','order':'task5_order','era':'task5_era','attention':'task5_attention','bpr':'task5_bpr_round3','fold':'task5_fold'}[kind]
  m=__import__(module);m.data=cohort;m.run(a)
 if kind=='attention':(OUT/'validation/fpmc32_uniform_model.npz').unlink(missing_ok=True)
 if kind=='fold':(OUT/'validation/fpmc64_useruniform_model.npz').unlink(missing_ok=True)
 print('WORKER_DONE',kind,flush=True)

def worker(kind):
 import fcntl
 with open(OUT/(kind+'.lock'),'w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (OUT/(kind+'.done')).exists():return
  _worker(kind)
  (OUT/(kind+'.done')).write_text('ok')

def args():return SimpleNamespace(data_dir=DATA,baseline=W/'task5/results',previous=W/'task5_improved',reference=W/'task5_next',out=W/'task5_round3')
def refreshed(width=260):
 from task5_preference_round3 import load
 # Discover the exact accepted input order; no refit or selection here.
 d,s,parts,extra=load(args());names=(json.loads((W/'task5/results/results.json').read_text())['feature_names']+parts[-1]+extra)[:width];del d,s,parts;gc.collect()
 bank={};base=np.load(OUT/'validation/base.npz')['scores'];base_names=json.loads((OUT/'base_names.json').read_text());bank.update(zip(base_names,base.T))
 for f in ['fpmc_features','fpmc_uniform_features','neighbor_features','metrics','ease','fossil','order','era','attention','bpr','fold']:
  if width==224 and f in ['bpr','fold']:continue
  with np.load(OUT/'validation'/(f+'.npz')) as z:
   for key in z.files:
    assert key not in bank,key
    bank[key]=z[key]
 assert len(names)==width and all(n in bank for n in names)
 score=np.column_stack([bank[n] for n in names]).astype('f');del bank;N=len(score)//2;return score[:N]-score[N:],score[:N]+score[N:],names

def combine():
 from task5_preference_round3 import load
 from task5_round3 import nn_margin,nn_fit,accuracy
 ed,es,names=refreshed();U=len(ed)//len(SEEDS);assert U==86786
 # Frozen accepted selection-only networks: clean rolling evaluation, no fitting on final targets.
 configs={'sequence':(224,71,16,W/'task5_round3/nn_evaluation.pkl'),'preference':(260,73,8,W/'task5_round3/preference/small_evaluation.pkl')};ev=0
 for key,(width,seed,epochs,path) in configs.items():
  model,scale=pickle.loads(path.read_bytes());ev+=.5*nn_margin(model,scale,ed[:,:width],es[:,:width])
 scores={str(seed):accuracy(ev[j*U:(j+1)*U]) for j,seed in enumerate(SEEDS)};print('REFRESHED_EVALUATION',scores,flush=True);np.savez_compressed(OUT/'evaluation.npz',margin=ev)
 # Each user's known final training target now joins the production meta-training set.
 # Three negative draws have total weight one by using one seed per reader, not duplicating rows.
 d,s,parts,extra=load(args());sd,ss=d[:U].copy(),s[:U].copy();del d,s,parts;gc.collect();chosen=np.random.RandomState(418).randint(len(SEEDS),size=U)*U+np.arange(U);td=np.r_[sd,ed[chosen]];ts=np.r_[ss,es[chosen]];del ed,es,sd,ss;gc.collect();models={}
 for key,(width,seed,epochs,path) in configs.items():
  dest=OUT/(key+'_production.pkl')
  if dest.exists():model=pickle.loads(dest.read_bytes())
  else:model,_=nn_fit(td[:,:width],ts[:,:width],epochs,seed=seed,hidden=(64,32),alpha=.1);dest.write_bytes(pickle.dumps(model,protocol=5))
  models[key]=model;print('PRODUCTION_FIT',key,flush=True)
 del td,ts;gc.collect();fd,fs,parts,extra=load(args(),True);pred=0
 for key,(width,seed,epochs,path) in configs.items():pred+=.5*nn_margin(*models[key],fd[:,:width],fs[:,:width])
 q=pd.read_csv(DATA/'task_5_test.csv');f=q.copy();f['which']=np.where(pred>=0,f.item1,f.item2);dest=Path('/workspace/output/task5_rolling_neural');dest.mkdir(exist_ok=True);path=dest/'task_5_predictions.csv';f.to_csv(path,index=False);read=pd.read_csv(path);assert len(read)==U and read.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert ((read.which==read.item1)|(read.which==read.item2)).all();old=pd.read_csv('/workspace/output/task5_round3_primary/task_5_predictions.csv');np.savez_compressed(OUT/'test.npz',margin=pred)
 r={'rolling_evaluation_accuracy_by_negative_seed':scores,'rolling_evaluation_mean':float(np.mean(list(scores.values()))),'old_stale_evaluation_accuracy':.9235245316064803,'comparison_limit':'Different fitted graph and freshly sampled negatives, not an apples-to-apples model gain. Production refit uses final training labels; no independent accuracy claimed for production weights.','evaluation_component_graph':'H[:-1], target H[-1] excluded','selection_component_graph':'H[:-2], target H[-2] excluded','production_component_graph':'H, existing fully fitted accepted component banks reused','combiner':'Accepted 224/260-score MLPs; 64/32 layers, alpha .1, seeds71/73,16/8epochs, fixed equal blend','production_training_pairs':2*U,'negative_sampling':'Event popularity, reject target only, three evaluation seeds; one randomly assigned seed per reader for production final-event pairs','output':str(path),'rows':U,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'changed_vs_accepted':int((read.which!=old.which).sum()),'test_labels_used':False,'rating_values_used':False,'numeric_timestamps_used':False,'input_hashes':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['task_5_train.csv','task_5_test.csv']}}
 (OUT/'results.json').write_text(json.dumps(r,indent=2));print('RESULT',json.dumps(r),flush=True)

def run():
 (OUT/'validation').mkdir(exist_ok=True)
 for kind in ['base','fpmc','attention','uniform','fold','neighbors','metrics','ease','fossil','order','era','bpr']:
  done=OUT/(kind+'.done')
  if done.exists():continue
  print('START',kind,time.ctime(),flush=True)
  with open(OUT/(kind+'.log'),'a') as log:subprocess.run([sys.executable,__file__,'--worker',kind],stdout=log,stderr=subprocess.STDOUT,check=True)
  done.write_text('ok');print('DONE',kind,time.ctime(),flush=True)
 combine()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--worker');a=ap.parse_args();worker(a.worker) if a.worker else run()
