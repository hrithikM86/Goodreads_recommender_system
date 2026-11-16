"""Add one SGNS Item2vec model to the accepted two-network Task5 architecture."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,json,pickle,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from task5_boosting import W,DATA,args
from task5_improve import data
from task5_preference_round3 import load
from task5_round3 import nn_fit,nn_margin,accuracy
OUT=W/'task5_item2vec';OUT.mkdir(exist_ok=True)
NAMES=['cos_last','cos_recent3','cos_recent10','cos_decay20','cos_max20','sgns_last','sgns_recent3','sgns_decay20']
def libraries():
 ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');train=ctypes.CDLL(str(OUT/'item2vec.so'));train.train.argtypes=[ctypes.c_int]*4+[ip]*3+[ctypes.c_int]+[fp]*2+[ctypes.c_int]*3;pred=ctypes.CDLL(str(OUT/'score.so'));pred.score.argtypes=[ctypes.c_int]*2+[ip]*3+[fp]*4;return train,pred

def features(full=False):
 path=OUT/('full_features.npz' if full else 'validation_features.npz')
 if path.exists():return np.load(path)['scores']
 train,pred=libraries();hist,contexts,qu,qi,U,I,q=data(DATA,W/'task5/results',full);lens=np.array(list(map(len,hist)));u=np.repeat(np.arange(U),lens).astype('i');items=np.concatenate(hist).astype('i');pop=np.bincount(items,minlength=I);active=np.flatnonzero(pop>=2);assert len(active)>1;keep=pop[items]>=2;su=u[keep];si=items[keep];length=np.bincount(su,minlength=U);source=np.bincount(si[length[su]>1],minlength=I);active=np.flatnonzero(source>0);p=source[active]**.75;p/=p.sum();rng=np.random.RandomState(1777);pool=rng.choice(active,size=524288,p=p).astype('i');k=32;E=rng.uniform(-.5/k,.5/k,(I,k)).astype('f');O=np.zeros_like(E);train.train(len(si),k,5,3,su,si,pool,len(pool),E,O,0,12,1777);assert np.isfinite(E).all() and np.isfinite(O).all();E[source==0]=0;O[source==0]=0;V=E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-8);qlens=np.tile(np.array(list(map(len,contexts))),2);ptr=np.r_[0,np.cumsum(qlens)].astype('i');qh=np.tile(np.concatenate(contexts).astype('i'),2);scores=np.zeros((len(qi),8),'f');pred.score(len(qi),k,ptr,qh,qi,E,O,V,scores);assert np.isfinite(scores).all();np.savez_compressed(path,scores=scores)
 # Retain small embedding/context samples for replay; full tables regenerate deterministically.
 rows=np.random.RandomState(419).choice(np.flatnonzero(qi>=0),128,replace=False);tokens=[];offset=[0]
 for r in rows:
  tokens.extend(qh[ptr[r]:ptr[r+1]][-20:].tolist());offset.append(len(tokens))
 tokens=np.array(tokens,'i');np.savez_compressed(OUT/('full_replay.npz' if full else 'validation_replay.npz'),rows=rows,target_ids=qi[rows],offset=np.array(offset),history_ids=tokens,history_E=E[tokens],history_V=V[tokens],target_O=O[qi[rows]],target_V=V[qi[rows]])
 r={'full':full,'factors':32,'epochs':12,'window':5,'negative_samples':3,'negative_frequency_power':.75,'min_count':2,'seed':1777,'training_events':len(si),'active_books':len(active),'features':NAMES,'ratings_used':False,'numeric_timestamps_used':False,'full_factor_tables_retained':False,'note':'Full factor tables regenerate deterministically; cached scores and sampled factors retained due disk capacity. Only fitting prefixes train embeddings. Query histories contain no own hidden targets.'};(OUT/('full_report.json' if full else 'validation_report.json')).write_text(json.dumps(r,indent=2));print('ITEM2VEC_READY',full,r,flush=True);return scores

def matrices(full=False):
 d,s,parts,_=load(args(),full);seqd,seqs=parts[:2];f=features(full);n=len(f)//2;assert n==len(d);dd=f[:n]-f[n:];ss=f[:n]+f[n:];return {'sequence':(np.c_[seqd,dd].astype('f'),np.c_[seqs,ss].astype('f')),'preference':(np.c_[d,dd].astype('f'),np.c_[s,ss].astype('f'))}
def combine():
 z=matrices();U=len(z['preference'][0])//2;models={};ev=0;cfgs={'sequence':{'seed':71,'epochs':16},'preference':{'seed':73,'epochs':8}}
 for kind,c in cfgs.items():
  d,s=z[kind];path=OUT/(kind+'_evaluation.pkl')
  if path.exists():m,scale=pickle.loads(path.read_bytes())
  else:(m,scale),_=nn_fit(d[:U],s[:U],c['epochs'],seed=c['seed'],hidden=(64,32),alpha=.1);path.write_bytes(pickle.dumps((m,scale),protocol=5))
  models[kind]=(m,scale);ev+=.5*nn_margin(m,scale,d[U:],s[U:]);print('NETWORK',kind,flush=True)
 old=np.load(W/'task5_round3/preference/evaluation.npz')['primary'];r={'evaluation_accuracy':accuracy(ev),'accepted_evaluation_accuracy':accuracy(old),'architecture':'Accepted two64/32 MLPs, same seeds71/73 and16/8 epochs, alpha.1; add8 Item2vec features to each. Existing accepted50/50 final structure retained; no new old/new-model blend.','networks':cfgs,'new_features':NAMES,'protocol':'One new underlying model only. All existing component features unchanged. Fixed accepted neural settings; train selection/penultimate pairs only. Final-event evaluation excluded from embedding and classifier fitting. Cohorts reused; local scores diagnostic.'};np.savez_compressed(OUT/'evaluation.npz',margin=ev);print('EVALUATION',r,flush=True);fz=matrices(True);p=0
 for kind,(m,scale) in models.items():d,s=fz[kind];p+=.5*nn_margin(m,scale,d,s)
 assert np.isfinite(p).all();q=pd.read_csv(DATA/'task_5_test.csv');f=q.copy();f['which']=np.where(p>=0,f.item1,f.item2);dest=Path('/workspace/output/task5_item2vec_neural');dest.mkdir(exist_ok=True);path=dest/'task_5_predictions.csv';f.to_csv(path,index=False);read=pd.read_csv(path);assert len(read)==86786 and read.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert ((read.which==read.item1)|(read.which==read.item2)).all();np.savez_compressed(OUT/'test.npz',margin=p);old=pd.read_csv('/workspace/output/task5_round3_primary/task_5_predictions.csv');r['output_audit']={'rows':len(read),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'changed_vs_accepted':int((read.which!=old.which).sum())};r['input_hashes']={n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['task_5_train.csv','task_5_test.csv']};(OUT/'results.json').write_text(json.dumps(r,indent=2));print('EXPORTED',r['output_audit'],flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['validation','full','combine'],required=True);a=ap.parse_args();combine() if a.stage=='combine' else features(a.stage=='full')
