"""Independent sequence-pooling formulas, cached embedding samples and neural replay."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,pickle,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from task5_item2vec import libraries,OUT,matrices
from task5_boosting import W,DATA
from task5_improve import data
from task5_round3 import nn_margin

def dense(ids,target,E,V,Ot,Vt):
 z=np.zeros(8);n=len(ids)
 if np.linalg.norm(Vt)==0:return z
 good=(ids!=target)&(np.linalg.norm(V,axis=1)>0);distance=np.arange(n)[::-1];c=V@Vt;raw=E@Ot
 if n and good[-1]:z[0]=c[-1];z[5]=raw[-1]
 for col,limit in [(1,3),(2,10)]:
  ok=good&(distance<limit)
  if ok.any():z[col]=c[ok].mean()
 ok=good&(distance<3)
 if ok.any():z[6]=raw[ok].mean()
 if good.any():
  weights=.9**distance[good];z[3]=c[good]@weights/weights.sum();z[4]=c[good].max();z[7]=raw[good]@weights/weights.sum()
 return z

def numerical():
 _,lib=libraries();E=np.array([[1,0],[0,1],[1,.5],[0,0]],'f');O=np.array([[.1,.2],[.2,.5],[.3,.7],[0,0]],'f');V=E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-8);ptr=np.array([0,3,5,6],'i');hist=np.array([0,2,1,1,2,0],'i');qi=np.array([2,-1,3],'i');out=np.zeros((3,8),'f');lib.score(3,2,ptr,hist,qi,E,O,V,out);assert np.allclose(out[0],dense(hist[:3],2,E[hist[:3]],V[hist[:3]],O[2],V[2]),atol=1e-6);assert (out[1:]==0).all();print('Independent dense recent-history/decay/SGNS scoring, self exclusion, unknown/cold handling passed',flush=True)

def sample_replay():
 for stage in ['validation','full']:
  z=np.load(OUT/(stage+'_replay.npz'));scores=np.load(OUT/(stage+'_features.npz'))['scores']
  for j,row in enumerate(z['rows']):
   lo,hi=z['offset'][j:j+2];value=dense(z['history_ids'][lo:hi],z['target_ids'][j],z['history_E'][lo:hi],z['history_V'][lo:hi],z['target_O'][j],z['target_V'][j]);assert np.allclose(value,scores[row],atol=2e-6)
 print('Sampled embedding/context replay passed:128 validation and128 test candidate rows',flush=True)

def heldout():
 h,ctx,qu,qi,U,I,_=data(DATA,W/'task5/results',False)
 for r in range(2*U):assert qi[r] not in ctx[r]
 pop=np.bincount(np.concatenate(h),minlength=I);report=json.loads((OUT/'validation_report.json').read_text());assert report['training_events']==sum(int((pop[x]>=2).sum()) for x in h);assert len(np.load(OUT/'validation_features.npz')['scores'])==len(qi);print('Chronological target exclusion and prefix-only embedding fitting counts passed',flush=True)

def replay():
 z=matrices(True);p=0
 for name in ['sequence','preference']:
  m,sc=pickle.loads((OUT/(name+'_evaluation.pkl')).read_bytes());d,s=z[name];assert d.shape[1]==(232 if name=='sequence' else 268);value=nn_margin(m,sc,d,s);assert np.allclose(value,-nn_margin(m,sc,-d,s));p+=.5*value
 assert np.allclose(p,np.load(OUT/'test.npz')['margin'],atol=1e-7);q=pd.read_csv(DATA/'task_5_test.csv');path=Path('/workspace/output/task5_item2vec_neural/task_5_predictions.csv');f=pd.read_csv(path);assert len(f)==86786 and f.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert np.array_equal(f.which,np.where(p>=0,q.item1,q.item2));old=pd.read_csv('/workspace/output/task5_round3_primary/task_5_predictions.csv');oldp=np.load(W/'task5_round3/preference/test_predictions.npz')['primary'];assert np.array_equal(old.which,np.where(oldp>=0,q.item1,q.item2));r={'independent_dense_pooling':True,'unknown_cold_and_self_handling':True,'sampled_embedding_replay_rows':256,'chronological_target_exclusion':True,'prefix_training_count_verified':True,'two_network_saved_model_replay':True,'pair_swap_antisymmetry':True,'accepted_reference_replay':True,'rows':len(f),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};(OUT/'audit.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');a=ap.parse_args();numerical();sample_replay();heldout()
 if a.replay:replay()
