"""Independent dense Item2vec scoring, SGNS learnability, heldout and model replay checks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import argparse,json,ctypes
from pathlib import Path
import numpy as np,pandas as pd
from task4_item2vec import library,OUT
from task4_tree_combiners import DATA,xgb,lgb,predict
from task4_two_approaches import queries

def numerical():
 lib=library();E=np.array([[1,0],[0,1],[1,.5],[0,0]],dtype='f');O=np.array([[.2,.1],[.5,-.2],[.3,.7],[0,0]],dtype='f');V=E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-8);ptr=np.array([0,2,4],'i');hist=np.array([0,1,1,2],'i');times=np.array([0,86400,0,172800.],'d');qu=np.array([0,1,0],'i');qi=np.array([2,2,3],'i');qt=np.full(3,129600.,dtype='d');out=np.zeros((3,12),'f');lib.score(3,2,ptr,hist,times,qu,qi,qt,E,O,V,out)
 for q in range(2):
  lo,hi=ptr[qu[q]:qu[q]+2];hh=hist[lo:hi];tt=times[lo:hi];keep=hh!=qi[q];hh,tt=hh[keep],tt[keep];c=V[hh]@V[qi[q]];raw=E[hh]@O[qi[q]];assert np.isclose(out[q,0],c.mean());assert np.isclose(out[q,1],c.max());
  for col,h in enumerate([30,180,730],6):
   w=np.exp(-abs(tt-qt[q])/86400/h);assert np.isclose(out[q,col],c@w/w.sum())
  assert np.isclose(out[q,9],raw.mean());w=np.exp(-abs(tt-qt[q])/86400/180);assert np.isclose(out[q,10],raw@w/w.sum())
 assert np.all(out[2]==0)
 rng=np.random.RandomState(81);items=np.r_[np.tile([0,1],100),np.tile([2,3],100)].astype('i');users=np.r_[np.zeros(200),np.ones(200)].astype('i');pool=np.tile(np.arange(4,dtype='i'),100);E=rng.uniform(-.05,.05,(4,8)).astype('f');O=np.zeros_like(E);lib.train(len(items),8,3,3,users,items,pool,len(pool),E,O,0,10,711);assert E[0]@O[1]>E[0]@O[2];assert np.isfinite(E).all();print('Independent dense temporal scoring, cold-book handling and SGNS learnability passed',flush=True)

def heldout():
 from scipy.sparse import csr_matrix
 u,i,t,qu,qi,qt,U,I=queries(False);X=csr_matrix((np.ones(len(u)),(u,i)),shape=(U,I));n=len(qu)//2;assert np.asarray(X[qu[:n],qi[:n]]).sum()==0;original=np.load(Path('/workspace/work/task4/results/pair_features.npz'));assert np.array_equal(qu,original['users']) and np.array_equal(qi,original['items']);z=np.load(OUT/'validation_features.npz');assert all(len(z[k])==len(qu) for k in z.files);print('Every heldout positive excluded from fitted histories; query-score dimensions verified',flush=True)

def embedding_replay():
 lib=library();u,i,t,qu,qi,qt,U,I=queries(True);valid=(t>=0)&(t<4102444800);order=np.lexsort((np.arange(len(u)),t,u));order=order[valid[order]];uu=u[order].astype('i');ii=i[order].astype('i');tt=t[order].astype('d');counts=np.bincount(ii,minlength=I);keep=counts[ii]>=2;uu,ii,tt=uu[keep],ii[keep],tt[keep];ptr=np.searchsorted(uu,np.arange(U+1)).astype('i');z=np.load(OUT/'full_embeddings.npz');E=z['input'];O=z['output'];V=E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-8);ix=np.random.RandomState(92).choice(len(qu),64,replace=False);value=np.zeros((len(ix),12),'f');lib.score(len(ix),E.shape[1],ptr,ii,tt,qu[ix].astype('i'),qi[ix].astype('i'),qt[ix].astype('d'),E,O,V,value);cache=np.load(OUT/'full_features.npz');names=[k for k in cache.files if k.startswith('item2vec_e12_')];expected=np.column_stack([cache[k][ix] for k in names]);assert np.allclose(value,expected,atol=1e-6);print('Final Item2vec embeddings replay 64 sampled real queries',flush=True)

def replay():
 from task4_item2vec_combiners import augmented
 d,s,_=augmented(True);report={}
 for kind in ['xgboost','lightgbm']:
  if kind=='xgboost':m=xgb.XGBClassifier(n_jobs=2);m.load_model(OUT/(kind+'.ubj'));p=predict(m,kind,d,s)
  else:m=lgb.Booster(model_file=str(OUT/(kind+'.txt')));p=m.predict(np.c_[d,s],num_threads=2)-m.predict(np.c_[-d,s],num_threads=2)
  assert np.allclose(p,np.load(OUT/(kind+'_test.npz'))['margin'],atol=1e-7);q=pd.read_csv(DATA/'task_4_test.csv');f=pd.read_csv(Path('/workspace/output')/('task4_item2vec_'+kind)/'task_4_predictions.csv');assert f.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert np.array_equal(f.which,np.where(p>=0,q.item1,q.item2));report[kind]={'rows':len(f),'serialized_model_replay':True}
 (OUT/'replay_audit.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');ap.add_argument('--heldout',action='store_true');a=ap.parse_args();numerical()
 if a.heldout:heldout()
 if a.replay:embedding_replay();replay()
