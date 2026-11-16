"""Finite-difference native FPMC update, query-time context and artifact replay checks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from task4_fpmc import library,contexts,score,OUT
from task4_two_approaches import queries

def numerical():
 lib=library();P=np.array([[.1,.3]],'f');Q=np.array([[.2,-.1],[.3,.2],[-.2,.1]],'f');A=np.array([[.1,.4],[.2,-.3],[-.1,.1]],'f');B=np.array([[.3,.1],[.1,-.2],[-.2,.2]],'f');bias=np.array([.03,0,-.02],'f');old=np.r_[P[0],Q[0],Q[2],A[1],B[0],B[2],bias[0],bias[2]].astype(float)
 def loss(v):
  p,qi,qj,a,bi,bj=v[:2],v[2:4],v[4:6],v[6:8],v[8:10],v[10:12];diff=v[12]-v[13]+p@(qi-qj)+a@(bi-bj);return np.logaddexp(0,-diff)+.01*(v@v)
 grad=np.array([(loss(old+np.eye(14)[j]*1e-5)-loss(old-np.eye(14)[j]*1e-5))/2e-5 for j in range(14)]);u=np.array([0],'i');last=np.array([1],'i');target=np.array([0],'i');ptr=np.array([0,2],'i');seen=np.array([0,1],'i');pool=np.array([2],'i');active=np.array([2],'i');lib.train_fpmc(1,2,u,last,target,ptr,seen,pool,1,active,1,P,Q,A,B,bias,0,1,1049,.02,.02,.5,0);actual=np.r_[P[0],Q[0],Q[2],A[1],B[0],B[2],bias[0],bias[2]];assert np.allclose(actual,old-.02*grad,atol=1e-6)
 u=np.array([0,0,0,1,1]);i=np.array([0,1,2,1,3]);t=np.array([0,86400,86400,0,172800.]);qu=np.array([0,0,1,1]);qt=np.array([86400,129600,86400,-1.]);prev,nxt,pw,nw,*_=contexts(u,i,t,qu,qt,2);assert np.array_equal(prev,[0,2,1,-1]);assert np.array_equal(nxt,[-1,-1,3,-1]);assert pw[-1]==nw[-1]==0
 rng=np.random.RandomState(93);P=rng.normal(size=(2,3)).astype('f');Q=rng.normal(size=(4,3)).astype('f');A=rng.normal(size=(4,3)).astype('f');B=rng.normal(size=(4,3)).astype('f');bias=rng.normal(size=4).astype('f');qi=np.array([1,3,2,0]);values=score(P,Q,A,B,bias,qu,qi,prev,nxt,pw,nw)
 for r in range(4):
  personal=P[qu[r]]@Q[qi[r]]+bias[qi[r]];f=A[prev[r]]@B[qi[r]] if prev[r]>=0 else 0;b=A[qi[r]]@B[nxt[r]] if nxt[r]>=0 else 0;assert np.allclose([v[r] for v in values],[personal,f,b,personal+f,personal+pw[r]*f+nw[r]*b],atol=1e-6)
 print('Finite-difference FPMC SGD, strict-time/tie contexts and direct score formulas passed',flush=True)

def heldout():
 from scipy.sparse import csr_matrix
 u,i,t,qu,qi,qt,U,I=queries(False);X=csr_matrix((np.ones(len(u)),(u,i)),shape=(U,I));n=len(qu)//2;assert np.asarray(X[qu[:n],qi[:n]]).sum()==0;old=np.load('/workspace/work/task4/results/pair_features.npz');assert np.array_equal(qu,old['users']) and np.array_equal(qi,old['items']);prev,nxt,*_=contexts(u,i,t,qu,qt,U);assert np.array_equal(prev[:n],prev[n:]) and np.array_equal(nxt[:n],nxt[n:]);features=np.load(OUT/'validation_features.npz');assert all(len(features[k])==len(qu) for k in features.files);print('Heldout exclusion, bank alignment and candidate-independent query contexts passed',flush=True)

def replay():
 from task4_fpmc_combiner import augmented
 from task4_tree_combiners import lgb,DATA
 u,i,t,qu,qi,qt,U,I=queries(True);prev,nxt,pw,nw,*_=contexts(u,i,t,qu,qt,U);z=np.load(OUT/'full_factors.npz');ix=np.random.RandomState(419).choice(len(qu),64,replace=False);values=score(z['P'],z['Q'],z['A'],z['B'],z['bias'],qu[ix],qi[ix],prev[ix],nxt[ix],pw[ix],nw[ix]);cache=np.load(OUT/'full_features.npz');expected=np.column_stack([cache[k][ix] for k in cache.files if k.startswith('fpmc_e100_')]);assert np.allclose(np.column_stack(values),expected,atol=1e-6)
 d,s,_=augmented(True);m=lgb.Booster(model_file=str(OUT/'lightgbm.txt'));p=m.predict(np.c_[d,s],num_threads=2)-m.predict(np.c_[-d,s],num_threads=2);assert np.allclose(p,np.load(OUT/'test_margins.npz')['margin'],atol=1e-7);q=pd.read_csv(DATA/'task_4_test.csv');f=pd.read_csv('/workspace/output/task4_fpmc_lightgbm/task_4_predictions.csv');assert f.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert np.array_equal(f.which,np.where(p>=0,q.item1,q.item2));r={'native_gradient_and_dense_scores':True,'strict_time_and_tie_contexts':True,'heldout_exclusion':True,'candidate_independent_contexts':True,'query_bank_alignment':True,'factor_real_query_replay':64,'serialized_model_replay':True,'rows':len(f)};(OUT/'audit.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--heldout',action='store_true');ap.add_argument('--replay',action='store_true');a=ap.parse_args();numerical()
 if a.heldout:heldout()
 if a.replay:replay()
