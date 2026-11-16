"""Independent FISM score formulas and finite-difference BPR gradient check."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import sys,argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from task4_fism import library,OUT
from task4_two_approaches import queries

def numerical():
 lib=library();A=np.array([[.2,.1],[.1,.3],[-.1,.2]],'f');B=np.array([[.4,-.2],[.2,.5],[-.1,.2]],'f');bias=np.array([.05,0,-.02],'f');old=np.r_[A[1],B[0],B[2],bias[0],bias[2]].astype(float)
 def loss(v):
  h,p,j=v[:2],v[2:4],v[4:6];diff=v[6]-v[7]+h@(p-j);return np.logaddexp(0,-diff)+.005*(v@v)
 grad=np.array([(loss(old+np.eye(8)[k]*1e-5)-loss(old-np.eye(8)[k]*1e-5))/2e-5 for k in range(8)]);users=np.array([0],'i');items=np.array([0],'i');ptr=np.array([0,1],'i');sp=np.array([0,2],'i');seen=np.array([0,1],'i');nb=np.array([[1]],'i');coef=np.array([[1]],'f');cnt=np.array([1],'i');lib.train(1,2,1,3,users,items,ptr,sp,seen,nb,coef,cnt,A,B,bias,0,1,937);actual=np.r_[A[1],B[0],B[2],bias[0],bias[2]];assert np.allclose(actual,old-.025*grad,atol=1e-6)
 # Exact dense pooled-context formula against native score, including target exclusion.
 ptr=np.array([0,3,3],'i');hist=np.array([0,1,2],'i');times=np.array([0,86400,172800.],'d');qu=np.array([0,0,1],'i');qi=np.array([2,0,1],'i');qt=np.full(3,129600.,dtype='d');out=np.zeros((3,4),'f');lib.score(3,2,32,ptr,hist,times,qu,qi,qt,A,B,bias,out)
 for r in range(2):
  good=hist!=qi[r];h=hist[good];t=times[good];w=np.exp(-abs(t-qt[r])/86400/180);x=w@A[h]/np.sqrt(w.sum());uniform=A[h].sum(0)/np.sqrt(len(h));past=h[t<qt[r]];expected=[bias[qi[r]]+x@B[qi[r]],bias[qi[r]]+uniform@B[qi[r]],x@B[qi[r]]/max(1e-8,np.linalg.norm(x)*np.linalg.norm(B[qi[r]])),bias[qi[r]]+A[past].sum(0)@B[qi[r]]/np.sqrt(len(past)) if len(past) else 0];assert np.allclose(out[r],expected,atol=1e-6)
 assert (out[2]==0).all();print('Finite-difference BPR parameter updates and independent dense temporal FISM scoring passed',flush=True)

def heldout():
 from scipy.sparse import csr_matrix
 u,i,t,qu,qi,qt,U,I=queries(False);X=csr_matrix((np.ones(len(u)),(u,i)),shape=(U,I));N=len(qu)//2;assert np.asarray(X[qu[:N],qi[:N]]).sum()==0;z=np.load(OUT/'validation_features.npz');assert all(len(z[k])==len(qu) for k in z.files);old=np.load(Path('/workspace/work/task4/results/pair_features.npz'));assert np.array_equal(qu,old['users']) and np.array_equal(qi,old['items']);print('Heldout positive exclusion and accepted-bank query alignment passed',flush=True)

def replay():
 from task4_fism_combiner import augmented
 from task4_tree_combiners import lgb,W,DATA
 lib=library();u,i,t,qu,qi,qt,U,I=queries(True);ix=np.lexsort((np.arange(len(u)),t,u));ix=ix[(t[ix]>=0)&(t[ix]<4102444800)];uu=u[ix].astype('i');ii=i[ix].astype('i');tt=t[ix].astype('d');ptr=np.searchsorted(uu,np.arange(U+1)).astype('i');z=np.load(OUT/'full_factors.npz');A,B,bias=z['history'],z['target'],z['bias'];rows=np.random.RandomState(419).choice(len(qu),64,replace=False);value=np.zeros((64,4),'f');lib.score(64,A.shape[1],32,ptr,ii,tt,qu[rows].astype('i'),qi[rows].astype('i'),qt[rows].astype('d'),A,B,bias,value);features=np.load(OUT/'full_features.npz');expected=np.column_stack([features[k][rows] for k in features.files if k.startswith('fism_e30_')]);assert np.allclose(value,expected,atol=1e-6)
 d,s,_=augmented(True);m=lgb.Booster(model_file=str(OUT/'lightgbm.txt'));p=m.predict(np.c_[d,s],num_threads=2)-m.predict(np.c_[-d,s],num_threads=2);assert np.allclose(p,np.load(OUT/'test_margins.npz')['margin'],atol=1e-7);q=pd.read_csv(DATA/'task_4_test.csv');f=pd.read_csv('/workspace/output/task4_fism_lightgbm/task_4_predictions.csv');assert f.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert np.array_equal(f.which,np.where(p>=0,q.item1,q.item2));report={'factor_real_query_replay':64,'serialized_LightGBM_replay':True,'rows':len(f),'heldout_exclusion':True,'query_alignment':True,'numerical_gradient_and_dense_formula':True};(OUT/'audit.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--heldout',action='store_true');ap.add_argument('--replay',action='store_true');a=ap.parse_args();numerical()
 if a.heldout:heldout()
 if a.replay:replay()
