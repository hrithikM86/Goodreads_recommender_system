"""Independent neighborhood arithmetic and exported pair-score replay checks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from tasks24_reader_neighbors import compute
from tasks245_round4 import load,pair,transform,W,DATA

def arithmetic():
 X=csr_matrix(np.array([[1,1,0],[1,0,1],[0,1,1]],dtype='f'));T=X.copy();T.data=np.array([10,20,12,25,18,30.]);qu=np.array([0,0,1]);qi=np.array([2,0,1]);qt=np.array([26.,10.,22.]);out=compute(X,T,qu,qi,qt,True)
 pop=np.asarray(X.sum(0)).ravel();z=X.toarray()/np.sqrt(pop+10);z/=np.linalg.norm(z,axis=1)[:,None];sim=z@z.T;np.fill_diagonal(sim,0)
 for q,(u,i,t) in enumerate(zip(qu,qi,qt)):
  ids=np.flatnonzero(X.toarray()[:,i]);w=sim[u,ids];assert np.isclose(out['readers50_sum'][q],sum(w));assert np.isclose(out['readers50_time30'][q],sum(w*np.exp(-abs(T.toarray()[ids,i]-t)/30)))
 print('Dense independent reader/time calculation passed',flush=True)

def audit(task,readers=False):
 tag=f'task{task}_round4'+('_readers' if readers else '');out=W/tag;r=json.loads((out/'results.json').read_text());z=np.load(out/'test_margins.npz');d,s,_=load(task,True)
 if readers:
  a=np.load(W/f'task{task}_round4/readers_full.npz');dd,ss=pair(np.column_stack([a[k] for k in a.files]));d,s=np.c_[d,dd],np.c_[s,ss]
 ix=np.random.RandomState(2026).choice(len(d),min(512,len(d)),replace=False);sc=np.load(out/'transform_scale.npy');X=transform(d[ix],s[ix],sc);Xm=transform(-d[ix],s[ix],sc);pp=r['weights'].get('reference',0)*z['reference'][ix]
 for k,w in r['weights'].items():
  if k=='reference':continue
  obj=pickle.loads((out/(k+'_evaluation.pkl')).read_bytes())
  if isinstance(obj,tuple):m,scale=obj;pa=m.predict_proba(scale.transform(X))[:,1];pb=m.predict_proba(scale.transform(Xm))[:,1]
  else:pa=obj.predict_proba(X)[:,1];pb=obj.predict_proba(Xm)[:,1]
  assert np.allclose(pa-pb,-(pb-pa));pp+=w*(pa-pb)
 assert np.allclose(pp,z['primary'][ix],atol=1e-7)
 test=pd.read_csv(DATA/f'task_{task}_test.csv');accepted=Path('/workspace/output')/({2:'task2_with_zero',4:'task4_conservative',5:'task5_round3_primary'}[task])/f'task_{task}_predictions.csv';old=pd.read_csv(accepted)
 assert np.array_equal(old.which,np.where(z['reference']>=0,test.item1,test.item2))
 report={'sampled_model_replay':len(ix),'accepted_reference_replay':True,'outputs':{}}
 for name,value in [('primary',z['primary']),('conservative',.5*z['primary']+.5*z['reference'])]:
  path=Path('/workspace/output')/(tag+'_'+name)/f'task_{task}_predictions.csv';f=pd.read_csv(path);assert len(f)==len(test);assert f.drop(columns='which').equals(test.drop(columns='which',errors='ignore'));assert ((f.which==f.item1)|(f.which==f.item2)).all();assert np.isfinite(value).all();assert np.array_equal(f.which,np.where(value>=0,test.item1,test.item2));report['outputs'][name]={'rows':len(f),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'changed_vs_accepted':int((f.which!=old.which).sum())}
 (out/'audit.json').write_text(json.dumps(report,indent=2));print('AUDIT',task,tag,report,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',type=int);p.add_argument('--readers',action='store_true');a=p.parse_args();arithmetic()
 if a.task:audit(a.task,a.readers)
