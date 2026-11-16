"""Independent FM formula and learnability checks; final file alignment audits."""
import ctypes,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/workspace/work/task4_two_approaches');DATA=Path('/workspace/work/Goodreads_recommender_system/Assignment 1/assignment1_2026')
def check():
 lib=ctypes.CDLL(str(ROOT/'pair_fm.so'));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.predict.argtypes=[ctypes.c_int]*3+[ip]+[fp]*3;lib.train.argtypes=[ctypes.c_int]*5+[ip]*2+[fp]*2+[ctypes.c_int]*3;rng=np.random.RandomState(42);ids=np.array([[0,2,4,5],[1,3,4,-1]],dtype='i');w=rng.normal(0,.1,6).astype('f');V=rng.normal(0,.1,(6,3)).astype('f');out=np.empty(2,'f');lib.predict(2,4,3,ids,w,V,out);expected=[]
 for row in ids:
  good=row[row>=0];v=sum(w[good]);v+=sum(np.dot(V[good[a]],V[good[b]]) for a in range(len(good)) for b in range(a+1,len(good)));expected.append(v)
 assert np.allclose(out,expected,atol=1e-6)
 # Two-book corpus, one known positive: repeated logistic training must rank it above the alternate.
 ids=np.tile(np.array([0,1,3,4],dtype='i'),(100,1));pool=np.zeros(100,'i');w=np.zeros(5,'f');V=rng.normal(0,.03,(5,4)).astype('f');queries=np.array([[0,1,3,4],[0,2,3,4]],dtype='i');lib.train(100,4,4,2,1,ids,pool,w,V,0,5,32);out=np.empty(2,'f');lib.predict(2,4,4,queries,w,V,out);assert out[0]>out[1]+1;assert np.isfinite(V).all();print('Independent FM pair expansion and positive/negative learnability checks passed')
def replay():
 from catboost import CatBoostClassifier
 from sklearn.linear_model import LogisticRegression
 from scipy.special import expit
 r=json.loads((ROOT/'blend_results.json').read_text());a=np.load(ROOT/'accepted.npz');v=np.load(ROOT/'fm_validation.npz')[r['fm_checkpoint']];f=np.load(ROOT/'fm_full.npz')[r['fm_checkpoint']];n=len(v)//2;d=v[:n]-v[n:];nsel=71462;m=LogisticRegression(C=1,fit_intercept=False,max_iter=300);x=d[:nsel,None];m.fit(np.r_[x,-x],np.r_[np.ones(nsel),np.zeros(nsel)]);N=len(f)//2;p=(1-r['FM_weight'])*a['test']+r['FM_weight']*(m.predict_proba((f[:N]-f[N:])[:,None])[:,1]-.5);assert np.allclose(p,np.load(ROOT/'blend_margins.npz')['test'],atol=1e-8)
 if (ROOT/'fresh_margins.npz').exists():
  z=np.load(ROOT/'fresh_full/scores.npz')['scores'];fm=np.load(ROOT/'fm_full.npz');z=np.c_[z,np.column_stack([fm[k] for k in fm.files])];N=len(z)//2;d=z[:N]-z[N:];ss=z[:N]+z[N:];model=CatBoostClassifier();model.load_model(str(ROOT/'fresh_combiner.cbm'));p=model.predict_proba(np.c_[d,ss])[:,1]-model.predict_proba(np.c_[-d,ss])[:,1];assert np.allclose(p,np.load(ROOT/'fresh_margins.npz')['test'],atol=1e-7)
 print('Saved component/model ensemble replay passed')

def audit():
 test=pd.read_csv(DATA/'task_4_test.csv');report={}
 for name,margin_file in [('task4_accepted_plus_fm','blend_margins.npz'),('task4_all_models_retrained_fm','fresh_margins.npz')]:
  p=Path('/workspace/output')/name/'task_4_predictions.csv'
  if not p.exists():continue
  f=pd.read_csv(p);v=np.load(ROOT/margin_file)['test'];assert f.drop(columns='which').equals(test.drop(columns='which',errors='ignore'));assert np.isfinite(v).all();assert np.array_equal(f.which,np.where(v>=0,test.item1,test.item2));assert ((f.which==f.item1)|(f.which==f.item2)).all();report[name]={'rows':len(f),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'changed_vs_accepted':int((f.which!=pd.read_csv('/workspace/output/task4_time_bpr/task_4_predictions.csv').which).sum())}
 (ROOT/'audit.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':check();replay();audit()
