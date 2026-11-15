"""Independent loss/update checks and exported neural probability replay."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import ctypes,subprocess,numpy as np
from task1_neural import head_gradient,softmax,RATINGS

def numerical():
 rng=np.random.RandomState(41);logits=rng.normal(0,.3,(3,5));y=np.array([1,3,5])
 for lam in [0.,2.]:
  analytic=head_gradient(softmax(logits),y,lam);numeric=np.zeros_like(logits)
  def loss(z):
   p=softmax(z);return np.mean(-np.log(p[np.arange(3),y-1])+lam*(p@RATINGS-y)**2)
  for ix in np.ndindex(logits.shape):
   a=logits.copy();b=logits.copy();a[ix]+=1e-5;b[ix]-=1e-5;numeric[ix]=(loss(a)-loss(b))/2e-5
  assert np.allclose(analytic,numeric,atol=1e-7)
 so=Path('/tmp/neural_check.so');subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task1_neural_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.neural_train.argtypes=[ctypes.c_int]*2+[ip]*5+[fp]*12+[ctypes.c_int]*3+[ctypes.c_float]
 for lam in [0.,2.]:
  parameters=[np.array([[.1,.2]],'f'),np.array([[.2,-.1]],'f'),np.array([[.01,-.02,0,.03,-.01]],'f'),np.array([[-.01,.02,.01,-.02,0]],'f'),rng.normal(0,.1,(5,2)).astype('f'),rng.normal(0,.1,5).astype('f')];base=[p.astype(float) for p in parameters];expected=[]
  def loss(P,Q,bu,bi,W,b):
   p=softmax(np.tanh(P*Q)@W.T+bu+bi+b);return -np.log(p[0,2])+lam*(p[0]@RATINGS-3)**2
  regs=[.01,.01,5.,5.,.00001,0.]
  for j,x in enumerate(base):
   grad=np.zeros_like(x)
   for ix in np.ndindex(x.shape):
    plus=[z.copy() for z in base];minus=[z.copy() for z in base];plus[j][ix]+=1e-5;minus[j][ix]-=1e-5;grad[ix]=(loss(*plus)-loss(*minus))/2e-5+regs[j]*x[ix]
   expected.append(x-.05*grad/np.sqrt(1+grad*grad))
  u=np.array([0],np.int32);rating=np.array([2],np.int32);count=np.array([1],np.int32);acc=[np.zeros_like(p) for p in parameters];lib.neural_train(1,2,u,u,rating,count,count,*parameters,*acc,0,1,133,lam)
  for x,correct in zip(parameters,expected):assert np.allclose(x,correct,atol=2e-7)
 print('PASS: CE/expected-MSE softmax gradients and native neural-CF AdaGrad updates match finite differences.',flush=True)

if __name__=='__main__':numerical()

def audit(a):
 import json,hashlib,pandas as pd
 from task1_neural import inputs,transform,Neural
 from task1_neural_cf import probabilities
 from task1_improve import dataset
 summary=json.loads((a.out/'ensemble_results.json').read_text());single=json.loads((a.out/'results.json').read_text());X,names,q,n=inputs(a,True);X=transform(X,names);ref=pd.read_csv(a.accepted);pd.testing.assert_frame_equal(ref.iloc[:,:-1],q.iloc[:,:-1]);pred=summary['weights']['accepted']*ref.rating.to_numpy();prob=np.zeros((len(q),5));nnweight=1-summary['weights']['accepted']
 for name,cfg in single['runs'].items():
  w=summary['weights'][name]
  if w==0:continue
  model=Neural.load(a.out/(name+'_full.npz'));p=model.probabilities(X);assert p.shape==(20000,5) and np.allclose(p.sum(1),1,atol=1e-6);pred+=w*(p@RATINGS);prob+=w*p/nnweight
 expected={'ensemble_primary':pred,'ensemble_neural_only':prob@RATINGS,'ensemble_blend50':.5*pred+.5*ref.rating.to_numpy()};audits={}
 for name,values in expected.items():
  path=a.out/name/'task_1_predictions.csv';frame=pd.read_csv(path);pd.testing.assert_frame_equal(frame.iloc[:,:-1],q.iloc[:,:-1]);assert len(frame)==20000 and np.isfinite(frame.rating).all() and frame.rating.between(1,5).all();assert np.allclose(frame.rating,values,atol=1e-6,rtol=1e-9);audits[name]={'rows':20000,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'ids_order_preserved':True,'saved_model_replay':True}
 for full in [False,True]:
  part=a.out/('full' if full else 'validation');p,tr,q,u,i,qu,qi,users,items,n=dataset(a.data_dir,full);cf=np.load(part/'cf_probabilities.npz');assert (tr.rating>0).all()
  if not full:
   seen=np.sort(u*len(items)+i);target=qu*len(items)+qi;where=np.searchsorted(seen,target);ok=where<len(seen);assert not np.any(seen[where[ok]]==target[ok])
  for name in ['ce','ce_mse']:
   model=dict(np.load(part/(name+'_model.npz')));replay=probabilities(model,qu,qi);assert np.allclose(replay,cf[name+'_e25'],atol=1e-7)
 summary['audit']=audits;summary['checks']={'softmax_loss_gradient':True,'native_neural_Cf_update':True,'CF_parameter_replay':True,'five_class_probabilities_normalized':True,'ensemble_parameter_replay':True,'zero_targets_excluded':True,'heldout_pairs_absent_from_CF_training':True};summary['inputs']={name:hashlib.sha256((a.data_dir/name).read_bytes()).hexdigest() for name in ['tasks_1_2_train.csv','task_1_test.csv']};summary['source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['task1_neural_kernel.cpp','task1_neural_cf.py','task1_neural.py','task1_neural_ensemble.py','task1_neural_checks.py']};(a.out/'ensemble_results.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'checks':summary['checks'],'exports':audits},indent=2));return summary
