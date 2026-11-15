"""Independent numerical and exclusion checks for next-round models."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import ctypes,subprocess,numpy as np

def numerical():
 path=Path('/tmp/task5_metric_check.so');subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_metric_kernel.cpp')),'-o',str(path)],check=True);lib=ctypes.CDLL(str(path));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.metric_train.argtypes=[ctypes.c_int]*2+[ip]*5+[ctypes.c_int,ip,ctypes.c_int]+[fp]*4+[ctypes.c_int]*4+[ctypes.c_float]*2
 ptr=np.array([0,2],np.int32);seq=np.array([0,1],np.int32);active=np.array([2],np.int32);lr=.02;reg=.01
 for kind in [0,1]:
  P=np.array([[.1,-.05]],'f');Q=np.array([[.1,.05],[-.1,.02],[.03,-.08]],'f');T=np.array([.03,.02],'f');b=np.array([.01,-.02,.03],'f');p=P[0].astype(float);l,i,j=Q.astype(float);glo=T.astype(float);cc=p+l+glo
  score=lambda PP,QQ,TT,bb:bb[1]-bb[2]+(.5*(np.sum((PP[0]-QQ[2])**2)-np.sum((PP[0]-QQ[1])**2)+np.sum((QQ[0]-QQ[2])**2)-np.sum((QQ[0]-QQ[1])**2)) if kind==0 else np.sum((PP[0]+QQ[0]+TT-QQ[2])**2)-np.sum((PP[0]+QQ[0]+TT-QQ[1])**2))
  g=1/(1+np.exp(score(P.astype(float),Q.astype(float),T.astype(float),b.astype(float))));arrays=[P.astype(float),Q.astype(float),T.astype(float),b.astype(float)];expect=[]
  for a,x in enumerate(arrays):
   grad=np.zeros_like(x)
   for ix in np.ndindex(x.shape):
    plus=[z.copy() for z in arrays];minus=[z.copy() for z in arrays];plus[a][ix]+=1e-6;minus[a][ix]-=1e-6;grad[ix]=(score(*plus)-score(*minus))/2e-6
   # PRME has no global term; unobserved item bias 0 is unchanged.
   update=g*grad-reg*x
   if a==2 and kind==0:update[:]=0
   if a==3:update[0]=0
   expect.append(x+lr*update)
  lib.metric_train(1,2,ptr,seq,ptr,seq,seq,2,active,1,P,Q,T,b,kind,0,1,42,lr,reg)
  for actual,target in zip([P,Q,T,b],expect):assert np.allclose(actual,target,atol=2e-7),(kind,actual,target)
 from task6_next import peers
 i=np.array([0,1,0,1,0,1]);u=np.array([0,0,0,1,1,1]);t=np.array([0,86400,3*86400,0,2*86400,6*86400],float);allowed=np.ones(6,bool);args=(np.array([1]),np.array([86400.]),np.array([0]),np.array([1.]))
 first=peers(i,t,u,allowed,*args);changed=t.copy();changed[2]=100*86400;second=peers(i,changed,u,allowed,*args);assert np.allclose(first,second,equal_nan=True),'own target influenced peers'
 assert first[0,0]>0
 print('PASS: metric gradients match finite differences; query-user target excluded from peer duration features.',flush=True)
if __name__=='__main__':numerical()

# Full export verification is callable after the component fits complete.
def audit_exports(repo,root):
 import json,hashlib,pandas as pd
 from catboost import CatBoostRegressor,CatBoostClassifier
 from types import SimpleNamespace
 from task1_next_stack import matrices as rating_features
 from task5_next_stack import matrices as sequence_features
 from task4_stacking import margin
 from task6_next_models import framing,forecast,anchor
 from scipy.linalg import cho_solve,cho_factor
 # EASE zero diagonal and constrained column optimum, independently on a toy graph.
 Z=np.array([[1,1,0],[0,1,1],[1,0,1],[1,0,0]],float);G=Z.T@Z;P=cho_solve(cho_factor(G+50*np.eye(3)),np.eye(3));B=-P/np.diag(P)[None,:];np.fill_diagonal(B,0)
 for j in range(3):
  nb=np.arange(3)!=j;assert np.allclose(B[nb,j],np.linalg.solve(G[np.ix_(nb,nb)]+50*np.eye(2),G[nb,j]))
 data=repo/'Assignment 1/assignment1_2026';legacy=root
 a=SimpleNamespace(data_dir=data,legacy=legacy,previous=root/'task1_improved',out=root/'task1_next',original=Path('/workspace/output/task_1_predictions.csv'))
 X,names,test,n=rating_features(a,True);r=json.loads((a.out/'results.json').read_text());raw=0
 for key,w in r['depth_weights'].items():
  if w==0:continue
  M=CatBoostRegressor();M.load_model(str(a.out/(key+'_full.cbm')));raw+=w*np.clip(M.predict(X),1,5)
 old=pd.read_csv(root/'task1_improved/new/task_1_predictions.csv').rating.to_numpy();expected={1:r['new_weight']*raw+(1-r['new_weight'])*old}
 # Check final SVD++ saved parameters independently against exported component scores.
 from task1_improve import dataset
 p,tr,q,u,i,qu,qi,users,items,n=dataset(data,True)
 for kind,key in [('svdpp','svdpp64low_e180'),('svdppstrong','svdppstrong96_e140'),('itemspp','itemspp32_e100')]:
  if kind=='itemspp':qu,qi=qi,qu
  z=np.load(a.out/'full'/(kind+'_final.npz'));pred=np.full(len(q),float(z['mu']));ku=qu>=0;ki=qi>=0;pred[ku]+=z['bu'][qu[ku]];pred[ki]+=z['bi'][qi[ki]];ok=ku&ki;pred[ok]+=np.einsum('ij,ij->i',z['effective'][qu[ok]],z['Q'][qi[ok]]);assert np.allclose(np.clip(pred,1,5),np.load(a.out/'full'/(kind+'.npz'))[key],atol=1e-6)
 print('PASS: Task1 final parameter replay.',flush=True)
 a=SimpleNamespace(data_dir=data,baseline=root/'task5/results',previous=root/'task5_improved',out=root/'task5_next');d,s,od,oscore,names=sequence_features(a,True);r=json.loads((a.out/'results.json').read_text());raw=0
 for key,w in r['depth_weights'].items():
  if w==0:continue
  M=CatBoostClassifier();M.load_model(str(a.out/(key+'_evaluation.cbm')));raw+=w*margin(M,d,s)
 prev=json.loads((a.previous/'results.json').read_text())['selected']['weights'];old=0
 for key,w in prev.items():
  M=CatBoostClassifier();M.load_model(str(a.previous/(key+'_full.cbm')));old+=w*margin(M,od,oscore)
 test=pd.read_csv(data/'task_5_test.csv');assert np.array_equal(pd.read_csv(root/'task5_improved/primary/task_5_predictions.csv').which,np.where(old>=0,test.item1,test.item2));expected[5]=r['new_weight']*raw+(1-r['new_weight'])*old
 # Final metric scores must replay from saved factors (restrict to known previous items).
 from task5_improve import data as seqdata
 h,contexts,qu,qi,U,I,test=seqdata(data,a.baseline,True);ctx=np.tile(np.arange(len(contexts)),2);last=np.array([h[-1] for h in contexts]);zs=np.load(a.out/'full/metrics.npz')
 for kind,name in [(0,'prme'),(1,'transrec')]:
  z=np.load(a.out/'full'/(name+'_model.npz'));known=qi>=0;known[known]&=z['pop'][qi[known]]>0;ok=known.copy();ok[known]&=z['pop'][last[ctx[known]]]>0;qq=qi[ok];us=qu[ok];prev=last[ctx[ok]];P=z['P'];Q=z['Q'];score=(.5*(-np.sum((P[us]-Q[qq])**2,1)-np.sum((Q[prev]-Q[qq])**2,1)) if kind==0 else -np.sum((P[us]+Q[prev]+z['globalv']-Q[qq])**2,1))+z['bias'][qq];assert np.allclose(score,zs[name+'_e120_joint'][ok],atol=1e-6)
 z=dict(np.load(a.out/'full/fossil_model.npz'));fs=np.load(a.out/'full/fossil.npz');sample=np.random.RandomState(77).choice(len(qi),512,replace=False)
 for row in sample:
  item=qi[row]
  if item<0 or z['pop'][item]==0:assert fs['fossil_e35_joint'][row]==0;continue
  hist=contexts[ctx[row]];hh=hist[-20:];last3=hist[-3:][::-1];us=qu[row];vec=z['A'][hh].sum(0)/np.sqrt(len(hh))+(z['wg'][:len(last3)]+z['wu'][us,:len(last3)])@z['A'][last3];score=vec@z['B'][item]+z['bias'][item];assert np.isclose(score,fs['fossil_e35_joint'][row],atol=1e-5)
 print('PASS: Task5 metric, FOSSIL and classifier replay.',flush=True)
 a=SimpleNamespace(out=root/'task6_next');r=json.loads((a.out/'stack_results.json').read_text());f=np.load(root/'task6_improved/full.npz');peer=np.load(a.out/'full_peer.npz');old=pd.read_csv(root/'task6_improved/blend/task_6_predictions.csv').delta.to_numpy()/86400;Q=np.c_[f['Q'],peer['Q'],np.log1p(old)];correction=0
 for key,w in r['weights'].items():
  if w==0:continue
  M=CatBoostRegressor();M.load_model(str(a.out/(key+'_full.cbm')));correction+=w*M.predict(Q)
 pd.testing.assert_frame_equal(pd.read_csv(root/'task6_improved/blend/task_6_predictions.csv').iloc[:,:-1],pd.read_csv(data/'task_6_test.csv').iloc[:,:-1]);expected[6]=np.maximum(0,old+r['correction_weight']*correction)*86400
 # Rebuild peer features for random samples with raw events.
 from task6_pipeline import load_data
 from task6_next import peers
 p,test,u,users,i,items,t,ratings,starts,early,v,e,rows=load_data(data);cached=np.load(root/'task6_improved/validation.npz');sample=np.random.RandomState(73).choice(len(rows),20,replace=False);allowed=np.ones(len(p),bool);allowed[v]=False;allowed[e]=False;got=peers(i,t,u,allowed,i[rows[sample]],t[rows[sample]-1],u[rows[sample]],cached['X'][sample,19]);assert np.allclose(got,np.load(a.out/'validation_peer.npz')['X'][sample],equal_nan=True,atol=1e-5)
 audit={}
 for task,pred in expected.items():
  folder=root/f'task{task}_next';variant='stack_primary' if task==6 else 'primary';frame=pd.read_csv(folder/variant/f'task_{task}_predictions.csv');test=pd.read_csv(data/f'task_{task}_test.csv');pd.testing.assert_frame_equal(frame.iloc[:,:-1],test.iloc[:,:-1]);assert not frame.isna().any().any()
  if task==5:assert np.array_equal(frame.which,np.where(pred>=0,test.item1,test.item2))
  else:assert np.allclose(frame.iloc[:,-1],pred,rtol=1e-9,atol=1e-6) and np.isfinite(pred).all();assert (pred>=0).all()
  result={'rows':len(frame),'sha256':hashlib.sha256((folder/variant/f'task_{task}_predictions.csv').read_bytes()).hexdigest(),'ids_order_preserved':True,'full_model_replay':True};resultfile=folder/('stack_results.json' if task==6 else 'results.json');summary=json.loads(resultfile.read_text());summary['audit']=result;summary['inputs']={name:hashlib.sha256((data/name).read_bytes()).hexdigest() for name in (['tasks_1_2_train.csv','task_1_test.csv'] if task==1 else [f'task_{task}_train.csv',f'task_{task}_test.csv'])};resultfile.write_text(json.dumps(summary,indent=2));audit[task]=result
 print(json.dumps(audit,indent=2));return audit

def fossil_gradient():
 so=Path('/tmp/fossil_check.so');subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_fossil_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.fossil_train.argtypes=[ctypes.c_int]*2+[ip]*5+[ctypes.c_int]*4+[ctypes.c_float]*2+[fp]*5
 ptr=np.array([0,2],np.int32);seq=np.array([0,1],np.int32);active=np.array([2],np.int32);A=np.array([[.1,.2],[.03,-.04],[-.1,.02]],'f');B=np.array([[.1,-.1],[.2,.05],[-.03,.06]],'f');wu=np.zeros((1,3),'f');wg=np.array([.5,.25,.1],'f');bias=np.array([0,.03,-.01],'f');arrays=[x.astype(float) for x in [A,B,wu,wg,bias]]
 def score(A,B,wu,wg,b):return b[1]-b[2]+((1+wu[0,0]+wg[0])*A[0])@(B[1]-B[2])
 g=1/(1+np.exp(score(*arrays)));expected=[]
 for a,x in enumerate(arrays):
  grad=np.zeros_like(x)
  for ix in np.ndindex(x.shape):
   plus=[z.copy() for z in arrays];minus=[z.copy() for z in arrays];plus[a][ix]+=1e-6;minus[a][ix]-=1e-6;grad[ix]=(score(*plus)-score(*minus))/2e-6
  upd=g*grad-.03*x
  if a==0:upd[1:]=0
  if a==1:upd[0]=0
  if a==2:upd[:,1:]=0
  if a==3:upd[1:]=0;upd*=.1
  if a==4:upd[0]=0
  expected.append(x+.015*upd)
 lib.fossil_train(1,2,ptr,seq,ptr,seq,active,1,0,1,71,.015,.03,A,B,wu,wg,bias)
 for x,y in zip([A,B,wu,wg,bias],expected):assert np.allclose(x,y,atol=2e-7),(x,y)
 print('PASS: FOSSIL update matches finite differences, including shared/global and per-user transition weights.',flush=True)
