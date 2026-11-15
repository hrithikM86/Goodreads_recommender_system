"""Independent skip-graph, latent-order and candidate-attention checks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import subprocess,ctypes,numpy as np
from task5_order import transitions,diffusion,score_graph
from task5_era import ordering

def numerical():
 h=[np.array([0,1,2,3],dtype='i'),np.array([1,4,3],dtype='i')];graphs=dict(transitions(h,5))
 for window,H in graphs.items():
  expected=np.zeros((5,5))
  for hist in h:
   for left in range(len(hist)):
    for right in range(left+1,min(len(hist),left+window+1)):expected[hist[left],hist[right]]+=1/(np.sqrt(len(hist))*(right-left))
  assert np.allclose(H.toarray(),expected,atol=1e-6)
 theta=ordering(graphs[3],1);H=graphs[3].toarray().astype(float);degree=H.sum(0)+H.sum(1);exact=np.linalg.solve(np.diag(degree+1)-H-H.T,H.sum(0)-H.sum(1));assert np.allclose(theta,exact,atol=1e-6) and theta[0]<theta[3]
 so=Path('/tmp/order_check.so');subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_order_kernel.cpp')),'-o',str(so)],check=True);contexts=[np.array([0,1]),np.array([1,4])];qi=np.array([3,3,2,-1]);vals,P=score_graph(graphs[3],contexts,qi);paths=diffusion(P,contexts,qi,so);square=P.toarray()@P.toarray();assert np.isclose(paths['twohop_last'][0],square[1,3]) and paths['twohop_last'][3]==0
 so=Path('/tmp/attention_check.so');subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_attention_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.attention_update.argtypes=[ctypes.c_int]*6+[ip]+[fp]*6+[ctypes.c_float];rng=np.random.RandomState(53);P=rng.normal(0,.1,(1,2)).astype('f');Q=rng.normal(0,.1,(4,2)).astype('f');E=rng.normal(0,.1,(4,2)).astype('f');V=rng.normal(0,.1,(4,2)).astype('f');bias=rng.normal(0,.1,4).astype('f');position=(-np.log1p(np.arange(10))).astype('f');seq=np.array([0,1,2],dtype='i');params=[E,V,bias,position];base=[x.astype(float) for x in params]
 def loss(E,V,bias,position):
  scores=[]
  for j in [2,3]:
   s=E[[1,0]]@V[j];logits=s/np.sqrt(2)+position[:2];w=np.exp(logits-logits.max());w/=w.sum();scores.append(w@s+bias[j]+P[0].astype(float)@Q[j].astype(float))
  return np.logaddexp(0,-(scores[0]-scores[1]))
 expected=[]
 for a,x in enumerate(base):
  grad=np.zeros_like(x)
  for ix in np.ndindex(x.shape):
   plus=[z.copy() for z in base];minus=[z.copy() for z in base];plus[a][ix]+=1e-5;minus[a][ix]-=1e-5;grad[ix]=(loss(*plus)-loss(*minus))/2e-5
  mask=np.zeros_like(x)
  if a==0:mask[[0,1]]=1
  elif a in [1,2]:mask[[2,3]]=1
  else:mask[:2]=1
  grad=(grad+(.01 if a==3 else .02)*x)*mask;expected.append(x-.005*(.1 if a==3 else 1)*grad)
 lib.attention_update(2,0,0,2,2,3,seq,P,Q,*params,.005)
 for x,y in zip(params,expected):assert np.allclose(x,y,atol=2e-7),(x,y)
 print('PASS: skip transitions match direct enumeration; diffusion matches dense P²; community-order solve matches dense optimum; attention gradients match finite differences.',flush=True)

def audit(a):
 import json,hashlib,pickle,pandas as pd
 from catboost import CatBoostClassifier
 from task5_round3 import load,reference,margin,nn_margin
 from task5_improve import data
 from task5_attention import scores
 from task5_era import era_scores
 d,s,old,olds,ed,es,names=load(a,True);r=json.loads((a.out/'results.json').read_text());h,ctx,qu,qi,U,I,test=data(a.data_dir,a.baseline,True);ref=reference(a,old,olds,ed,es,'evaluation');assert np.array_equal(pd.read_csv(a.accepted).which,np.where(ref>=0,test.item1,test.item2));pred=0
 for key,w in r['weights'].items():
  if w==0:continue
  if key=='nn':model,scale=pickle.loads((a.out/'nn_evaluation.pkl').read_bytes());value=nn_margin(model,scale,d,s)
  else:model=CatBoostClassifier();model.load_model(str(a.out/(key+'_evaluation.cbm')));value=margin(model,d,s)
  # Candidate swap must reverse the score.
  reverse=nn_margin(model,scale,-d[:32],s[:32]) if key=='nn' else margin(model,-d[:32],s[:32]);assert np.allclose(value[:32],-reverse)
  pred+=w*value
 primary=r['new_weight']*pred+(1-r['new_weight'])*ref;expected={'primary':primary,'blend50':.5*primary+.5*ref,'new_only':pred};audits={}
 for label,value in expected.items():
  path=a.out/label/'task_5_predictions.csv';frame=pd.read_csv(path);assert len(frame)==86786;pd.testing.assert_frame_equal(frame.iloc[:,:-1],test.iloc[:,:-1]);assert np.array_equal(frame.which,np.where(value>=0,test.item1,test.item2));audits[label]={'rows':len(frame),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'ids_order_preserved':True,'saved_model_replay':True}
 sample=np.random.RandomState(91).choice(len(ctx),64,replace=False);contexts=[ctx[x] for x in sample];N=len(ctx);squ=np.r_[qu[sample],qu[N+sample]];sqi=np.r_[qi[sample],qi[N+sample]];take=np.r_[sample,N+sample];model=dict(np.load(a.out/'full/attention_model.npz'));warm=dict(np.load(a.previous/'full/fpmc32_uniform_model.npz'));replay=scores(model,contexts,squ,sqi,warm['P'],warm['Q']);cache=np.load(a.out/'full/attention.npz')
 for key,value in replay.items():assert np.allclose(value,cache['attention_e20_'+key][take],atol=1e-6)
 theta=dict(np.load(a.out/'full/era_parameters.npz'));cache=np.load(a.out/'full/era.npz')
 for pen in [.1,1.,10.]:
  replay=era_scores(theta['theta_'+str(pen)],contexts,sqi)
  for key,value in replay.items():assert np.allclose(value,cache['era'+str(pen)+'_'+key][take],atol=1e-6)
 train,contexts,vu,vi,U,I,q=data(a.data_dir,a.baseline,False);assert len(contexts)==2*U
 for user in range(U):assert np.array_equal(contexts[user],train[user]) and len(contexts[U+user])==len(train[user])+1 and vi[user] not in train[user] and vi[U+user] not in contexts[U+user]
 r['audit']=audits;r['checks']={'direct_transition_enumeration':True,'exact_two_hop_path_check':True,'community_order_dense_optimum':True,'attention_finite_difference_gradient':True,'final_attention_parameter_replay':True,'community_order_parameter_replay':True,'chronological_user_targets_excluded':True,'candidate_swap_antisymmetry':True};r['input_sha256']={f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['task_5_train.csv','task_5_test.csv']};(a.out/'results.json').write_text(json.dumps(r,indent=2));print(json.dumps({'audit':audits,'checks':r['checks']},indent=2));return r

def fold_numerical():
 so=Path('/tmp/fold_check.so');subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(Path(__file__).with_name('task5_fold_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.fold_users.argtypes=[ctypes.c_int]*2+[ip]*6+[ctypes.c_int,ip]+[fp]*4+[ctypes.c_int]*3
 ptr=np.array([0,1],dtype='i');zero=np.array([0],dtype='i');active=np.array([1],dtype='i');known=np.array([1,1],dtype='i');P=np.array([[.1,-.2]],'f');Q=np.array([[.3,.2],[-.2,.1]],'f');bias=np.array([.1,-.1],'f');out=P.copy();expected=P[0].astype(float).copy();origin=expected.copy();delta=(Q[0]-Q[1]).astype(float)
 for step in range(2):
  def loss(x):return np.logaddexp(0,-(float(bias[0]-bias[1])+x@delta))+.05*np.sum((x-origin)**2)
  g=np.zeros(2)
  for j in range(2):a=expected.copy();b=expected.copy();a[j]+=1e-5;b[j]-=1e-5;g[j]=(loss(a)-loss(b))/2e-5
  expected-=.02*g
 lib.fold_users(1,2,ptr,zero,ptr,zero,zero,active,1,known,P,Q,bias,out,3,0,2);assert np.allclose(out[0],expected,atol=2e-7)
 print('PASS: query-user recency adaptation matches finite-difference BPR gradients with an anchoring penalty.',flush=True)

def audit_preference(a):
 import json,pickle,hashlib,pandas as pd
 from task5_preference_round3 import load,current
 from task5_round3 import nn_margin
 from task5_improve import data
 meta=a.out/'preference';d,s,parts,names=load(a,True);r=json.loads((meta/'results.json').read_text());ref=current(a,parts,'evaluation',slice(None));new=0
 for key,w in r['weights'].items():
  if w==0:continue
  model,scale=pickle.loads((meta/(key+'_evaluation.pkl')).read_bytes());value=nn_margin(model,scale,d,s);assert np.allclose(nn_margin(model,scale,-d[:32],s[:32]),-value[:32]);new+=w*value
 primary=r['new_weight']*new+(1-r['new_weight'])*ref;h,ctx,qu,qi,U,I,test=data(a.data_dir,a.baseline,True);pd.testing.assert_frame_equal(pd.read_csv(a.out/'primary/task_5_predictions.csv').iloc[:,:-1],test.iloc[:,:-1]);assert np.array_equal(pd.read_csv(a.out/'primary/task_5_predictions.csv').which,np.where(ref>=0,test.item1,test.item2));public=np.load(a.out/'test_predictions.npz')['reference'];audit={}
 for name,values in [('primary',primary),('blend50',.5*primary+.5*public)]:
  path=meta/name/'task_5_predictions.csv';frame=pd.read_csv(path);pd.testing.assert_frame_equal(frame.iloc[:,:-1],test.iloc[:,:-1]);assert len(frame)==86786 and np.array_equal(frame.which,np.where(values>=0,test.item1,test.item2));audit[name]={'rows':len(frame),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'ids_order_preserved':True,'saved_model_replay':True}
 for stage in ['validation','full']:
  sample=dict(np.load(a.out/stage/'bpr_replay_sample.npz'));cache=np.load(a.out/stage/'bpr.npz')
  for name,last in [('bpr128',350),('bpr96',300)]:
   pred=np.einsum('ij,ij->i',sample[name+'_users'],sample[name+'_items']);assert np.allclose(pred,sample[name+'_scores'],atol=1e-6);assert np.allclose(pred,cache[f'{name}_e{last}_personal'][sample[name+'_rows']],atol=1e-6)
  sample=dict(np.load(a.out/stage/'fold_replay_sample.npz'));cache=np.load(a.out/stage/'fold.npz')
  for cap in [0,3,10]:
   pred=np.einsum('ij,ij->i',sample[f'fold{cap}_user'],sample[f'fold{cap}_item']);assert np.allclose(pred,sample[f'fold{cap}_score'],atol=1e-6);assert np.allclose(pred,cache[f'fold{cap}_e100_score'][sample[f'fold{cap}_rows']],atol=1e-6)
 path=meta/'larger_refit/task_5_predictions.csv'
 if path.exists():
  model,scale=pickle.loads((meta/'base_budget_refit.pkl').read_bytes());ref_large=nn_margin(model,scale,parts[0],parts[1]);new_large=0
  for key,w in r['weights'].items():
   if w==0 or r['new_weight']==0:continue
   model,scale=pickle.loads((meta/(key+'_budget_refit.pkl')).read_bytes());new_large+=w*nn_margin(model,scale,d,s)
  pred=r['new_weight']*new_large+(1-r['new_weight'])*ref_large;frame=pd.read_csv(path);pd.testing.assert_frame_equal(frame.iloc[:,:-1],test.iloc[:,:-1]);assert np.array_equal(frame.which,np.where(pred>=0,test.item1,test.item2));audit['larger_refit']={'rows':len(frame),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'ids_order_preserved':True,'saved_model_replay':True,'independent_local_accuracy':None}
 r['audit']=audit;r['checks']={'BPR_factor_replay_samples':True,'query_fold_replay_samples':True,'query_fold_gradient':True,'pair_swap_antisymmetry':True,'final_ranker_replay':True};(meta/'results.json').write_text(json.dumps(r,indent=2));print(json.dumps({'preference_audit':audit,'checks':r['checks']},indent=2));return r

if __name__=='__main__':numerical();fold_numerical()
