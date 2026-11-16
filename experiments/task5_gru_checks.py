"""Independent numpy recurrence, cohort exclusion and final CSV/model replay audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,pickle,gc,hashlib
import numpy as np,pandas as pd
from task5_gru import OUT,W,DATA,K,library,stage_data,add,bankargs
from task5_preference_round3 import load
from task5_round3 import nn_margin
reports={};lib=library()
assert np.array_equal(np.load(OUT/'holdout.npz')['readers'],np.load(W/'task5_rolling/holdout.npz')['readers'])
for stage in ['selection','evaluation','full']:
 h,contexts,qu,qi,U,I,q=stage_data(stage);folder=OUT/stage;z=np.load(folder/'model.npz');E,Q,b,Wm,gb=[z[n] for n in ['E','Q','b','W','gb']];pop=z['pop'];source=z['source_count'];assert np.array_equal(pop,np.bincount(np.concatenate(h),minlength=I));assert (E[source==0]==0).all() and (Q[pop==0]==0).all() and (b[pop==0]==0).all()
 if stage!='full':
  N=len(qi)//2;assert all(target not in hh for target,hh in zip(qi[:N],contexts))
 replay=np.load(folder/'replay.npz');errors=[]
 for v,(lo,hi) in enumerate(zip(replay['offset'][:-1],replay['offset'][1:])):
  state=np.zeros(K,'f')
  for x in replay['history_E'][lo:hi]:
   update=1/(1+np.exp(-(Wm[0]@x+Wm[1]@state+gb[0])));reset=1/(1+np.exp(-(Wm[2]@x+Wm[3]@state+gb[1])));proposal=np.tanh(Wm[4]@x+Wm[5]@(reset*state)+gb[2]);state=(1-update)*state+update*proposal
  errors.append(float(abs(state-replay['encoded'][v]).max()))
 assert max(errors)<2e-5,(stage,max(errors))
 ptr=np.r_[0,np.cumsum(list(map(len,h)))].astype('i');seq=np.concatenate(h).astype('i');states=np.zeros((U,K),'f');lib.gru_encode(U,K,ptr,seq,E,Wm,gb,states);valid=qi>=0;valid[valid]&=pop[qi[valid]]>0;dot=np.zeros(len(qi),'f');bias=np.zeros_like(dot);dot[valid]=np.einsum('ij,ij->i',states[qu[valid]],Q[qi[valid]]);bias[valid]=b[qi[valid]];expected=np.column_stack([dot,bias,dot+bias]);actual=np.load(folder/'scores.npz')['scores'];assert np.allclose(actual,expected,rtol=0,atol=1e-6);assert (actual[~valid]==0).all();reports[stage]={'fitting_events':len(seq),'own_target_exclusion':True if stage!='full' else None,'independent_numpy_encoder_max_error':max(errors),'candidate_score_replay':True,'cold_and_unknown_score_zero':True};del z,E,Q,b,Wm,gb,actual,expected,states;gc.collect()
d,s,parts,extra=load(bankargs(),True);features=np.load(OUT/'full/scores.npz')['scores'];pred=0;rows=np.random.RandomState(73).choice(len(d),512,False)
for key,width in [('sequence',224),('preference',260)]:
 m,scale=pickle.loads((OUT/(key+'_production.pkl')).read_bytes());dd,ss=add(d[:,:width],s[:,:width],features);assert m.n_features_in_==2*(width+3);margin=nn_margin(m,scale,dd,ss);assert np.isfinite(margin).all();assert np.allclose(margin[rows],-nn_margin(m,scale,-dd[rows],ss[rows]),atol=1e-6);pred+=.5*margin
assert np.allclose(pred,np.load(OUT/'test.npz')['margin'],atol=1e-12,rtol=0)
q=pd.read_csv(DATA/'task_5_test.csv');path=W.parent/'output/task5_gru_neural/task_5_predictions.csv';f=pd.read_csv(path);assert len(f)==86786;assert f.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert np.array_equal(f.which,np.where(pred>=0,q.item1,q.item2));assert ((f.which==f.item1)|(f.which==f.item2)).all();reports['output']={'rows':len(f),'row_order_and_ids_unchanged':True,'choices_valid':True,'saved_model_margin_replay':True,'candidate_swap_antisymmetry':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};(OUT/'checks.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
