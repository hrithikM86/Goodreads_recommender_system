"""Compact GRU4Rec-style sequential BPR model and controlled neural-stack addition.
Custom native CPU implementation; numeric finite-difference gradients checked first.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,json,pickle,hashlib,gc,time,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from task5_rolling import W,DATA,args as bankargs,cohort,refreshed,SEEDS
from task5_improve import data,graph
from task5_preference_round3 import load
from task5_round3 import nn_fit,nn_margin,accuracy
OUT=W/'task5_gru';OUT.mkdir(exist_ok=True);K=24;EPOCHS=6;NEG=4;TRUNC=20;SEED=2027
NAMES=['gru_dot','gru_bias','gru_joint']
def library():
 lib=ctypes.CDLL(str(W/'task5_gru.so'));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.gru_grad.argtypes=[ctypes.c_int]*3+[ip]*2+[fp]*13;lib.gru_grad.restype=ctypes.c_double
 lib.gru_train.argtypes=[ctypes.c_int]*8+[ip]*5+[ctypes.c_int]+[fp]*10+[ctypes.c_float];lib.gru_train.restype=ctypes.c_double
 lib.gru_encode.argtypes=[ctypes.c_int]*2+[ip]*2+[fp]*4;return lib

def init(I,k=K):
 rng=np.random.RandomState(SEED);E=rng.normal(0,.05,(I,k)).astype('f');Q=rng.normal(0,.05,(I,k)).astype('f');b=np.zeros(I,'f');Wm=np.empty((6,k,k),'f')
 for g in range(3):
  Wm[2*g]=rng.normal(0,.2,(k,k)).astype('f');Wm[2*g+1]=np.linalg.qr(rng.normal(size=(k,k)))[0].astype('f')*.5
 gb=np.zeros((3,k),'f');gb[0]=-1;return E,Q,b,Wm,gb

def gradients():
 lib=library();k=3;I=8;T=4;neg=2;rng=np.random.RandomState(1);E,Q,b,Wm,gb=init(I,k);E*=3;Q*=4;b[:]=rng.normal(0,.1,I);h0=rng.normal(0,.2,k).astype('f');xi=np.array([0,1,2,3],'i');c=np.array([[1,5,6],[2,6,7],[3,4,5],[4,5,7]],'i');gx=np.zeros((T,k),'f');gq=np.zeros((T,neg+1,k),'f');gbi=np.zeros((T,neg+1),'f');gw=np.zeros_like(Wm);ggb=np.zeros_like(gb);gh=np.zeros(k,'f');last=np.zeros(k,'f')
 def calc():return lib.gru_grad(k,T,neg,xi,c,h0,E,Q,b,Wm,gb,gx,gq,gbi,gw,ggb,gh,last)
 loss=calc();anal=[np.zeros_like(E),np.zeros_like(Q),np.zeros_like(b),gw.copy(),ggb.copy(),gh.copy()]
 for t in range(T):
  anal[0][xi[t]]+=gx[t]
  for j in range(neg+1):anal[1][c[t,j]]+=gq[t,j];anal[2][c[t,j]]+=gbi[t,j]
 checks=[]
 for name,a,g in zip(['input_embeddings','output_embeddings','item_bias','GRU_matrices','GRU_biases','initial_hidden'],[E,Q,b,Wm,gb,h0],anal):
  flat=a.ravel();gf=g.ravel();indices=np.arange(len(flat)) if len(flat)<=40 else np.random.RandomState(77).choice(len(flat),40,False);diff=[];relative=[]
  for j in indices:
   old=float(flat[j]);eps=.002;flat[j]=old+eps;plus=calc();flat[j]=old-eps;minus=calc();flat[j]=old;numeric=(plus-minus)/(2*eps);error=abs(numeric-float(gf[j]));diff.append(error);relative.append(error/max(1e-4,abs(numeric)+abs(float(gf[j]))))
  assert max(diff)<3e-5,(name,max(diff));checks.append({'parameter':name,'entries_checked':len(indices),'max_absolute_error':max(diff),'max_relative_error_floor_1e4':max(relative)})
 # Independent numpy forward verifies gate layout and exact loss/encoding convention.
 h=h0.copy();hidden=[];numpy_loss=0
 for t in range(T):
  x=E[xi[t]];z=1/(1+np.exp(-(Wm[0]@x+Wm[1]@h+gb[0])));r=1/(1+np.exp(-(Wm[2]@x+Wm[3]@h+gb[1])));n=np.tanh(Wm[4]@x+Wm[5]@(r*h)+gb[2]);h=(1-z)*h+z*n;hidden.append(h.copy());ss=Q[c[t]]@h+b[c[t]];numpy_loss+=np.logaddexp(0,-(ss[0]-ss[1:])).sum()/(T*neg)
 calc();assert np.allclose(h,last,atol=1e-7);assert abs(numpy_loss-loss)<1e-7
 # Verify causal encoder separately, beginning from zero, against numpy.
 p=np.array([0,T],'i');encoded=np.zeros((1,k),'f');lib.gru_encode(1,k,p,xi,E,Wm,gb,encoded);h=np.zeros(k,'f')
 for j in xi:
  x=E[j];z=1/(1+np.exp(-(Wm[0]@x+Wm[1]@h+gb[0])));r=1/(1+np.exp(-(Wm[2]@x+Wm[3]@h+gb[1])));n=np.tanh(Wm[4]@x+Wm[5]@(r*h)+gb[2]);h=(1-z)*h+z*n
 assert np.allclose(h,encoded[0],atol=1e-7)
 report={'gradient_checks':checks,'numpy_forward_loss_and_hidden_replay':True,'causal_encoder_replay':True,'loss':loss};(OUT/'gradient_checks.json').write_text(json.dumps(report,indent=2));print('GRADIENT_CHECKS',json.dumps(report),flush=True)

def stage_data(stage):
 if stage=='evaluation':return cohort()
 h,ctx,qu,qi,U,I,q=data(DATA,W/'task5/results',stage=='full')
 if stage=='selection':qu=np.tile(np.arange(U,dtype='i'),2);qi=np.r_[qi[:U],qi[2*U:3*U]];ctx=h
 return h,ctx,qu.astype('i'),qi.astype('i'),U,I,q

def train(stage):
 assert (OUT/'gradient_checks.json').exists(),'Run gradient checks first'
 folder=OUT/stage;folder.mkdir(exist_ok=True);path=folder/'scores.npz'
 if path.exists():return
 lib=library();hist,ctx,qu,qi,U,I,q=stage_data(stage);X=graph(hist,U,I);seq=np.concatenate(hist).astype('i');ptr=np.r_[0,np.cumsum(list(map(len,hist)))].astype('i');pop=np.bincount(seq,minlength=I);sources=np.concatenate([h[:-1] for h in hist]);source_count=np.bincount(sources,minlength=I);active=np.flatnonzero(pop);prob=pop[active]**.75;prob/=prob.sum();pool=np.random.RandomState(SEED+1).choice(active,size=524288,p=prob).astype('i');model=folder/'model.npz';started=time.time()
 if model.exists():
  z=np.load(model);E,Q,b,Wm,gb=[z[n] for n in ['E','Q','b','W','gb']]
 else:
  E,Q,b,Wm,gb=init(I);states=[np.full_like(v,1e-6) for v in [E,Q,b,Wm,gb]];loss=lib.gru_train(U,I,K,EPOCHS,0,SEED,TRUNC,NEG,ptr,seq,X.indptr.astype('i'),X.indices.astype('i'),pool,len(pool),E,Q,b,Wm,gb,*states,.03);assert all(np.isfinite(v).all() for v in [E,Q,b,Wm,gb]);E[source_count==0]=0;Q[pop==0]=0;b[pop==0]=0;np.savez_compressed(model,E=E,Q=Q,b=b,W=Wm,gb=gb,pop=pop,source_count=source_count);del states;gc.collect()
 encoded=np.zeros((U,K),'f');lib.gru_encode(U,K,ptr,seq,E,Wm,gb,encoded);valid=qi>=0;valid[valid]&=pop[qi[valid]]>0;dot=np.zeros(len(qi),'f');bias=np.zeros_like(dot);dot[valid]=np.einsum('ij,ij->i',encoded[qu[valid]],Q[qi[valid]]);bias[valid]=b[qi[valid]];scores=np.column_stack([dot,bias,dot+bias]).astype('f');assert np.isfinite(scores).all();np.savez_compressed(path,scores=scores)
 # Independent encoder replay samples and representative trained factors for output audit.
 rows=np.random.RandomState(37).choice(U,32,False);tokens=np.concatenate([hist[u] for u in rows]);lens=np.array([len(hist[u]) for u in rows]);np.savez_compressed(folder/'replay.npz',users=rows,offset=np.r_[0,np.cumsum(lens)],history_E=E[tokens],encoded=encoded[rows]);r={'stage':stage,'factors':K,'epochs':EPOCHS,'negative_samples':NEG,'negative_sampling':'frequency^.75, reject all observed fitting-history books','BPTT_truncation':TRUNC,'learning_rate':.03,'optimizer':'Adagrad, accumulator initial1e-6, gradient norm clip5, recurrent/input matrices decay1e-5','loss':'mean pairwise BPR softplus, four sampled negatives per next-book target','seed':SEED,'fitting_events':len(seq),'next_book_training_steps_per_epoch':len(seq)-U,'seconds':time.time()-started,'cold_input_rows_zeroed':int((source_count==0).sum()),'query_scores':list(scores.shape),'rating_values_used':False,'timestamps_used':False};(folder/'report.json').write_text(json.dumps(r,indent=2));print('GRU_STAGE',json.dumps(r),flush=True)

def add(d,s,f):
 n=len(f)//2;assert n==len(d);return np.c_[d,f[:n]-f[n:]].astype('f'),np.c_[s,f[:n]+f[n:]].astype('f')
def combine():
 for stage in ['selection','evaluation','full']:assert (OUT/stage/'scores.npz').exists()
 d,s,parts,extra=load(bankargs());U=len(d)//2;sd,ss=d[:U].copy(),s[:U].copy();del d,s,parts;gc.collect();sf=np.load(OUT/'selection/scores.npz')['scores'];ed,es,names=refreshed();ef=np.load(OUT/'evaluation/scores.npz')['scores'];split=np.random.RandomState(17).permutation(U);tr=split[:int(.8*U)];va=split[int(.8*U):];draw=np.random.RandomState(418).randint(3,size=U)*U+np.arange(U);evrows=np.r_[va,va+U,va+2*U];models={};holdout=0
 # Fixed architecture/settings; only append the three direct-GRU scores.
 for key,width,seed,epochs in [('sequence',224,71,16),('preference',260,73,8)]:
  ad,ass=add(sd[:,:width],ss[:,:width],sf);aed,aes=add(ed[:,:width],es[:,:width],ef);td=np.r_[ad,aed[draw[tr]]];ts=np.r_[ass,aes[draw[tr]]];dest=OUT/(key+'_holdout.pkl')
  if dest.exists():m=pickle.loads(dest.read_bytes())
  else:m,_=nn_fit(td,ts,epochs,seed=seed,hidden=(64,32),alpha=.1);dest.write_bytes(pickle.dumps(m,protocol=5))
  holdout+=.5*nn_margin(*m,aed[evrows],aes[evrows]);del td,ts;gc.collect();td=np.r_[ad,aed[draw]];ts=np.r_[ass,aes[draw]];dest=OUT/(key+'_production.pkl')
  if dest.exists():m=pickle.loads(dest.read_bytes())
  else:m,_=nn_fit(td,ts,epochs,seed=seed,hidden=(64,32),alpha=.1);dest.write_bytes(pickle.dumps(m,protocol=5))
  models[key]=m;del ad,ass,aed,aes,td,ts;gc.collect();print('GRU_STACK',key,flush=True)
 baseline=np.load(W/'task5_rolling/holdout.npz')['refit'];assert baseline.shape==holdout.shape;size=len(va);corr=lambda x:(x>0).astype(float)+.5*(x==0);delta=(corr(holdout)-corr(baseline)).reshape(3,size).mean(0);se=delta.std(ddof=1)/np.sqrt(size);r={'baseline_public_accuracy':.9221302975134238,'baseline_holdout_by_seed':{str(seed):accuracy(baseline[j*size:(j+1)*size]) for j,seed in enumerate(SEEDS)},'GRU_augmented_holdout_by_seed':{str(seed):accuracy(holdout[j*size:(j+1)*size]) for j,seed in enumerate(SEEDS)},'mean_paired_holdout_change':float(delta.mean()),'approx_reader_clustered95pct_delta_interval':[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)],'heldout_readers':size,'final_label_reader_overlap':int(len(np.intersect1d(tr,va))),'new_features':NAMES,'production_pairs':2*U,'settings':'Accepted two64/32 MLPs, alpha.1, seeds71/73,epochs16/8,equal average; append3 scores from one24-dimensional GRU model','validation_limit':'Historical diagnostic reader split reused; three draws share positives/readers. No hidden labels. Production fits all known historical meta-targets; no unbiased score for final full-data weights.'};np.savez_compressed(OUT/'holdout.npz',readers=va,accepted=baseline,gru=holdout);print('GRU_HOLDOUT',json.dumps(r),flush=True)
 del sd,ss,ed,es;gc.collect();fd,fs,parts,extra=load(bankargs(),True);ff=np.load(OUT/'full/scores.npz')['scores'];pred=0
 for key,width in [('sequence',224),('preference',260)]:dd,sss=add(fd[:,:width],fs[:,:width],ff);pred+=.5*nn_margin(*models[key],dd,sss)
 assert np.isfinite(pred).all();q=pd.read_csv(DATA/'task_5_test.csv');f=q.copy();f['which']=np.where(pred>=0,f.item1,f.item2);dest=Path('/workspace/output/task5_gru_neural');dest.mkdir(exist_ok=True);path=dest/'task_5_predictions.csv';f.to_csv(path,index=False);read=pd.read_csv(path);assert len(read)==U and read.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert ((read.which==read.item1)|(read.which==read.item2)).all();old=pd.read_csv('/workspace/output/task5_rolling_neural/task_5_predictions.csv');np.savez_compressed(OUT/'test.npz',margin=pred);r.update(output=str(path),rows=U,changed_vs_accepted=int((read.which!=old.which).sum()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_sha256={n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['task_5_train.csv','task_5_test.csv']});(OUT/'results.json').write_text(json.dumps(r,indent=2));print('EXPORTED',r['output'],r['changed_vs_accepted'],flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['gradients','selection','evaluation','full','combine'],required=True);a=ap.parse_args();gradients() if a.stage=='gradients' else combine() if a.stage=='combine' else train(a.stage)
