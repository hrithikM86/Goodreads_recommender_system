"""Order/attention augmented Task 5, inner-only selection and final-event validation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,pickle,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from task5_next_stack import matrices as previous_matrices
from task5_improve import data

def design(d,s):return np.c_[d,s].astype('f')
def accuracy(x):return float(np.mean((x>0)+.5*(x==0)))
def margin(model,d,s):return model.predict_proba(design(d,s),thread_count=2)[:,1]-model.predict_proba(design(-d,s),thread_count=2)[:,1]
def save(model,path):
 temp=path.with_suffix('.tmp.cbm');model.save_model(str(temp));temp.replace(path)
def cfg(depth,iters=1100):return dict(iterations=iters,depth=depth,learning_rate=.04,l2_leaf_reg=20,loss_function='Logloss',eval_metric='Accuracy',random_seed=91,thread_count=2,verbose=200,allow_writing_files=False)
def load(a,full=False):
 arg=SimpleNamespace(data_dir=a.data_dir,baseline=a.baseline,previous=a.previous,out=a.reference);old,olds,earlier,earliers,names=previous_matrices(arg,full);cols=[];extra=[]
 for kind in ['order','era','attention']:
  z=dict(np.load(a.out/('full' if full else 'validation')/(kind+'.npz')));cols.extend(z.values());extra+=list(z)
 scores=np.column_stack(cols);N=len(scores)//2;d=np.c_[old,scores[:N]-scores[N:]].astype('f');s=np.c_[olds,scores[:N]+scores[N:]].astype('f');assert np.isfinite(d).all() and np.isfinite(s).all();return d,s,old,olds,earlier,earliers,names+extra

def reference(a,d,s,ed,es,stage):
 r=json.loads((a.reference/'results.json').read_text());pred=0
 for key,w in r['depth_weights'].items():
  if w==0:continue
  m=CatBoostClassifier();m.load_model(str(a.reference/(key+'_'+stage+'.cbm')));pred+=r['new_weight']*w*margin(m,d,s)
 if r['new_weight']<1:
  prev=json.loads((a.previous/'results.json').read_text())['selected']['weights']
  for key,w in prev.items():
   m=CatBoostClassifier();m.load_model(str(a.previous/(key+'_'+('evaluation' if stage=='full' else stage)+'.cbm')));pred+=(1-r['new_weight'])*w*margin(m,ed,es)
 return pred

def nn_margin(model,scale,d,s):return model.predict_proba(scale.transform(design(d,s)))[:,1]-model.predict_proba(scale.transform(design(-d,s)))[:,1]
def nn_fit(d,s,epochs,validation=None,path=None,seed=71,hidden=(64,32),alpha=.1):
 X=np.r_[design(d,s),design(-d,s)];y=np.r_[np.ones(len(d)),np.zeros(len(d))];scale=StandardScaler().fit(X);X=scale.transform(X);model=MLPClassifier(hidden_layer_sizes=hidden,alpha=alpha,learning_rate_init=.001,batch_size=512,random_state=seed);best=-1;best_epoch=0;report=None
 for epoch in range(1,epochs+1):
  model.partial_fit(X,y,classes=[0,1])
  if validation is not None:
   vd,vs,ref=validation;value=nn_margin(model,scale,vd,vs);score,w=max(((accuracy(w*value+(1-w)*ref),w) for w in [0,.25,.5,.75,1]),key=lambda x:x[0])
   if score>best:best=score;best_epoch=epoch;report={'epochs':epoch,'inner_accuracy':accuracy(value),'inner_blend_accuracy':score,'new_weight':w};temp=path.with_suffix('.tmp.pkl');temp.write_bytes(pickle.dumps((model,scale),protocol=5));temp.replace(path)
   if epoch%5==0:print('NN_RANKER',epoch,'accuracy',accuracy(value),'best_blend',best,flush=True)
   if epoch-best_epoch>=8:break
  elif epoch%10==0:print('NN_REFIT',epoch,'/',epochs,flush=True)
 return (model,scale),report

def main():
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','previous','reference','out','accepted']:ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--predict',action='store_true');a=ap.parse_args();d,s,old,olds,ed,es,names=load(a);U=len(d)//2;idx=np.random.RandomState(17).permutation(U);tr=idx[:int(.8*U)];va=idx[int(.8*U):];refinner=reference(a,old[va],olds[va],ed[va],es[va],'inner');refeval=reference(a,old[U:],olds[U:],ed[U:],es[U:],'evaluation');assert accuracy(refeval)==accuracy(np.load(a.reference/'evaluation.npz')['primary']);reports={};inner={}
 for depth in [5,7]:
  key=f'depth{depth}';path=a.out/(key+'_inner.cbm');m=CatBoostClassifier(**cfg(depth))
  if path.exists():m.load_model(str(path))
  else:
   m.fit(np.r_[design(d[tr],s[tr]),design(-d[tr],s[tr])],np.r_[np.ones(len(tr)),np.zeros(len(tr))],eval_set=(np.r_[design(d[va],s[va]),design(-d[va],s[va])],np.r_[np.ones(len(va)),np.zeros(len(va))]),early_stopping_rounds=100)
   count=max([(accuracy(m.predict_proba(design(d[va],s[va]),ntree_end=c,thread_count=2)[:,1]-m.predict_proba(design(-d[va],s[va]),ntree_end=c,thread_count=2)[:,1]),c) for c in sorted(set(list(range(50,m.tree_count_+1,50))+[m.tree_count_]))],key=lambda x:(x[0],-x[1]))[1];m.shrink(count);save(m,path)
  inner[key]=margin(m,d[va],s[va]);reports[key]={'depth':depth,'iterations':m.tree_count_,'inner_accuracy':accuracy(inner[key])};print('INNER',key,reports[key],flush=True)
 path=a.out/'nn_inner.pkl';info=a.out/'nn_info.json'
 if info.exists():r=json.loads(info.read_text())
 else:_,r=nn_fit(d[tr],s[tr],40,(d[va],s[va],refinner),path);info.write_text(json.dumps(r,indent=2))
 model,scale=pickle.loads(path.read_bytes());inner['nn']=nn_margin(model,scale,d[va],s[va]);reports['nn']=r;choices=[]
 for depth5 in [0,.25,.5,.75,1]:
  cat=depth5*inner['depth5']+(1-depth5)*inner['depth7']
  for nw in [0,.25,.5,.75,1]:
   new=(1-nw)*cat+nw*inner['nn']
   for weight in [0,.25,.5,.75,1]:choices.append((accuracy(weight*new+(1-weight)*refinner),depth5,nw,weight))
 score,depth5,nw,weight=max(choices,key=lambda x:x[0]);weights={'depth5':(1-nw)*depth5,'depth7':(1-nw)*(1-depth5),'nn':nw};evals={};models={}
 for key,w in weights.items():
  if w==0:continue
  if key=='nn':
   path=a.out/'nn_evaluation.pkl'
   if path.exists():model,scale=pickle.loads(path.read_bytes())
   else:(model,scale),_=nn_fit(d[:U],s[:U],r['epochs']);path.write_bytes(pickle.dumps((model,scale),protocol=5))
   evals[key]=nn_margin(model,scale,d[U:],s[U:]);models[key]=(model,scale)
  else:
   cfgd=reports[key];path=a.out/(key+'_evaluation.cbm');m=CatBoostClassifier(**cfg(cfgd['depth'],cfgd['iterations']))
   if path.exists():m.load_model(str(path))
   else:m.fit(np.r_[design(d[:U],s[:U]),design(-d[:U],s[:U])],np.r_[np.ones(U),np.zeros(U)]);save(m,path)
   evals[key]=margin(m,d[U:],s[U:]);models[key]=m
 new=sum(weights[k]*v for k,v in evals.items());primary=weight*new+(1-weight)*refeval;summary={'weights':weights,'new_weight':weight,'inner_accuracy':score,'reference_inner_accuracy':accuracy(refinner),'runs':reports,'reference_evaluation_accuracy':accuracy(refeval),'new_evaluation_accuracy':accuracy(new),'primary_evaluation_accuracy':accuracy(primary),'blend50_evaluation_accuracy':accuracy(.5*primary+.5*refeval),'reference_public_accuracy':.9136957573802226,'features':names,'zero_rating_events_retained':True,'numeric_timestamps_used':False};(a.out/'results.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(a.out/'evaluation.npz',reference=refeval,new=new,primary=primary,**evals);print('ROUND3_RESULT',{k:v for k,v in summary.items() if k!='features'},flush=True)
 if not a.predict:return
 fd,fs,fo,fos,fe,fes,fn=load(a,True);assert fn==names;h,ctx,qu,qi,U,I,test=data(a.data_dir,a.baseline,True);ref=reference(a,fo,fos,fe,fes,'evaluation');oldcsv=__import__('pandas').read_csv(a.accepted);assert np.array_equal(oldcsv.which,np.where(ref>=0,test.item1,test.item2));new=0
 for key,model in models.items():new+=weights[key]*(nn_margin(*model,fd,fs) if key=='nn' else margin(model,fd,fs))
 primary=weight*new+(1-weight)*ref
 for label,pred in [('primary',primary),('blend50',.5*primary+.5*ref),('new_only',new)]:
  folder=a.out/label;folder.mkdir(exist_ok=True);f=test.copy();f['which']=np.where(pred>=0,f.item1,f.item2);f.to_csv(folder/'task_5_predictions.csv',index=False)
 np.savez_compressed(a.out/'test_predictions.npz',primary=primary,new=new,reference=ref)
if __name__=='__main__':main()
