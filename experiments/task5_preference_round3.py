"""Augment sequence/order rankers with stronger global preference models."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,pickle
from pathlib import Path
import numpy as np,pandas as pd
from task5_round3 import load as core_load,nn_fit,nn_margin,margin,reference,accuracy
from task5_improve import data

def load(a,full=False):
 parts=core_load(a,full);d,s=parts[:2];cols=[];names=[]
 for kind in ['bpr','fold']:
  z=np.load(a.out/('full' if full else 'validation')/(kind+'.npz'));cols.extend([z[k] for k in z.files]);names+=list(z.files)
 scores=np.column_stack(cols);N=len(scores)//2;return np.c_[d,scores[:N]-scores[N:]].astype('f'),np.c_[s,scores[:N]+scores[N:]].astype('f'),parts,names
def current(a,parts,stage,index):
 d,s,od,oscore,ed,es,names=parts;r=json.loads((a.out/'results.json').read_text());pred=0
 for key,w in r['weights'].items():
  if w==0:continue
  if key=='nn':m,scale=pickle.loads((a.out/('nn_'+stage+'.pkl')).read_bytes());value=nn_margin(m,scale,d[index],s[index])
  else:
   from catboost import CatBoostClassifier
   m=CatBoostClassifier();m.load_model(str(a.out/(key+'_'+stage+'.cbm')));value=margin(m,d[index],s[index])
  pred+=r['new_weight']*w*value
 if r['new_weight']<1:pred+=(1-r['new_weight'])*reference(a,od[index],oscore[index],ed[index],es[index],stage)
 return pred

def main():
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','previous','reference','out','accepted']:ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--predict',action='store_true');a=ap.parse_args();meta=a.out/'preference';meta.mkdir(exist_ok=True);d,s,parts,names=load(a);U=len(d)//2;idx=np.random.RandomState(17).permutation(U);tr=idx[:int(.8*U)];va=idx[int(.8*U):];ri=current(a,parts,'inner',va);re=current(a,parts,'evaluation',slice(U,None));configs={'small':{'seed':73,'hidden':[64,32],'alpha':.1},'wide':{'seed':117,'hidden':[128,64],'alpha':1.}};inner={};reports={}
 for key,cfg in configs.items():
  path=meta/(key+'_inner.pkl');info=meta/(key+'_info.json')
  if info.exists():r=json.loads(info.read_text())
  else:_,r=nn_fit(d[tr],s[tr],40,(d[va],s[va],ri),path,**cfg);r.update(cfg);info.write_text(json.dumps(r,indent=2))
  m,scale=pickle.loads(path.read_bytes());inner[key]=nn_margin(m,scale,d[va],s[va]);reports[key]=r;print('PREFERENCE_INNER',key,r,flush=True)
 options=[]
 for small in [0,.25,.5,.75,1]:
  pred=small*inner['small']+(1-small)*inner['wide']
  for w in [0,.25,.5,.75,1]:options.append((accuracy(w*pred+(1-w)*ri),small,w))
 score,small,w=max(options,key=lambda x:x[0]);weights={'small':small,'wide':1-small};pred=0;models={}
 for key,weight in weights.items():
  if weight==0:continue
  r=reports[key];path=meta/(key+'_evaluation.pkl')
  if path.exists():model=pickle.loads(path.read_bytes())
  else:model,_=nn_fit(d[:U],s[:U],r['epochs'],seed=r['seed'],hidden=r['hidden'],alpha=r['alpha']);path.write_bytes(pickle.dumps(model,protocol=5))
  models[key]=model;pred+=weight*nn_margin(*model,d[U:],s[U:])
 primary=w*pred+(1-w)*re;summary={'weights':weights,'new_weight':w,'runs':reports,'inner_accuracy':score,'reference_inner_accuracy':accuracy(ri),'reference_evaluation_accuracy':accuracy(re),'new_evaluation_accuracy':accuracy(pred),'primary_evaluation_accuracy':accuracy(primary),'comparison_to_public_reference':.9136957573802226,'preference_features':names};(meta/'results.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(meta/'evaluation.npz',primary=primary,new=pred,reference=re);print('PREFERENCE_RESULT',summary,flush=True)
 if not a.predict:return
 fd,fs,fp,fn=load(a,True);assert fn==names;ref=current(a,fp,'evaluation',slice(None));h,ctx,qu,qi,U,I,test=data(a.data_dir,a.baseline,True);assert np.array_equal(pd.read_csv(a.out/'primary/task_5_predictions.csv').which,np.where(ref>=0,test.item1,test.item2));new=sum(weights[k]*nn_margin(*m,fd,fs) for k,m in models.items());primary=w*new+(1-w)*ref;public=np.load(a.out/'test_predictions.npz')['reference']
 for label,values in [('primary',primary),('blend50',.5*primary+.5*public)]:
  folder=meta/label;folder.mkdir(exist_ok=True);f=test.copy();f['which']=np.where(values>=0,f.item1,f.item2);f.to_csv(folder/'task_5_predictions.csv',index=False)
 np.savez_compressed(meta/'test_predictions.npz',primary=primary,new=new,reference=ref)
if __name__=='__main__':main()
