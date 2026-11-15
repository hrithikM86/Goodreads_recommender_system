"""Metric-augmented antisymmetric sequential ensemble with inner user holdout."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
from task5_improve_stack import load,cfg,acc
from task5_improve import data
from task4_stacking import margin,design

def matrices(a,full=False):
 # load expects previous experiment features under .out
 from types import SimpleNamespace
 d,s,od,oscore,names=load(SimpleNamespace(baseline=a.baseline,out=a.previous),full);part=a.out/('full' if full else 'validation');cols=[];extra=[]
 for kind in ['metrics','ease','fossil']:
  z=dict(np.load(part/(kind+'.npz')))
  if kind=='metrics':
   h,ctx,qu,qi,U,I,q=data(a.data_dir,a.baseline,full);pop=np.load(part/'prme_model.npz')['pop'];cold=np.tile(np.array([pop[x[-1]]==0 for x in ctx]),2)
   for name in z:
    if name.endswith('_last'):z[name][cold]=0
    if name.endswith('_joint'):z[name][cold]=z[name.replace('_joint','_bias')][cold]
  cols.extend(list(z.values()));extra+=list(z)
 scores=np.column_stack(cols);N=len(scores)//2;return np.c_[d,scores[:N]-scores[N:]],np.c_[s,scores[:N]+scores[N:]],d,s,names+extra

def save(model,path):
 temp=path.with_suffix('.tmp.cbm');model.save_model(str(temp));os.replace(temp,path)

def main():
 ap=argparse.ArgumentParser()
 for key in ['data-dir','baseline','previous','out','accepted']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--predict',action='store_true');a=ap.parse_args();d,s,old,olds,names=matrices(a);U=len(d)//2;idx=np.random.RandomState(17).permutation(U);tr=idx[:int(.8*U)];va=idx[int(.8*U):];prev=json.loads((a.previous/'results.json').read_text())['selected']['weights'];refinner=0;refeval=0
 for key,w in prev.items():
  M=CatBoostClassifier();M.load_model(str(a.previous/(key+'_inner.cbm')));refinner+=w*margin(M,old[va],olds[va]);M.load_model(str(a.previous/(key+'_evaluation.cbm')));refeval+=w*margin(M,old[U:],olds[U:])
 inner={};reports={}
 for depth in [5,7]:
  key=f'depth{depth}';M=CatBoostClassifier(**cfg(depth,1400));path=a.out/(key+'_inner.cbm')
  if path.exists():M.load_model(str(path))
  else:
   M.fit(np.r_[design(d[tr],s[tr]),design(-d[tr],s[tr])],np.r_[np.ones(len(tr)),np.zeros(len(tr))],eval_set=(np.r_[design(d[va],s[va]),design(-d[va],s[va])],np.r_[np.ones(len(va)),np.zeros(len(va))]),early_stopping_rounds=125)
   choices=[(acc(M.predict_proba(design(d[va],s[va]),ntree_end=count)[:,1]-M.predict_proba(design(-d[va],s[va]),ntree_end=count)[:,1]),count) for count in sorted(set(list(range(50,M.tree_count_+1,50))+[M.tree_count_]))];score,count=max(choices,key=lambda r:(r[0],-r[1]));M.shrink(count);save(M,path)
  inner[key]=margin(M,d[va],s[va]);reports[key]={'depth':depth,'iterations':M.tree_count_,'inner_accuracy':acc(inner[key])};print('INNER',reports[key],flush=True)
 choices=[]
 for v in [0,.25,.5,.75,1]:
  pp=v*inner['depth5']+(1-v)*inner['depth7']
  for w in [0,.25,.5,.75,1]:choices.append((acc(w*pp+(1-w)*refinner),v,w))
 score,v,w=max(choices,key=lambda t:t[0]);weights={'depth5':v,'depth7':1-v};raw=0;models={}
 for key,mix in weights.items():
  if mix==0:continue
  r=reports[key];M=CatBoostClassifier(**cfg(r['depth'],r['iterations']));path=a.out/(key+'_evaluation.cbm')
  if path.exists():M.load_model(str(path))
  else:M.fit(np.r_[design(d[:U],s[:U]),design(-d[:U],s[:U])],np.r_[np.ones(U),np.zeros(U)]);save(M,path)
  raw+=mix*margin(M,d[U:],s[U:]);models[key]=M
 primary=w*raw+(1-w)*refeval;summary={'depth_weights':weights,'new_weight':w,'inner_accuracy':score,'reference_inner_accuracy':acc(refinner),'runs':reports,'reference_evaluation_accuracy':acc(refeval),'new_evaluation_accuracy':acc(raw),'primary_evaluation_accuracy':acc(primary),'features':names,'reference_public_accuracy':.9107920632360058};(a.out/'results.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(a.out/'evaluation.npz',reference=refeval,new=raw,primary=primary);print('RESULT',summary,flush=True)
 if not a.predict:return
 fd,fs,fo,fsold,fn=matrices(a,True);assert fn==names;h,ctx,qu,qi,_,_,test=data(a.data_dir,a.baseline,True);reference=0
 for key,ww in prev.items():
  M=CatBoostClassifier();M.load_model(str(a.previous/(key+'_full.cbm')));reference+=ww*margin(M,fo,fsold)
 raw=sum(weights[k]*margin(M,fd,fs) for k,M in models.items());primary=w*raw+(1-w)*reference
 for label,pp in [('primary',primary),('new_only',raw),('blend50',.5*primary+.5*reference)]:
  f=test.copy();f['which']=np.where(pp>=0,f.item1,f.item2);dest=a.out/label;dest.mkdir(exist_ok=True);f.to_csv(dest/'task_5_predictions.csv',index=False)
 np.savez_compressed(a.out/'test_predictions.npz',primary=primary,new_only=raw,reference=reference)
if __name__=='__main__':main()
