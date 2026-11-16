"""Blend fixed tree combiners. Reserve half of historical evaluation for a blend check.
These cohorts have been reused previously; results are local diagnostics, not unbiased estimates.
"""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
W=Path('/workspace/work');T=W/'task4_tree_combiners';R=W/'task4_two_approaches';OUT=W/'task4_tree_blend';OUT.mkdir(exist_ok=True)
def acc(x):return float(np.mean((x>0)+.5*(x==0)))
def main():
 names=['catboost','xgboost','lightgbm'];ev=np.column_stack([np.load(R/'fresh_margins.npz')['evaluation'],np.load(T/'xgboost_evaluation.npz')['margin'],np.load(T/'lightgbm_evaluation.npz')['margin']]);test=np.column_stack([np.load(R/'fresh_margins.npz')['test'],np.load(T/'xgboost_test.npz')['margin'],np.load(T/'lightgbm_test.npz')['margin']]);assert np.isfinite(ev).all() and np.isfinite(test).all();idx=np.random.RandomState(923).permutation(len(ev));sel,check=np.split(idx,[len(idx)//2]);choices=[]
 # Require a contribution from all three for the requested ensemble candidate.
 for c in range(1,9):
  for x in range(1,10-c):
   l=10-c-x
   if l<1:continue
   weights=np.array([c,x,l])/10;score=acc(ev[sel]@weights);choices.append((score,-np.sum((weights-1/3)**2),weights))
 score,_,weights=max(choices,key=lambda x:(x[0],x[1]));p=test@weights;data=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026';q=pd.read_csv(data/'task_4_test.csv');accepted=pd.read_csv('/workspace/output/task4_lightgbm/task_4_predictions.csv');assert len(test)==len(q)==20000
 for col,name in enumerate(names):
  folder={'catboost':'task4_all_models_retrained_fm','xgboost':'task4_xgboost','lightgbm':'task4_lightgbm'}[name];old=pd.read_csv(Path('/workspace/output')/folder/'task_4_predictions.csv');assert np.array_equal(old.which,np.where(test[:,col]>=0,q.item1,q.item2))
 f=q.copy();f['which']=np.where(p>=0,f.item1,f.item2);dest=Path('/workspace/output/task4_three_tree_ensemble');dest.mkdir(exist_ok=True);path=dest/'task_4_predictions.csv';f.to_csv(path,index=False);reread=pd.read_csv(path);assert reread.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert ((reread.which==reread.item1)|(reread.which==reread.item2)).all();assert np.array_equal(reread.which,np.where(p>=0,q.item1,q.item2));assert np.allclose((-test)@weights,-p)
 r={'weights':dict(zip(names,weights.tolist())),'blend_selection_accuracy':score,'blend_check_accuracy':acc(ev[check]@weights),'whole_historical_evaluation_accuracy':acc(ev@weights),'components':{name:{'selection':acc(ev[sel,k]),'check':acc(ev[check,k]),'whole':acc(ev[:,k])} for k,name in enumerate(names)},'rows':len(f),'changed_vs_public_best_lightgbm':int((reread.which!=accepted.which).sum()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'protocol':'Fixed component models; split prior evaluation seed923 into blend-selection/check halves; weights selected on selection only. All three weights >=.1. Cohorts reused across earlier experiments; not unbiased hidden-test estimates. No test labels used.'};(OUT/'results.json').write_text(json.dumps(r,indent=2));np.savez_compressed(OUT/'margins.npz',evaluation=ev@weights,test=p);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
