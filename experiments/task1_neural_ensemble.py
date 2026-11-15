"""Simplex ensemble of independent five-class neural classifiers, selected on inner data."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize
from task1_neural import inputs,transform,inner_reference,Neural,RATINGS

def main():
 ap=argparse.ArgumentParser()
 for key in ['data-dir','legacy','previous','reference','original','out','accepted']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--predict',action='store_true');a=ap.parse_args();X,names,q,n=inputs(a);ref=inner_reference(a,X,n);X=transform(X,names);y=q.rating.to_numpy(dtype='f');idx=np.random.RandomState(17).permutation(n);va=idx[int(.8*n):];r=json.loads((a.out/'results.json').read_text());keys=list(r['runs']);inner={'accepted':ref[va]}
 for name in keys:inner[name]=Neural.load(a.out/(name+'_inner.npz')).probabilities(X[va])@RATINGS
 namesw=list(inner);P=np.column_stack(list(inner.values()));obj=lambda w:float(np.mean((P@w-y[va])**2));opt=minimize(obj,np.ones(len(namesw))/len(namesw),method='SLSQP',bounds=[(0,1)]*len(namesw),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'ftol':1e-12,'maxiter':300});assert opt.success;w=np.maximum(0,opt.x);w[w<1e-5]=0;w/=w.sum();weights=dict(zip(namesw,w.tolist()));reference=np.load(a.reference/'evaluation.npz')['primary'];preds={'accepted':reference};probs={}
 for name in keys:
  if weights[name]==0:continue
  cfg=r['runs'][name];path=a.out/(name+'_evaluation.npz')
  if path.exists():m=Neural.load(path)
  else:m=Neural([X.shape[1]]+cfg['hidden']+[5],cfg['seed'],cfg['lam']);m.fit(X[:n],y[:n],cfg['epochs']);m.save(path)
  probs[name]=m.probabilities(X[n:]);preds[name]=probs[name]@RATINGS
 primary=sum(weights[k]*v for k,v in preds.items());nnweight=1-weights['accepted'];neural=sum(weights[k]*v for k,v in preds.items() if k!='accepted')/nnweight;summary={'weights':weights,'inner_MSE':obj(w),'reference_evaluation_MSE':float(np.mean((y[n:]-reference)**2)),'neural_evaluation_MSE':float(np.mean((y[n:]-neural)**2)),'primary_evaluation_MSE':float(np.mean((y[n:]-primary)**2)),'selected_on':'seed17 inner validation only','reference_public_MSE':.5368518492394629};(a.out/'ensemble_results.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(a.out/'ensemble_evaluation.npz',y=y[n:],reference=reference,neural=neural,primary=primary);print('NEURAL_ENSEMBLE',summary,flush=True)
 if not a.predict:return
 FX,fn,test,_=inputs(a,True);assert fn==names;FX=transform(FX,names);refcsv=pd.read_csv(a.accepted);pd.testing.assert_frame_equal(refcsv.iloc[:,:-1],test.iloc[:,:-1]);preds={'accepted':refcsv.rating.to_numpy()};probability=np.zeros((len(test),5));
 for name in keys:
  if weights[name]==0:continue
  cfg=r['runs'][name];path=a.out/(name+'_full.npz')
  if path.exists():m=Neural.load(path)
  else:m=Neural([X.shape[1]]+cfg['hidden']+[5],cfg['seed'],cfg['lam']);m.fit(X,y,cfg['epochs']);m.save(path)
  p=m.probabilities(FX);preds[name]=p@RATINGS;probability+=weights[name]*p/nnweight
 primary=sum(weights[k]*v for k,v in preds.items());neural=probability@RATINGS
 for label,pred in [('ensemble_primary',primary),('ensemble_neural_only',neural),('ensemble_blend50',.5*primary+.5*refcsv.rating.to_numpy())]:
  folder=a.out/label;folder.mkdir(exist_ok=True);f=test.copy();f['rating']=np.clip(pred,1,5);assert np.isfinite(pred).all();f.to_csv(folder/'task_1_predictions.csv',index=False)
 np.savez_compressed(a.out/'ensemble_test_predictions.npz',probabilities=probability,neural=neural,primary=primary)
if __name__=='__main__':main()
