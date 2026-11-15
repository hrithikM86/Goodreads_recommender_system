"""Neural five-rating classification and expectation predictions for Task 1."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from task1_next_stack import matrices
from task1_improve import dataset

RATINGS=np.arange(1,6,dtype=np.float32)
def softmax(z):
 z=z-z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)
def head_gradient(p,y,lam):
 mean=p@RATINGS;g=p.copy();g[np.arange(len(y)),y.astype(int)-1]-=1;g+=2*lam*(mean-y)[:,None]*p*(RATINGS[None,:]-mean[:,None]);return g/len(y)
def blend(y,new,ref):
 delta=new-ref;w=float(np.clip(np.dot(y-ref,delta)/max(np.dot(delta,delta),1e-12),0,1));return float(np.mean((y-(ref+w*delta))**2)),w

def inputs(a,full=False):
 arg=SimpleNamespace(data_dir=a.data_dir,legacy=a.legacy,previous=a.previous,out=a.reference,original=a.original);X,names,q,n=matrices(arg,full);p,tr,_,u,i,qu,qi,users,items,_=dataset(a.data_dir,full);prior=np.bincount(tr.rating.astype(int)-1,minlength=5)/len(tr)
 for key,ids in [('user',q.user),('item',q.item)]:
  table=pd.crosstab(tr[key],tr.rating).reindex(columns=[1,2,3,4,5],fill_value=0);counts=table.sum(1).to_numpy();freq=(table.to_numpy()+20*prior)/(counts[:,None]+20);mapped=pd.Index(table.index).get_indexer(ids);v=np.tile(prior,(len(q),1));ok=mapped>=0;v[ok]=freq[mapped[ok]];X=np.c_[X,v];names += [f'{key}_freq{r}' for r in range(1,6)]
 z=np.load(a.out/('full' if full else 'validation')/'cf_probabilities.npz')
 for key in z.files:X=np.c_[X,z[key],z[key]@RATINGS];names += [key+'_p'+str(r) for r in range(1,6)]+[key+'_mean']
 assert np.isfinite(X).all();return X.astype('f'),names,q,n

def transform(X,names):
 X=X.copy()
 for j,name in enumerate(names):
  if name.endswith('_count'):X[:,j]=np.log1p(X[:,j])
 return X

def inner_reference(a,X,n):
 r=json.loads((a.reference/'results.json').read_text());prev=json.loads((a.previous/'results.json').read_text());new=np.zeros(n);old=np.zeros(n)
 for key,w in r['depth_weights'].items():
  if w==0:continue
  m=CatBoostRegressor();m.load_model(str(a.reference/(key+'_inner.cbm')));new+=w*np.clip(m.predict(X[:n,:44],thread_count=1),1,5)
 for key,w in prev['weights'].items():
  m=CatBoostRegressor();m.load_model(str(a.previous/(key+'_inner.cbm')));old+=w*np.clip(m.predict(X[:n,:23],thread_count=1),1,5)
 return r['new_weight']*new+(1-r['new_weight'])*old

class Neural:
 def __init__(self,dims,seed,lam,drop=.1):
  self.seed=seed;self.lam=lam;self.drop=drop;rng=np.random.RandomState(seed);self.W=[(rng.normal(size=(l,r))*np.sqrt(2/l)).astype('f') for l,r in zip(dims[:-1],dims[1:])];self.W[-1]*=.05;self.b=[np.zeros(r,'f') for r in dims[1:]];self.mean=None;self.std=None
 def probabilities(self,X):
  out=[]
  for rows in np.array_split(np.arange(len(X)),max(1,(len(X)+4095)//4096)):
   a=np.clip((X[rows]-self.mean)/self.std,-10,10)
   for W,b in zip(self.W[:-1],self.b[:-1]):a=np.maximum(a@W+b,0)
   logits=a@self.W[-1]+self.b[-1]-(RATINGS[None,:]-X[rows,0,None])**2/(2*.7);out.append(softmax(logits))
  return np.concatenate(out)
 def save(self,path):
  z={**{f'W{j}':w for j,w in enumerate(self.W)},**{f'b{j}':b for j,b in enumerate(self.b)},'mean':self.mean,'std':self.std,'seed':self.seed,'lam':self.lam,'drop':self.drop};temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**z);temp.replace(path)
 @classmethod
 def load(cls,path):
  z=dict(np.load(path));W=[z[k] for k in z if k.startswith('W')];m=cls([W[0].shape[0]]+[w.shape[1] for w in W],int(z['seed']),float(z['lam']),float(z['drop']));m.W=W;m.b=[z[f'b{j}'] for j in range(len(W))];m.mean=z['mean'];m.std=z['std'];return m
 def fit(self,X,y,epochs,validation=None,path=None):
  self.mean=X.mean(0);self.std=np.maximum(X.std(0),.05);Z=np.clip((X-self.mean)/self.std,-10,10);anchor=-(RATINGS[None,:]-X[:,0,None])**2/(2*.7);rng=np.random.RandomState(self.seed);params=self.W+self.b;moment=[np.zeros_like(p) for p in params];variance=[np.zeros_like(p) for p in params];step=0;best=1e9;best_epoch=0;report=None
  for epoch in range(1,epochs+1):
   order=rng.permutation(len(X));lr=.001*(.985**(epoch-1))
   for ids in np.array_split(order,max(1,(len(order)+511)//512)):
    acts=[Z[ids]];masks=[];pre=[]
    for W,b in zip(self.W[:-1],self.b[:-1]):
     v=acts[-1]@W+b;pre.append(v);mask=(rng.random_sample(v.shape)>=self.drop).astype('f')/(1-self.drop);masks.append(mask);acts.append(np.maximum(v,0)*mask)
    prob=softmax(acts[-1]@self.W[-1]+self.b[-1]+anchor[ids]);g=head_gradient(prob,y[ids],self.lam);gw=[None]*len(self.W);gb=[None]*len(self.b)
    for layer in range(len(self.W)-1,-1,-1):
     gw[layer]=acts[layer].T@g+.0001*self.W[layer];gb[layer]=g.sum(0)
     if layer:g=(g@self.W[layer].T)*(pre[layer-1]>0)*masks[layer-1]
    step+=1
    for j,grad in enumerate(gw+gb):
     moment[j]*=.9;moment[j]+=.1*grad;variance[j]*=.999;variance[j]+=.001*grad*grad;params[j]-=lr*(moment[j]/(1-.9**step))/(np.sqrt(variance[j]/(1-.999**step))+1e-8)
   if validation is not None:
    V,vy,ref=validation;pred=self.probabilities(V)@RATINGS;score,w=blend(vy,pred,ref)
    if score<best-1e-7:best=score;best_epoch=epoch;report={'epochs':epoch,'inner_MSE':float(np.mean((vy-pred)**2)),'inner_blend_MSE':score,'neural_weight':w};self.save(path)
    if epoch%10==0:print('NN epoch',epoch,'blend MSE',score,'best',best,flush=True)
    if epoch-best_epoch>=15:break
   elif epoch%20==0:print('NN refit epoch',epoch,'/',epochs,flush=True)
  return report

def main():
 ap=argparse.ArgumentParser()
 for key in ['data-dir','legacy','previous','reference','original','out','accepted']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--predict',action='store_true');a=ap.parse_args();X,names,q,n=inputs(a);y=q.rating.to_numpy(dtype='f');idx=np.random.RandomState(17).permutation(n);tr=idx[:int(.8*n)];va=idx[int(.8*n):];ref=inner_reference(a,X,n);X=transform(X,names);reference=np.load(a.reference/'evaluation.npz')['primary'];reports={};configs={'ce64':([64,32],0.,42),'joint64':([64,32],2.,73),'joint128':([128,64],2.,117)}
 for name,(hidden,lam,seed) in configs.items():
  path=a.out/(name+'_inner.npz');info=a.out/(name+'_info.json')
  if info.exists():r=json.loads(info.read_text())
  else:m=Neural([X.shape[1]]+hidden+[5],seed,lam);r=m.fit(X[tr],y[tr],120,(X[va],y[va],ref[va]),path);r.update(hidden=hidden,lam=lam,seed=seed);info.write_text(json.dumps(r,indent=2))
  reports[name]=r;print('NEURAL_INNER',name,r,flush=True)
 name=min(reports,key=lambda k:reports[k]['inner_blend_MSE']);r=reports[name];path=a.out/(name+'_evaluation.npz')
 if path.exists():m=Neural.load(path)
 else:m=Neural([X.shape[1]]+r['hidden']+[5],r['seed'],r['lam']);m.fit(X[:n],y[:n],r['epochs']);m.save(path)
 prob=m.probabilities(X[n:]);pred=prob@RATINGS;w=r['neural_weight'];blended=w*pred+(1-w)*reference;mse=lambda p:float(np.mean((y[n:]-p)**2));cf=np.load(a.out/'validation/cf_probabilities.npz');cf_reports={}
 for key in cf.files:
  cfp=cf[key]@RATINGS;s,cw=blend(y[va],cfp[va],ref[va]);cf_reports[key]={'inner_blend_MSE':s,'neural_weight':cw,'evaluation_MSE':mse(cfp[n:]),'evaluation_blend_MSE':mse(cw*cfp[n:]+(1-cw)*reference)}
 summary={'selected':name,'runs':reports,'features':names,'neural_weight':w,'reference_evaluation_MSE':mse(reference),'neural_only_evaluation_MSE':mse(pred),'blend_evaluation_MSE':mse(blended),'cf_runs':cf_reports,'reference_public_MSE':.5368518492394629,'classes':[1,2,3,4,5],'zero_ratings_excluded':True};(a.out/'results.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(a.out/'evaluation_predictions.npz',y=y[n:],reference=reference,probabilities=prob,neural=pred,blend=blended);print('NEURAL_RESULT',summary['reference_evaluation_MSE'],summary['neural_only_evaluation_MSE'],summary['blend_evaluation_MSE'],flush=True)
 if not a.predict:return
 FX,fn,test,_=inputs(a,True);assert fn==names;FX=transform(FX,names);reference_csv=pd.read_csv(a.accepted);pd.testing.assert_frame_equal(reference_csv.iloc[:,:-1],test.iloc[:,:-1]);path=a.out/(name+'_full.npz')
 if path.exists():m=Neural.load(path)
 else:m=Neural([X.shape[1]]+r['hidden']+[5],r['seed'],r['lam']);m.fit(X,y,r['epochs']);m.save(path)
 prob=m.probabilities(FX);pred=prob@RATINGS;blended=w*pred+(1-w)*reference_csv.rating.to_numpy()
 for label,values in [('neural_only',pred),('primary',blended),('blend50',.5*blended+.5*reference_csv.rating.to_numpy())]:
  folder=a.out/label;folder.mkdir(exist_ok=True);frame=test.copy();frame['rating']=np.clip(values,1,5);assert np.isfinite(values).all();frame.to_csv(folder/'task_1_predictions.csv',index=False)
 np.savez_compressed(a.out/'test_predictions.npz',probabilities=prob,neural=pred,primary=blended)
if __name__=='__main__':main()
