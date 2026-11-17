"""Regularized local-linear residual smoothers; fit histories exclude targets."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import pickle
from pathlib import Path
import numpy as np,pandas as pd
from task3_optimize import dataset
W=Path('/workspace/work');DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026'
SETTINGS=[('user',30.,5.,20.),('user',90.,5.,20.),('user',365.,5.,20.),('item',90.,20.,80.),('item',365.,20.,80.)]
def moments_fit(dx,r,penalty,slope_penalty):
 w=np.exp(-abs(dx));s0=w.sum(1);s1=(w*dx).sum(1);s2=(w*dx*dx).sum(1);r0=w@r;r1=(w*dx)@r;det=(s0+penalty)*(s2+slope_penalty)-s1*s1
 intercept=(r0*(s2+slope_penalty)-r1*s1)/det;slope=(r1*(s0+penalty)-r0*s1)/det
 return r0/(s0+penalty),intercept,slope,s0

def unit_checks():
 x=np.linspace(-3,3,31);r=.2+.4*x+.05*np.sin(x);dx=(x-.3)[None,:];c,b,s,n=moments_fit(dx,r,3.,10.);D=np.c_[np.ones(len(x)),dx[0]];w=np.exp(-abs(dx[0]));expected=np.linalg.solve(D.T@(w[:,None]*D)+np.diag([3.,10.]),D.T@(w*r));assert np.allclose([b[0],s[0]],expected,atol=1e-12)
 _,bb,ss,_=moments_fit(dx,r,3.,1e16);assert np.allclose(c,bb,atol=1e-12)
 c,b,s,n=moments_fit(np.zeros((2,31)),r,3.,10.);assert np.isfinite(b).all() and np.allclose(b,c) and np.allclose(s,0)
 return {'weighted_ridge_solver_matches_direct_solution':True,'infinite_slope_penalty_recovers_local_constant':True,'identical_timestamps_stable':True}

def generate(out,full=False):
 file=out/('full_features.npz' if full else 'validation_features.npz')
 if file.exists():
  z=np.load(file);return z['features'],list(z['names']),list(map(int,z['constant_indices']))
 p,tr,q,u,i,t,y,qu,qi,*_=dataset(DATA,full);assert (y>0).all()
 with open(W/'task3_optimized'/('full' if full else 'validation')/'temporal_bias.pkl','rb') as f:base=pickle.load(f)
 assert len(base.training_prediction)==len(y)
 # Check exact training row alignment and use64-bit composite temporal IDs.
 assert np.allclose(base.training_prediction,base.predict(u.astype('int64'),i.astype('int64'),t),atol=1e-7)
 residual=y-base.training_prediction;bq=base.predict(qu.astype('int64'),qi.astype('int64'),q.timestamp.to_numpy(float));qt=q.timestamp.to_numpy(float)/86400;tt=t/86400;valid=(t>=0)&(t<=4102444800);qvalid=(q.timestamp.to_numpy(float)>=0)&(q.timestamp.to_numpy(float)<=4102444800)
 features=[];names=[];constant=[]
 for key,tau,pen,spen in SETTINGS:
  vals=np.zeros((len(q),5));train_groups=tr.groupby(key,sort=False).indices
  for k,ix in q.groupby(key,sort=False).indices.items():
   rows=train_groups.get(k)
   if rows is None:continue
   rows=rows[valid[rows]];qs=ix[qvalid[ix]]
   if not len(rows) or not len(qs):continue
   for start in range(0,len(qs),64):
    take=qs[start:start+64];dx=(tt[rows][None,:]-qt[take,None])/tau;c,b,s,support=moments_fit(dx,residual[rows],pen,spen);vals[take]=np.column_stack([c,b,b-c,s,np.log1p(support)])
  start=len(features);features.extend([np.clip(bq+vals[:,0],1,5),np.clip(bq+vals[:,1],1,5),vals[:,2],vals[:,3],vals[:,4]]);constant.extend([start,start+4]);names.extend([f'{kind}_{key}_{int(tau)}' for kind in ['local_constant','local_linear','linear_minus_constant','local_slope','local_support']]);print('LOCAL_FEATURES','full' if full else 'validation',key,tau,flush=True)
 X=np.column_stack(features).astype('f');assert X.shape==(len(q),25) and np.isfinite(X).all();np.savez_compressed(file,features=X,names=names,constant_indices=constant);return X,names,constant
