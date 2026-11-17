"""Multinomial variational autoencoder: L2-normalized binary history,
input dropout, Gaussian posterior, multinomial decoder and annealed KL.
NumPy implementation with finite-difference gradient verification.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
from scipy.sparse import csr_matrix,diags

def normalize(X):
 return (diags(1/np.sqrt(np.maximum(1,np.asarray(X.multiply(X).sum(1)).ravel())))@X).tocsr().astype('f')
class MultVAE:
 def __init__(self,items,hidden=128,latent=64,seed=3181,dtype=np.float32):
  self.rng=np.random.RandomState(seed);self.latent=latent;self.steps=0
  dims=[(items,hidden),(hidden,2*latent),(latent,hidden),(hidden,items)]
  self.p={}
  for j,(a,b) in enumerate(dims):
   self.p['W'+str(j)]=(self.rng.normal(size=(a,b))*np.sqrt(2/(a+b))).astype(dtype);self.p['b'+str(j)]=np.zeros(b,dtype)
  self.m={k:np.zeros_like(v) for k,v in self.p.items()};self.v={k:np.zeros_like(v) for k,v in self.p.items()}
 def loss_grad(self,X,input_x,beta,eps):
  p=self.p;B=X.shape[0];L=self.latent
  h=np.tanh(input_x@p['W0']+p['b0']);v=h@p['W1']+p['b1'];mu=v[:,:L];raw=v[:,L:];lv=np.clip(raw,-8,8);std=np.exp(.5*lv);z=mu+std*eps
  h2=np.tanh(z@p['W2']+p['b2']);logits=h2@p['W3']+p['b3'];mx=logits.max(1);prob=np.exp(logits-mx[:,None]);sm=prob.sum(1);prob/=sm[:,None]
  count=np.asarray(X.sum(1)).ravel();rr=np.repeat(np.arange(B),np.diff(X.indptr));cc=X.indices
  reconstruction=(np.dot(count,mx+np.log(sm))-np.dot(X.data,logits[rr,cc]))/B
  kl=.5*np.sum(mu*mu+np.exp(lv)-1-lv)/B
  g=prob*count[:,None];g[rr,cc]-=X.data;g/=B
  grad={'W3':h2.T@g,'b3':g.sum(0)}
  dh=(g@p['W3'].T)*(1-h2*h2);grad['W2']=z.T@dh;grad['b2']=dh.sum(0)
  dz=dh@p['W2'].T;dm=dz+beta*mu/B;dl=(dz*eps*.5*std+beta*.5*(np.exp(lv)-1)/B)*((raw>-8)&(raw<8))
  dv=np.c_[dm,dl];grad['W1']=h.T@dv;grad['b1']=dv.sum(0);dh=(dv@p['W1'].T)*(1-h*h);grad['W0']=np.asarray(input_x.T@dh);grad['b0']=dh.sum(0)
  return float(reconstruction+beta*kl),grad
 def step(self,X,normalized,lr=.001):
  inp=normalized.copy();inp.data*=self.rng.binomial(1,.5,size=len(inp.data)).astype('f')*2
  eps=self.rng.normal(size=(X.shape[0],self.latent)).astype('f');self.steps+=1;beta=min(.2,self.steps/10000)
  loss,grad=self.loss_grad(X,inp,beta,eps)
  assert np.isfinite(loss)
  for k,g in grad.items():
   self.m[k]*=.9;self.m[k]+=.1*g;self.v[k]*=.999;self.v[k]+=.001*g*g
   self.p[k]-=lr/(1-.9**self.steps)*self.m[k]/(np.sqrt(self.v[k]/(1-.999**self.steps))+1e-8)
  return loss
 def scores(self,X,users,items,batch=256):
  # Return only queried scores, avoiding a users-by-items output cache.
  xn=normalize(X);h=np.tanh(xn@self.p['W0']+self.p['b0']);mu=(h@self.p['W1']+self.p['b1'])[:,:self.latent];dec=np.tanh(mu@self.p['W2']+self.p['b2'])
  raw=np.zeros(len(users),dtype='f');logp=np.zeros(len(users),dtype='f');groups={}
  for j,u in enumerate(users):groups.setdefault(int(u),[]).append(j)
  keys=list(groups)
  for start in range(0,len(keys),batch):
   us=np.array(keys[start:start+batch]);v=dec[us]@self.p['W3']+self.p['b3'];mx=v.max(1);norm=mx+np.log(np.exp(v-mx[:,None]).sum(1))
   for row,u in enumerate(us):
    ix=np.asarray(groups[u]);raw[ix]=v[row,items[ix]];logp[ix]=raw[ix]-norm[row]
  return raw,logp
 def save(self,path):np.savez_compressed(path,**{k:v.astype('float16') for k,v in self.p.items()})
 def restore(self,path):
  z=np.load(path);self.p={k:z[k].astype('f') for k in z.files}

def gradient_check():
 X=csr_matrix(np.array([[1,0,1,0,0,0,1],[0,1,0,1,0,0,0],[0,0,0,0,1,1,0]],float));xx=normalize(X).astype(float)
 m=MultVAE(7,4,2,dtype=np.float64);eps=np.random.RandomState(5).normal(size=(3,2));_,g=m.loss_grad(X,xx,.13,eps);errors=[]
 for k in m.p:
  for idx in [0,m.p[k].size//2,m.p[k].size-1]:
   flat=m.p[k].reshape(-1);old=flat[idx];flat[idx]=old+1e-5;hi=m.loss_grad(X,xx,.13,eps)[0];flat[idx]=old-1e-5;lo=m.loss_grad(X,xx,.13,eps)[0];flat[idx]=old
   errors.append(abs((hi-lo)/2e-5-g[k].reshape(-1)[idx]))
 assert max(errors)<1e-6,max(errors)
 return max(errors)
