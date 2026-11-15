"""Peer book-duration statistics: leave-query-user-out, prefix-only validation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from task6_pipeline import load_data

def peers(i,t,u,allowed,candidate,last,queryuser,pace):
 gap=np.r_[0,np.diff(t)/86400];same=np.r_[False,u[1:]==u[:-1]];valid=allowed&same&(gap>=0)&(t>=0)&(t<=4102444800)
 lg=np.log1p(np.maximum(gap,0));# past mean log gap excludes the current event
 csum=np.cumsum(np.where(same,lg,0));starts=np.searchsorted(u,np.arange(u.max()+2));position=np.arange(len(u))-starts[u];base=np.r_[0,csum[:-1]];prior=(base-np.where(starts[u]>0,csum[np.maximum(starts[u]-1,0)],0))/np.maximum(position-1,1);prior[position<=1]=lg[position<=1] # no pace available at first event
 valid&=position>1
 rows=np.flatnonzero(valid);rows=rows[np.argsort(i[rows],kind='stable')];ids=i[rows];ptr=np.searchsorted(ids,np.arange(i.max()+2));out=np.full((len(candidate),23),np.nan,'f');out[:,0]=0;groups={}
 for z,j in enumerate(candidate):
  if j>=0:groups.setdefault(j,[]).append(z)
 for j,indices in groups.items():
  lo,hi=ptr[j:j+2];rr=rows[lo:hi]
  if not len(rr):continue
  for qs in np.array_split(np.array(indices),max(1,(len(indices)+63)//64)):
   mask=u[rr][None,:]!=queryuser[qs,None];g=lg[rr];norm=g-prior[rr];D=abs(t[rr][None,:]-last[qs,None])/86400;count=mask.sum(1);out[qs,0]=np.log1p(count)
   for z,row in enumerate(qs):
    good=mask[z]
    if not good.any():continue
    vals=g[good];out[row,1:6]=np.quantile(vals,[.1,.25,.5,.75,.9]);out[row,6:9]=[vals.mean(),vals.std(),np.mean(vals==0)];out[row,9:14]=np.quantile(norm[good],[.1,.25,.5,.75,.9])+pace[row]
   for kind,w in enumerate([np.exp(-D/180)*mask,np.exp(-D/730)*mask,np.exp(-.5*((prior[rr][None,:]-pace[qs,None])/1.5)**2)*mask]):
    mass=w.sum(1);out[qs,14+3*kind]=np.log1p(mass);out[qs,15+3*kind]=(w@g)/np.maximum(mass,1e-12);order=np.argsort(g);cw=np.cumsum(w[:,order],axis=1);where=np.argmax(cw>=.5*mass[:,None],axis=1);out[qs,16+3*kind]=np.where(mass>1e-8,g[order[where]],np.nan)
 assert not np.isinf(out).any();return out

def main():
 ap=argparse.ArgumentParser()
 for key in ['data-dir','previous','out']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--full',action='store_true');a=ap.parse_args();a.out.mkdir(exist_ok=True,parents=True);path=a.out/('full_peer.npz' if a.full else 'validation_peer.npz')
 if path.exists():return
 p,test,u,users,i,items,t,ratings,starts,early,v,e,rows=load_data(a.data_dir);allowed=np.ones(len(p),bool)
 if not a.full:allowed[v]=False;allowed[e]=False
 old=np.load(a.previous/('full.npz' if a.full else 'validation.npz'));core=old['X'];out=peers(i,t,u,allowed,i[rows],t[rows-1],u[rows],core[:,19]);result={'X':out}
 if a.full:
  qu=pd.Index(users).get_indexer(test.user);qi=pd.Index(items).get_indexer(test.item);result['Q']=peers(i,t,u,allowed,qi,t[starts[qu+1]-1],qu,old['Q'][:,19])
 np.savez_compressed(path,**result);print('Task6 peer features ready',a.full,out.shape,flush=True)
if __name__=='__main__':main()
