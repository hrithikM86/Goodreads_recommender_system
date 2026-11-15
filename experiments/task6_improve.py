"""MAE gap prediction with leakage-safe conditional book-date distributions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd
from task6_pipeline import load_data

def event_features(item,t,allowed,candidate,last,pace,target=None):
    target=np.full(len(candidate),-1,int) if target is None else np.asarray(target);valid=allowed&(t>=0)&(t<=4102444800);rows=np.flatnonzero(valid);rows=rows[np.lexsort((t[rows],item[rows]))];codes=item[rows];dates=t[rows]/86400;ptr=np.searchsorted(codes,np.arange(item.max()+2));rank=np.full(len(t),-1,int);rank[rows]=np.arange(len(rows));out=np.full((len(candidate),29),np.nan,np.float32);out[:,0]=0;out[:,-1]=1;groups={}
    for z,j in enumerate(candidate):
        if j>=0:groups.setdefault(j,[]).append(z)
    for j,indices in groups.items():
        lo,hi=ptr[j:j+2];A=dates[lo:hi]
        if not len(A):continue
        for qs in np.array_split(np.array(indices),max(1,(len(indices)+127)//128)):
            q=last[qs]/86400;gap=A[None,:]-q[:,None];weights=(gap>=0)&(last[qs,None]>=0)&(last[qs,None]<=4102444800);removed=np.where(target[qs]>=0,rank[np.maximum(target[qs],0)]-lo,-1);rem=(removed>=0)&(removed<len(A));weights[np.flatnonzero(rem),removed[rem]]=False;count=weights.sum(1);ok=count>0;out[qs,0]=np.log1p(count);out[qs,-1]=~ok
            total=weights.sum(1);mean=(np.maximum(gap,0)*weights).sum(1)/np.maximum(total,1);var=((np.maximum(gap,0)**2)*weights).sum(1)/np.maximum(total,1)-mean**2;out[qs,1]=np.where(ok,np.log1p(np.maximum(var,0)**.5),np.nan);out[qs,2]=np.where(ok,np.log1p(mean),np.nan);cumulative=np.cumsum(weights,axis=1)
            for col,quantile in enumerate([0,.1,.25,.5,.75,.9,1],3):
                # Linear interpolation on the actual leave-one-event-out future set.
                frac=quantile*np.maximum(count-1,0);left=np.floor(frac).astype(int);right=np.ceil(frac).astype(int);il=np.argmax(cumulative>left[:,None],axis=1);ir=np.argmax(cumulative>right[:,None],axis=1);value=A[il]*(1-(frac-left))+A[ir]*(frac-left)-q;out[qs,col]=np.where(ok,np.log1p(np.maximum(value,0)),np.nan)
            for col,window in enumerate([7,30,90,365],10):
                c=(weights&(gap<=window)).sum(1);out[qs,col]=np.log1p(c);out[qs,col+4]=np.divide(c,count,out=np.zeros(len(q)),where=ok)
            for k,sigma in enumerate([1.,2.]):
                w=np.exp(-.5*((np.log1p(np.maximum(gap,0))-pace[qs,None])/sigma)**2)*weights;mass=w.sum(1);cw=np.cumsum(w,axis=1);good=mass>1e-12;offset=18+5*k
                for col,quantile in enumerate([.25,.5,.75]):
                    where=np.argmax(cw>=quantile*mass[:,None],axis=1);out[qs,offset+col]=np.where(good,np.log1p(np.maximum(A[where]-q,0)),np.nan)
                mean=(w*np.maximum(gap,0)).sum(1)/np.maximum(mass,1e-12);out[qs,offset+3]=np.where(good,np.log1p(mean),np.nan);out[qs,offset+4]=np.log1p(mass)
    assert not np.isinf(out).any();return out

def build(a):
    a.out.mkdir(parents=True,exist_ok=True);manifest={f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['task_6_train.csv','task_6_test.csv']};path=a.out/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    path.write_text(json.dumps(manifest,indent=2));p,test,u,users,i,items,t,ratings,starts,early,v,e,rows=load_data(a.data_dir);z=np.load(a.legacy/'task6_rich.npz');old=np.load(a.legacy/'task6_final/book_features.npz');full=np.load(a.legacy/'task6_final/full_features.npz');assert np.array_equal(rows,old['all_rows']);core=np.r_[z['X'],z['V'],z['T']];y=np.r_[z['y'],z['vy'],z['ty']];assert np.allclose(y,(t[rows]-t[rows-1])/86400);n=len(early);m=len(v)
    if not (a.out/'validation.npz').exists():
        allowed=np.ones(len(p),bool);allowed[v]=False;allowed[e]=False;extras=event_features(i,t,allowed,i[rows],t[rows-1],core[:,19],rows);book=np.r_[old['X'],old['V'],old['T']];X=np.c_[core,book,extras];progress=(rows-starts[u[rows]])/(starts[u[rows]+1]-starts[u[rows]]-1);np.savez_compressed(a.out/'validation.npz',X=X,y=y,rows=rows,n=n,m=m,progress=progress);print('Task6 conditional validation features ready',X.shape,flush=True)
    if a.full and not (a.out/'full.npz').exists():
        extras=event_features(i,t,np.ones(len(p),bool),i[rows],t[rows-1],core[:,19],rows);um=pd.Index(users).get_indexer(test.user);last=t[starts[um+1]-1];qi=pd.Index(items).get_indexer(test.item);qextras=event_features(i,t,np.ones(len(p),bool),qi,last,full['query_rich'][:,19]);X=np.c_[core,full['book'],extras];Q=np.c_[full['query_rich'],full['query_book'],qextras];np.savez_compressed(a.out/'full.npz',X=X,Q=Q,rows=rows,y=y);print('Task6 full conditional features ready',X.shape,Q.shape,flush=True)

def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','legacy','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--full',action='store_true');a=ap.parse_args();build(a)
if __name__=='__main__':main()
