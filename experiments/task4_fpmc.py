"""Query-time FPMC for Task4: reader preference plus previous-book transition.
Known following-book transition scores provide an additional interpolation feature.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from task4_two_approaches import queries
from task4_tree_combiners import W,DATA
OUT=W/'task4_fpmc';OUT.mkdir(exist_ok=True)
NAMES=['personal','previous_transition','following_transition','forward_joint','time180_bidirectional_joint']
def library():
 lib=ctypes.CDLL(str(OUT/'fpmc.so'));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.train_fpmc.argtypes=[ctypes.c_int]*2+[ip]*6+[ctypes.c_int,ip,ctypes.c_int]+[fp]*5+[ctypes.c_int]*3+[ctypes.c_float]*3+[ctypes.c_int];return lib

def contexts(u,i,t,qu,qt,U):
 good=(t>=0)&(t<4102444800);order=np.lexsort((np.arange(len(u)),t,u));order=order[good[order]];hu=u[order];hi=i[order];ht=t[order];ptr=np.searchsorted(hu,np.arange(U+1));previous=np.full(len(qu),-1,'i');following=np.full(len(qu),-1,'i');pw=np.zeros(len(qu),'f');nw=np.zeros(len(qu),'f');qo=np.argsort(qu,kind='stable');keys,starts=np.unique(qu[qo],return_index=True);ends=np.r_[starts[1:],len(qo)]
 for user,a,b in zip(keys,starts,ends):
  lo,end=ptr[user:user+2];dates=ht[lo:end];books=hi[lo:end];rows=qo[a:b];valid=(qt[rows]>=0)&(qt[rows]<4102444800);rows=rows[valid]
  if not len(dates) or not len(rows):continue
  before=np.searchsorted(dates,qt[rows],side='left')-1;after=np.searchsorted(dates,qt[rows],side='right');has=before>=0;ix=rows[has];previous[ix]=books[before[has]];pw[ix]=np.exp(-(qt[ix]-dates[before[has]])/(86400*180));has=after<len(dates);ix=rows[has];following[ix]=books[after[has]];nw[ix]=np.exp(-(dates[after[has]]-qt[ix])/(86400*180))
 return previous,following,pw,nw,hu,hi,ht

def score(P,Q,A,B,bias,qu,qi,prev,nxt,pw,nw):
 personal=np.einsum('ij,ij->i',P[qu],Q[qi])+bias[qi];forward=np.zeros(len(qi),'f');backward=np.zeros(len(qi),'f');good=prev>=0;forward[good]=np.einsum('ij,ij->i',A[prev[good]],B[qi[good]]);good=nxt>=0;backward[good]=np.einsum('ij,ij->i',A[qi[good]],B[nxt[good]]);return [personal,forward,backward,personal+forward,personal+pw*forward+nw*backward]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');a=ap.parse_args();lib=library();u,i,t,qu,qi,qt,U,I=queries(a.full);assert (qu>=0).all() and (qi>=0).all();prev,nxt,pw,nw,hu,hi,ht=contexts(u,i,t,qu,qt,U);mask=(hu[1:]==hu[:-1])&(ht[1:]>ht[:-1]);tu=hu[1:][mask].astype('i');last=hi[:-1][mask].astype('i');target=hi[1:][mask].astype('i');assert len(tu)>0 and (np.diff(tu)>=0).all();X=csr_matrix((np.ones(len(u)),(u,i)),shape=(U,I));X.sort_indices();ptr=X.indptr.astype('i');seen=X.indices.astype('i');source_count=np.bincount(last,minlength=I);target_count=np.bincount(target,minlength=I);user_count=np.bincount(tu,minlength=U);active=np.flatnonzero(target_count>0).astype('i');pool=hi.astype('i');k=32;rng=np.random.RandomState(1049);P=rng.normal(0,.05,(U,k)).astype('f');Q=rng.normal(0,.05,(I,k)).astype('f');A=rng.normal(0,.05,(I,k)).astype('f');B=rng.normal(0,.05,(I,k)).astype('f');bias=np.zeros(I,'f');old=0;result={};start=time.time();qu=qu.astype('i');qi=qi.astype('i')
 for epoch in [15,40,100]:
  lib.train_fpmc(len(tu),k,tu,last,target,ptr,seen,pool,len(pool),active,len(active),P,Q,A,B,bias,old,epoch-old,1049,.02,.02,.5,0);old=epoch;P[user_count==0]=0;Q[target_count==0]=0;A[source_count==0]=0;B[target_count==0]=0;bias[target_count==0]=0;assert all(np.isfinite(x).all() for x in [P,Q,A,B,bias]);values=score(P,Q,A,B,bias,qu,qi,prev,nxt,pw,nw);assert all(np.isfinite(x).all() for x in values);result.update({f'fpmc_e{epoch}_{name}':value for name,value in zip(NAMES,values)});print('FPMC',a.full,epoch,'seconds',round(time.time()-start),flush=True)
 stage='full' if a.full else 'validation';np.savez_compressed(OUT/(stage+'_features.npz'),**result)
 if a.full:np.savez_compressed(OUT/'full_factors.npz',P=P,Q=Q,A=A,B=B,bias=bias)
 r={'stage':stage,'factors':32,'epochs':[15,40,100],'seed':1049,'learning_rate':.02,'L2':.02,'decay_per_epoch':.98,'negative_sampling':'half fitting-event popularity, half active-book uniform; reject known reader books','training_transitions':len(tu),'strict_time_transitions':True,'features':list(result),'query_rows':len(qu),'no_previous_context_rows':int((prev<0).sum()),'no_following_context_rows':int((nxt<0).sum()),'context':'strictly earlier/later KNOWN fitting events relative to supplied query timestamp; tied timestamps do not establish direction','ratings_used':False,'zero_rating_events_retained':True,'input_hashes':{name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['tasks_3_4_train.csv','task_4_test.csv']}};(OUT/(stage+'_report.json')).write_text(json.dumps(r,indent=2));print('EXPORTED',stage,flush=True)
if __name__=='__main__':main()
