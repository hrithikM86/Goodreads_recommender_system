"""Time-weighted FISM: directly learn pooled history/candidate factors with BPR."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from task4_two_approaches import queries
from task4_tree_combiners import W,DATA
OUT=W/'task4_fism';OUT.mkdir(exist_ok=True)
NAMES=['time180_score','uniform_local_score','time180_cosine','past_score']
def library():
 lib=ctypes.CDLL(str(OUT/'fism.so'));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
 lib.prepare.argtypes=[ctypes.c_int]*2+[ip]*2+[dp]+[ip]*2+[fp]+[ip]*2
 lib.train.argtypes=[ctypes.c_int]*4+[ip]*6+[fp]+[ip]+[fp]*3+[ctypes.c_int]*3
 lib.score.argtypes=[ctypes.c_int]*3+[ip]*2+[dp]+[ip]*2+[dp]+[fp]*4
 return lib

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');a=ap.parse_args();lib=library();u,i,t,qu,qi,qt,U,I=queries(a.full);assert (qi>=0).all();seen=csr_matrix((np.ones(len(u)),(u,i)),shape=(U,I));seen.sort_indices();sp=seen.indptr.astype('i');si=seen.indices.astype('i');order=np.lexsort((np.arange(len(u)),t,u));order=order[(t[order]>=0)&(t[order]<4102444800)];uu=u[order].astype('i');ii=i[order].astype('i');tt=t[order].astype('d');ptr=np.searchsorted(uu,np.arange(U+1)).astype('i');n=len(ii);k=32;cap=32;nb=np.full((n,cap),-1,'i');coef=np.zeros((n,cap),'f');count=np.zeros(n,'i');context_count=np.zeros(I,'i');lib.prepare(n,cap,uu,ii,tt,ptr,nb,coef,count,context_count);assert np.isfinite(coef).all();rng=np.random.RandomState(937);A=rng.normal(0,.03,(I,k)).astype('f');B=rng.normal(0,.03,(I,k)).astype('f');bias=np.zeros(I,'f');pop=np.bincount(ii,minlength=I);old=0;result={};start=time.time();qu=qu.astype('i');qi=qi.astype('i');qt=qt.astype('d')
 for epoch in [5,15,30]:
  lib.train(n,k,cap,I,uu,ii,ptr,sp,si,nb,coef,count,A,B,bias,old,epoch-old,937);old=epoch;assert all(np.isfinite(x).all() for x in [A,B,bias]);A[context_count==0]=0;B[pop==0]=0;bias[pop==0]=0;value=np.zeros((len(qu),4),'f');lib.score(len(qu),k,cap,ptr,ii,tt,qu,qi,qt,A,B,bias,value);assert np.isfinite(value).all();result.update({f'fism_e{epoch}_{name}':value[:,j].copy() for j,name in enumerate(NAMES)});print('FISM',a.full,epoch,'seconds',round(time.time()-start),flush=True)
 stage='full' if a.full else 'validation';np.savez_compressed(OUT/(stage+'_features.npz'),**result)
 if a.full:np.savez_compressed(OUT/'full_factors.npz',history=A,target=B,bias=bias)
 report={'stage':stage,'training_rows_with_valid_times':n,'factors':k,'context_cap':cap,'normalization_power':.5,'time_bandwidth_days':180,'epochs':[5,15,30],'loss':'BPR pairwise logistic; same target-excluded context for positive/negative','negative_sampling':'half uniform global book, half training-event popularity; reject all known books of reader, max100 attempts','learning_rate':.025,'decay_per_epoch':.98,'L2':.01,'seed':937,'features':list(result),'queries':len(qu),'input_hashes':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['tasks_3_4_train.csv','task_4_test.csv']},'zero_rating_interactions_retained':True,'ratings_used':False};(OUT/(stage+'_report.json')).write_text(json.dumps(report,indent=2));print('EXPORTED',stage,flush=True)
if __name__=='__main__':main()
