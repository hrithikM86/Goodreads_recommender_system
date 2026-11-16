"""Chronological Item2vec SGNS scores for Task4; no heldout events in embeddings/context."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,subprocess,json,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd
from task4_two_approaches import queries
from task4_tree_combiners import W,DATA
OUT=W/'task4_item2vec';OUT.mkdir(exist_ok=True)
NAMES=['cos_mean','cos_max','cos_prev','cos_next','cos_prev3','cos_next3','cos_time30','cos_time180','cos_time730','sgns_mean','sgns_time180','sgns_prev']
def library():
 so=OUT/'item2vec.so'
 if not so.exists():subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_suffix('.cpp')),'-o',str(so)],check=True)
 lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');lib.train.argtypes=[ctypes.c_int]*4+[ip]*3+[ctypes.c_int]+[fp]*2+[ctypes.c_int]*3;lib.score.argtypes=[ctypes.c_int]*2+[ip]*2+[dp]+[ip]*2+[dp]+[fp]*4;return lib

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');a=ap.parse_args();lib=library();u,i,t,qu,qi,qt,U,I=queries(a.full);assert (qu>=0).all() and (qi>=0).all();valid=(t>=0)&(t<4102444800);order=np.lexsort((np.arange(len(u)),t,u));order=order[valid[order]];uu=u[order].astype('i');ii=i[order].astype('i');tt=t[order].astype('d');counts=np.bincount(ii,minlength=I);keep=counts[ii]>=2;uu,ii,tt=uu[keep],ii[keep],tt[keep];lens=np.bincount(uu,minlength=U);source=np.bincount(ii[lens[uu]>1],minlength=I);active=np.flatnonzero(source>0);prob=source[active]**.75;prob/=prob.sum();rng=np.random.RandomState(733);pool=rng.choice(active,size=262144,p=prob).astype('i');ptr=np.searchsorted(uu,np.arange(U+1)).astype('i');k=32;E=rng.uniform(-.5/k,.5/k,(I,k)).astype('f');O=np.zeros_like(E);features={};old=0;start=time.time();qu=qu.astype('i');qi=qi.astype('i');qt=qt.astype('d')
 for epoch in [5,12]:
  lib.train(len(ii),k,5,3,uu,ii,pool,len(pool),E,O,old,epoch-old,733);old=epoch;assert np.isfinite(E).all() and np.isfinite(O).all();E[source==0]=0;O[source==0]=0;V=E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-8);value=np.zeros((len(qu),12),'f');lib.score(len(qu),k,ptr,ii,tt,qu,qi,qt,E,O,V,value);assert np.isfinite(value).all();features.update({f'item2vec_e{epoch}_{name}':value[:,col].copy() for col,name in enumerate(NAMES)});print('ITEM2VEC',a.full,epoch,'seconds',round(time.time()-start),flush=True)
 stage='full' if a.full else 'validation';np.savez_compressed(OUT/(stage+'_features.npz'),**features);np.savez_compressed(OUT/(stage+'_embeddings.npz'),input=E,output=O,source_count=source);report={'stage':stage,'factors':k,'window':5,'negative_samples':3,'negative_distribution':'eligible interaction frequency power .75, 262144-entry sampled table','min_count':2,'epochs':[5,12],'seed':733,'training_interactions_in_sequence':len(ii),'active_books':len(active),'features':list(features),'queries':len(qu),'ratings_used':False,'zero_rating_events_retained':True,'context':'only fitted training rows; query book excluded from pooled history; prior and following KNOWN training books relative to supplied timestamp; invalid dates omitted from ordered sequence','input_hashes':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['tasks_3_4_train.csv','task_4_test.csv']}};(OUT/(stage+'_report.json')).write_text(json.dumps(report,indent=2));print('EXPORTED',stage,flush=True)
if __name__=='__main__':main()
