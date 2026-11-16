"""Optional scheduling helper: refit sequence stack while stronger BPR is running.
Produces the same fixed sequence models as the complete runner/holdout diagnostic.
Run only after all 224 sequence score columns have finished refreshing.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import pickle,gc
import numpy as np
from task5_rolling import OUT,args,refreshed
from task5_preference_round3 import load
from task5_round3 import nn_fit
ed,es,names=refreshed(224);U=len(ed)//3;d,s,parts,extra=load(args());sd,ss=d[:U,:224].copy(),s[:U,:224].copy();del d,s,parts;gc.collect();draw=np.random.RandomState(418).randint(3,size=U)*U+np.arange(U);split=np.random.RandomState(17).permutation(U);tr=split[:int(.8*U)]
for label,rows in [('production',np.arange(U)),('holdout',tr)]:
 path=OUT/('sequence_'+label+'.pkl')
 if path.exists():continue
 td=np.r_[sd,ed[draw[rows]]];ts=np.r_[ss,es[draw[rows]]];m,_=nn_fit(td,ts,16,seed=71,hidden=(64,32),alpha=.1);path.write_bytes(pickle.dumps(m,protocol=5));del td,ts;gc.collect();print('SEQUENCE_READY',label,flush=True)
