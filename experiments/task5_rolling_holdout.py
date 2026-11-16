"""Reader-held-out diagnostic for adding refreshed final-event meta-training pairs.
Fit selection examples for all readers (past labels), final examples only for80%.
Evaluate the remaining readers' final examples across three negative seeds.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import pickle,json,gc
import numpy as np
from task5_rolling import OUT,W,args,refreshed,SEEDS
from task5_preference_round3 import load
from task5_round3 import nn_fit,nn_margin,accuracy
ed,es,names=refreshed();U=len(ed)//3;split=np.random.RandomState(17).permutation(U);tr=split[:int(.8*U)];va=split[int(.8*U):];draw=np.random.RandomState(418).randint(3,size=U)*U+np.arange(U);d,s,parts,extra=load(args());td=np.r_[d[:U],ed[draw[tr]]];ts=np.r_[s[:U],es[draw[tr]]];del d,s,parts;gc.collect();evrows=np.r_[va,va+U,va+2*U];new=0;baseline=0
for key,width,seed,epochs,path in [('sequence',224,71,16,W/'task5_round3/nn_evaluation.pkl'),('preference',260,73,8,W/'task5_round3/preference/small_evaluation.pkl')]:
 frozen=pickle.loads(path.read_bytes());baseline+=.5*nn_margin(*frozen,ed[evrows,:width],es[evrows,:width]);dest=OUT/(key+'_holdout.pkl')
 if dest.exists():m=pickle.loads(dest.read_bytes())
 else:m,_=nn_fit(td[:,:width],ts[:,:width],epochs,seed=seed,hidden=(64,32),alpha=.1);dest.write_bytes(pickle.dumps(m,protocol=5))
 new+=.5*nn_margin(*m,ed[evrows,:width],es[evrows,:width]);print('HOLDOUT_NETWORK',key,flush=True)
size=len(va);r={'heldout_readers':size,'final_training_reader_overlap':int(len(np.intersect1d(tr,va))),'training_pairs':len(td),'accepted_frozen_accuracy':{str(seed):accuracy(baseline[j*size:(j+1)*size]) for j,seed in enumerate(SEEDS)},'refit_accuracy':{str(seed):accuracy(new[j*size:(j+1)*size]) for j,seed in enumerate(SEEDS)},'protocol':'All past selection labels plus80% reader final labels for fitting; remaining20% reader final labels only evaluation. All component models exclude all final targets. Fixed accepted neural settings. Diagnostic historical split reused in prior experiments, not an untouched hidden test.'}
# Pool seed-level changes within reader before uncertainty calculation.
corr=lambda x:(x>0).astype(float)+.5*(x==0)
delta=(corr(new)-corr(baseline)).reshape(3,size).mean(axis=0);se=delta.std(ddof=1)/np.sqrt(size);r.update(mean_accuracy_change=float(delta.mean()),approx_reader_clustered_95pct_delta_interval=[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)]);(OUT/'holdout.json').write_text(json.dumps(r,indent=2));np.savez_compressed(OUT/'holdout.npz',readers=va,accepted=baseline,refit=new);print('HOLDOUT_RESULT',json.dumps(r),flush=True)
