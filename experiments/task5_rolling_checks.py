"""Independent final-output and rolling-target exclusion checks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,pickle,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from task5_rolling import OUT,DATA,W,args,cohort,SEEDS
from task5_preference_round3 import load
from task5_round3 import nn_margin
h,ctx,qu,qi,U,I,_=cohort();N=len(ctx);assert N==3*U
assert all(qi[j] not in ctx[j] for j in range(N))
assert np.array_equal(qu[:N],np.tile(np.arange(U),3))
pop=np.bincount(np.concatenate(h),minlength=I);assert np.all(pop[[x[-1] for x in ctx]]>0)
d,s,parts,extra=load(args(),True);assert d.shape==(86786,260)
rng=np.random.RandomState(33);ix=rng.choice(U,512,replace=False);p=0
for key,width in [('sequence',224),('preference',260)]:
 m,scale=pickle.loads((OUT/(key+'_production.pkl')).read_bytes());v=nn_margin(m,scale,d[:,:width],s[:,:width]);assert np.isfinite(v).all();assert np.allclose(v[ix],-nn_margin(m,scale,-d[ix,:width],s[ix,:width]),atol=1e-6);p+=.5*v
accepted_replay=0
for path,width in [(W/'task5_round3/nn_evaluation.pkl',224),(W/'task5_round3/preference/small_evaluation.pkl',260)]:
 m,scale=pickle.loads(path.read_bytes());accepted_replay+=.5*nn_margin(m,scale,d[ix,:width],s[ix,:width])
assert np.allclose(accepted_replay,np.load(W/'task5_round3/preference/test_predictions.npz')['primary'][ix],rtol=1e-5,atol=1e-6)
saved=np.load(OUT/'test.npz')['margin'];assert np.allclose(p,saved,rtol=0,atol=1e-12)
q=pd.read_csv(DATA/'task_5_test.csv');f=pd.read_csv('/workspace/output/task5_rolling_neural/task_5_predictions.csv');assert f.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert np.array_equal(f.which,np.where(p>=0,q.item1,q.item2));assert len(f)==U
r={'rows':U,'own_target_in_query_history':0,'last_context_cold_rate':0,'production_model_margin_replay':True,'candidate_swap_antisymmetry':True,'test_row_order_and_ids_unchanged':True,'choices_valid':True,'evaluation_negative_seeds':SEEDS}
(OUT/'checks.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
