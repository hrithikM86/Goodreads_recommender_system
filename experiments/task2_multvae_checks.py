"""Independent saved-VAE subset replay and final CSV checks; no training."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix
from task2_multvae_numpy import MultVAE,gradient_check
W=Path('/workspace/work');OUT=W/'task2_multvae';DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026'
r=json.loads((OUT/'results.json').read_text());p=pd.read_csv(DATA/'tasks_1_2_train.csv');u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);X=coo_matrix((np.ones(len(p),dtype='f'),(u,i)),shape=(len(users),len(items))).tocsr();X.data[:]=1
q=pd.read_csv(DATA/'task_2_test.csv');tu=pd.Index(users).get_indexer(q.user);qi=np.r_[pd.Index(items).get_indexer(q.item1),pd.Index(items).get_indexer(q.item2)];qu=np.r_[tu,tu];ix=np.random.RandomState(98).choice(len(qu),128,replace=False)
m=MultVAE(X.shape[1]);m.restore(OUT/'production_vae.npz');raw,lp=m.scores(X,qu[ix],qi[ix]);z=np.load(OUT/'test_scores.npz');error=max(float(abs(raw-z['raw'][ix]).max()),float(abs(lp-z['logprob'][ix]).max()));assert error<2e-4,error
for label,info in r['outputs'].items():
 path=Path(info['path']);frame=pd.read_csv(path);pd.testing.assert_frame_equal(frame.iloc[:,:-1],q.iloc[:,:-1]);assert len(frame)==20000 and ((frame.which==frame.item1)|(frame.which==frame.item2)).all();assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
accepted=Path('/workspace/output/task2_xgboost_blend/task_2_predictions.csv');assert hashlib.sha256(accepted.read_bytes()).hexdigest()=='9dd469b074af077ca895b1e1ed53b3b2fae456b3a98c9b8dde603d0169816880'
a={'gradient_check_max_error':gradient_check(),'saved_VAE_subset_replay_max_error':error,'replayed_candidate_scores':128,'prediction_rows_per_file':20000,'input_columns_preserved':True,'accepted_CSV_unchanged':True}
(OUT/'checks.json').write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2))
