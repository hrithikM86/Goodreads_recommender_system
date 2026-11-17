"""Task3 error slices from original final meta holdout; no fitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np,pandas as pd
from task3_optimize import dataset
W=Path('/workspace/work');DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026';OUT=W/'task3_error_audit';OUT.mkdir(exist_ok=True)
p,tr,q,*_=dataset(DATA,False);n=int(.95*len(p))-int(.9*len(p));ev=np.load(W/'task3_scaled_temporal/evaluation.npz');y=ev['y'];pred=ev['blend'];assert np.array_equal(y,q.rating.to_numpy()[n:]);q=q.iloc[n:].copy();q['error2']=(y-pred)**2;q['residual']=y-pred
rows=[]
def summarize(kind,labels):
 for k in pd.unique(labels):
  ix=np.asarray(labels==k);count=int(ix.sum())
  rows.append({'slice':kind,'group':str(k),'rows':count,'MSE':float(q.error2.to_numpy()[ix].mean()),'mean_residual':float(q.residual.to_numpy()[ix].mean()),'share_of_total_squared_error':float(q.error2.to_numpy()[ix].sum()/q.error2.sum())})
for key in ['user','item']:
 count=q[key].map(tr.groupby(key).size()).fillna(0).to_numpy();summarize(key+'_support',pd.cut(count,[-1,10,30,100,300,np.inf]).astype(str))
t=q.timestamp.to_numpy();dates=pd.to_datetime(t,unit='s',errors='coerce');summarize('calendar_year',np.asarray(dates.year.astype(str)));summarize('true_rating_diagnostic_only',y.astype(int));summarize('predicted_rating',pd.cut(pred,[0,2,3,3.5,4,4.5,5.01]).astype(str))
for key in ['user','item']:
 std=q[key].map(tr.groupby(key).rating.std()).fillna(0).to_numpy();summarize(key+'_rating_std',pd.cut(std,[-1,.5,1,1.5,np.inf]).astype(str))
r={'reference_public_MSE':.6804250167145823,'local_MSE':float(q.error2.mean()),'evaluation_rows':len(q),'slices':rows,'note':'Slices describe reused historical evaluation only. True-rating buckets are diagnostics, never inference features. Second validation will be user-disjoint within existing out-of-sample meta rows; base90% training remains unchanged.'};(OUT/'slices.json').write_text(json.dumps(r,indent=2))
for a in rows:
 if a['slice'] in ['user_support','item_support','true_rating_diagnostic_only']:print(a)
