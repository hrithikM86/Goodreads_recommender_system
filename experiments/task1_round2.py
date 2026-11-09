from pathlib import Path
ARTIFACTS=Path(__file__).resolve().parent/"artifacts"
ARTIFACTS.mkdir(exist_ok=True)
import os,sys,time,json
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('SURPRISE_DATA_FOLDER', str(Path(__file__).resolve().parent/'artifacts'/'surprise_data'))
# Install scikit-surprise in the active Python environment.
import numpy as np,pandas as pd
from surprise import Dataset,Reader,SVD,BaselineOnly
ROOT=str(Path(__file__).resolve().parents[1]/'Assignment 1/assignment1_2026')+'/'
p=pd.read_csv(ROOT+'tasks_1_2_train.csv',dtype={'user':str,'item':str})
p=p[p.rating>0].reset_index(drop=True)
order=np.random.RandomState(42).permutation(len(p));n=len(p);a=int(n*.9);b=int(n*.95)
train=p.iloc[order[:a]];valid=p.iloc[order[a:b]];test=p.iloc[order[b:]]
reader=Reader(rating_scale=(1,5));data=Dataset.load_from_df(train[['user','item','rating']],reader).build_full_trainset()
vr=list(valid[['user','item','rating']].itertuples(index=False,name=None));tr=list(test[['user','item','rating']].itertuples(index=False,name=None))
vy=valid.rating.to_numpy();ty=test.rating.to_numpy()
np.savez_compressed(str(ARTIFACTS / 'task1_split.npz'),order=order,validation_y=vy,evaluation_y=ty)
print('Filtered rows',n,'split',len(train),len(valid),len(test),flush=True)
results={};vp={};tp={}
trials=[('svd32_reg.02_100',SVD(n_factors=32,n_epochs=100,reg_all=.02,random_state=42)),('svd64_reg.02_100',SVD(n_factors=64,n_epochs=100,reg_all=.02,random_state=42)),('svd128_reg.05_100',SVD(n_factors=128,n_epochs=100,reg_all=.05,random_state=42))]
for name,model in trials:
 start=time.time();model.fit(data)
 vp[name]=np.array([model.predict(u,i).est for u,i,r in vr]);tp[name]=np.array([model.predict(u,i).est for u,i,r in tr])
 results[name]={'selection_MSE':float(np.mean((vp[name]-vy)**2)),'evaluation_MSE':float(np.mean((tp[name]-ty)**2)),'seconds':time.time()-start}
 print(name,results[name],flush=True)
 with open(str(ARTIFACTS / 'task1_round2_results.json'),'w') as f:json.dump(results,f,indent=2)
 np.savez_compressed(str(ARTIFACTS / 'task1_round2_predictions.npz'),**{'v_'+k:v for k,v in vp.items()},**{'t_'+k:v for k,v in tp.items()})
