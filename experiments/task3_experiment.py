from task3_paths import DATA,ART
import os,sys,json,pickle,time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('SURPRISE_DATA_FOLDER',str(ART/'surprise_data'))
# scikit-surprise is installed in the active environment.
import numpy as np,pandas as pd
from surprise import SVD,Dataset,Reader
from task3_temporal_model import TemporalBias
root=str(DATA)+'/'
p=pd.read_csv(root+'tasks_3_4_train.csv');p=p[p.rating>0].reset_index(drop=True)
u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);t=p.timestamp.to_numpy(dtype=float);y=p.rating.to_numpy(dtype=float)
order=np.random.RandomState(42).permutation(len(p));a=int(.9*len(p));b=int(.95*len(p));tr=order[:a];va=order[a:b];te=order[b:]
np.savez_compressed(str(ART/'split.npz'),order=order,users=u,items=i,time=t,y=y)
results={};out={};best=float('inf')
configs=[('static',{'user':5,'item':10}),('drift_item',{'user':5,'item':10,'month':100,'drift':20,'item90':30}),('temporal180',{'user':5,'item':10,'month':100,'drift':20,'user180':10,'item180':30,'userday':10}),('temporal90',{'user':5,'item':10,'month':100,'drift':20,'user90':10,'item90':30,'userday':5}),('temporal90_strong',{'user':5,'item':10,'month':100,'drift':40,'user90':30,'item90':60,'userday':20})]
for name,config in configs:
 start=time.time();m=TemporalBias(config).fit(u[tr],i[tr],t[tr],y[tr],len(users));v=np.clip(m.predict(u[va],i[va],t[va]),1,5);q=np.clip(m.predict(u[te],i[te],t[te]),1,5)
 results[name]={'selection_MSE':float(np.mean((v-y[va])**2)),'evaluation_MSE':float(np.mean((q-y[te])**2)),'config':config,'seconds':time.time()-start};out['v_'+name]=v;out['t_'+name]=q
 print(name,results[name],flush=True)
 if results[name]['selection_MSE']<best:
  best=results[name]['selection_MSE']
  with open(str(ART/'best_bias.pkl'),'wb') as f:pickle.dump(m,f)
 with open(str(ART/'results.json'),'w') as f:json.dump(results,f,indent=2)
 np.savez_compressed(str(ART/'predictions.npz'),**out)
data=Dataset.load_from_df(p.iloc[tr][['user','item','rating']],Reader(rating_scale=(1,5))).build_full_trainset()
for factors in [16,32]:
 name='svd'+str(factors);m=SVD(n_factors=factors,n_epochs=50,reg_all=.1,random_state=42);m.fit(data)
 v=np.array([m.predict(uu,ii).est for uu,ii in p.iloc[va][['user','item']].itertuples(index=False,name=None)]);q=np.array([m.predict(uu,ii).est for uu,ii in p.iloc[te][['user','item']].itertuples(index=False,name=None)])
 results[name]={'selection_MSE':float(np.mean((v-y[va])**2)),'evaluation_MSE':float(np.mean((q-y[te])**2))};out['v_'+name]=v;out['t_'+name]=q
 print(name,results[name],flush=True)
 if factors==16:
  # Reproduce the baseline's post-fit global-month correction on positive ratings.
  trainpred=np.array([m.predict(uu,ii).est for uu,ii in p.iloc[tr][['user','item']].itertuples(index=False,name=None)])
  dates=pd.to_datetime(p.timestamp,unit='s',errors='coerce',utc=True);months=dates.dt.year.fillna(-1)*12+dates.dt.month.fillna(0)
  frame=pd.DataFrame({'key':months.iloc[tr].to_numpy(),'res':y[tr]-trainpred});agg=frame.groupby('key').res.agg(['sum','count']);corr=agg['sum']/(agg['count']+100)
  vv=np.clip(v+months.iloc[va].map(corr).fillna(0).to_numpy(),1,5);qq=np.clip(q+months.iloc[te].map(corr).fillna(0).to_numpy(),1,5)
  results['svd_global_month']={'selection_MSE':float(np.mean((vv-y[va])**2)),'evaluation_MSE':float(np.mean((qq-y[te])**2))};out['v_svd_global_month']=vv;out['t_svd_global_month']=qq
  print('svd_global_month',results['svd_global_month'],flush=True)
 with open(str(ART/'results.json'),'w') as f:json.dump(results,f,indent=2)
 np.savez_compressed(str(ART/'predictions.npz'),**out)
