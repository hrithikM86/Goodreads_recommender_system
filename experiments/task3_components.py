import os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('SURPRISE_DATA_FOLDER',str(Path(__file__).resolve().parent/'artifacts'/'surprise_data'))
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix
from surprise import SVD,Dataset,Reader
from task3_temporal_model import TemporalBias


def extra_features(train,query):
    cols=[query.timestamp.to_numpy()/86400/365.25]
    for key in ['user','item']:
        for field in ['rating','timestamp']:
            agg=train.groupby(key)[field].agg(['count','mean','std','min','max'])
            for stat in agg:
                value=query[key].map(agg[stat]).fillna(0).to_numpy()
                if field=='timestamp':value=value/86400/365.25
                cols.append(value)
            if field=='timestamp':
                mean=query[key].map(agg['mean']).fillna(train.timestamp.mean()).to_numpy()
                cols.append((query.timestamp.to_numpy()-mean)/86400/365.25)
    return np.nan_to_num(np.column_stack(cols),nan=0,posinf=0,neginf=0)


def generate(train,query,configs,user_index=None,item_index=None):
    users=pd.Index(pd.unique(train.user)) if user_index is None else pd.Index(user_index)
    items=pd.Index(pd.unique(train.item)) if item_index is None else pd.Index(item_index)
    u=users.get_indexer(train.user);i=items.get_indexer(train.item)
    qu=pd.Index(users).get_indexer(query.user);qi=pd.Index(items).get_indexer(query.item)
    t=train.timestamp.to_numpy(dtype=float);qt=query.timestamp.to_numpy(dtype=float);y=train.rating.to_numpy(dtype=float)
    U=len(users);I=len(items);out={};saved={}
    for name,config in configs.items():
        m=TemporalBias(config).fit(u,i,t,y,U);out[name]=np.clip(m.predict(qu,qi,qt),1,5)
        if name=='bias_static':saved['static']=m
        if name=='bias_temporal90_strong':saved['temporal']=m
    data=Dataset.load_from_df(train[['user','item','rating']],Reader(rating_scale=(1,5))).build_full_trainset()
    for factors in [16,32]:
        print('SVD',factors,flush=True)
        m=SVD(n_factors=factors,n_epochs=50,reg_all=.1,random_state=42);m.fit(data)
        pred=np.array([m.predict(uu,ii).est for uu,ii in query[['user','item']].itertuples(index=False,name=None)])
        out['bias_svd'+str(factors)]=pred
        if factors==16:
            trainpred=np.array([m.predict(uu,ii).est for uu,ii in train[['user','item']].itertuples(index=False,name=None)])
            def monthkeys(series):
                d=pd.to_datetime(series,unit='s',errors='coerce',utc=True)
                return d.dt.year.fillna(-1)*12+d.dt.month.fillna(0)
            frame=pd.DataFrame({'key':monthkeys(train.timestamp).to_numpy(),'res':y-trainpred});agg=frame.groupby('key').res.agg(['sum','count']);correction=agg['sum']/(agg['count']+100)
            out['bias_svd_global_month']=np.clip(pred+monthkeys(query.timestamp).map(correction).fillna(0).to_numpy(),1,5)
    X=coo_matrix((np.ones(len(y),dtype=np.float32),(u,i)),shape=(U,I)).tocsr();G=X.T.tocsr();pop=np.asarray(X.sum(axis=0)).ravel()
    Times=coo_matrix((t,(u,i)),shape=(U,I)).tocsr()
    R={name:coo_matrix((y-m.training_prediction,(u,i)),shape=(U,I)).tocsr() for name,m in saved.items()}
    bases={name:m.predict(qu,qi,qt) for name,m in saved.items()}
    settings=[(20,1,0),(20,2,0),(20,1,365),(20,2,365),(20,1,90),(20,2,90)]
    for name,base in bases.items():
        for k,power,tau in settings:out['neighbor_'+name+'_'+str(k)+'_'+str(power)+'_'+str(tau)]=base.copy()
    groups={}
    for q,key in enumerate(qi):
        if key>=0 and qu[q]>=0:groups.setdefault(key,[]).append(q)
    keys=np.array(list(groups))
    for offset in range(0,len(keys),64):
        batch=keys[offset:offset+64];co=(G[batch]@X).toarray()
        for row,key in enumerate(batch):
            cosine=co[row]/np.sqrt(np.maximum(pop[key]*pop,1));cosine[key]=0
            for q in groups[key]:
                lo,hi=X.indptr[qu[q]:qu[q]+2];sim=cosine[X.indices[lo:hi]];distance=abs(Times.data[lo:hi]-qt[q])/86400
                for k,power,tau in settings:
                    w=sim**power
                    if tau:w=w*np.exp(-distance/tau)
                    idx=np.argpartition(w,-k)[-k:] if len(w)>k else np.arange(len(w));weights=w[idx];den=weights.sum()+1e-12
                    for name in R:out['neighbor_'+name+'_'+str(k)+'_'+str(power)+'_'+str(tau)][q]+=(weights@R[name].data[lo:hi][idx])/den
    settings=[(tau,reg) for tau in [7,30,90,365] for reg in [1,5]]
    for name,base in bases.items():
        for tau,reg in settings:out['kernel_'+name+'_'+str(tau)+'_'+str(reg)]=base.copy()
    trainorder=np.argsort(u);tu=u[trainorder];starts=np.searchsorted(tu,np.arange(U+1));groups={}
    for q,user in enumerate(qu):
        if user>=0:groups.setdefault(user,[]).append(q)
    for user,qs in groups.items():
        rows=trainorder[starts[user]:starts[user+1]]
        if not len(rows):continue
        distance=abs(qt[qs,None]-t[rows][None,:])/86400
        for tau,reg in settings:
            w=np.exp(-distance/tau);den=w.sum(axis=1)+reg
            for name,m in saved.items():out['kernel_'+name+'_'+str(tau)+'_'+str(reg)][qs]+=w@(y[rows]-m.training_prediction[rows])/den
    return {name:np.clip(pred,1,5) for name,pred in out.items()}
