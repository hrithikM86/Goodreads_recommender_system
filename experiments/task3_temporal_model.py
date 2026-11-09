"""Regularized temporal biases inspired by the course temporal-model slides."""
import numpy as np,pandas as pd

class TemporalBias:
    def __init__(self, config, epochs=25):
        self.config=config;self.epochs=epochs

    def keys(self,u,i,t):
        good=(t>=0)&(t<=4102444800)
        days=np.floor(np.clip(t,0,4102444800)/86400).astype(np.int64)
        days=np.where(good,days,-1)
        center=self.mean_time[np.maximum(u,0)]
        delta=(t-center)/(86400*365.25)
        drift=np.where(good,np.clip(np.sign(delta)*abs(delta)**.4,-5,5),0)
        stride=1000000000
        return {
            'user':(u,np.ones(len(u))), 'item':(i,np.ones(len(u))),
            'month':(np.where(good,days//30,-1),np.ones(len(u))),
            'drift':(u,drift),
            'drift_pos':(u,np.maximum(drift,0)),
            'drift_neg':(u,np.minimum(drift,0)),
            'user180':((u+1)*stride+np.where(good,days//180,-1),np.ones(len(u))),
            'user90':((u+1)*stride+np.where(good,days//90,-1),np.ones(len(u))),
            'item180':((i+1)*stride+np.where(good,days//180,-1),np.ones(len(u))),
            'item90':((i+1)*stride+np.where(good,days//90,-1),np.ones(len(u))),
            'userday':((u+1)*stride+days,np.ones(len(u))),
        }

    def fit(self,u,i,t,y,n_users):
        self.mu=float(np.mean(y));good=(t>=0)&(t<=4102444800)
        count=np.bincount(u[good],minlength=n_users);total=np.bincount(u[good],weights=t[good],minlength=n_users)
        self.mean_time=np.divide(total,count,out=np.full(n_users,float(np.mean(t[good]))),where=count>0)
        keys=self.keys(u,i,t);self.terms=[];res=y.astype(float)-self.mu
        for name,penalty in self.config.items():
            labels,value=keys[name];code,index=pd.factorize(labels,sort=False)
            self.terms.append({'name':name,'penalty':penalty,'index':pd.Index(index),'code':code,'value':value,'coef':np.zeros(len(index)),'den':np.bincount(code,weights=value**2)+penalty})
        for epoch in range(self.epochs):
            for term in self.terms:
                code=term['code'];value=term['value'];res+=term['coef'][code]*value
                term['coef']=np.bincount(code,weights=res*value,minlength=len(term['index']))/term['den']
                res-=term['coef'][code]*value
        self.training_prediction=y-res
        for term in self.terms:
            del term['code'],term['value'],term['den']
        return self

    def predict(self,u,i,t):
        keys=self.keys(u,i,t);out=np.full(len(u),self.mu)
        for term in self.terms:
            labels,value=keys[term['name']];idx=term['index'].get_indexer(labels);good=idx>=0
            out[good]+=term['coef'][idx[good]]*value[good]
        return out
