"""Temporal bias extension: item_bias(t) * (1 + user_scale(t)).
Exact regularized coordinate updates, conditional on each parameter block.
Existing additive terms and multiplicative scale terms are jointly refitted.
"""
import copy
import numpy as np
class ScaledTemporalBias:
 def __init__(self,base,epochs=20):
  self.base=copy.deepcopy(base);self.epochs=epochs
  # Scale deviations shrink toward0, hence total scale shrinks toward1.
  self.specs=[('user',20.),('drift',40.),('user90',50.),('userday',100.)]
 def fit(self,u,i,t,y):
  u=np.asarray(u,dtype=np.int64);i=np.asarray(i,dtype=np.int64)
  keys=self.base.keys(u,i,t);terms=[]
  for original in self.base.terms:
   name=original['name'];lab,val=keys[name];code=original['index'].get_indexer(lab);assert (code>=0).all()
   terms.append(dict(original,code=code,value=val,item=name.startswith('item')))
  self.scales=[]
  for name,pen in self.specs:
   original=next(a for a in terms if a['name']==('user' if name=='drift' else name))
   # Drift shares the user index, with a different feature value.
   lab,val=keys[name];code=original['index'].get_indexer(lab);assert (code>=0).all()
   self.scales.append({'name':name,'penalty':pen,'index':original['index'],'code':code,'value':val,'coef':np.zeros(len(original['index']))})
  item=sum(a['coef'][a['code']]*a['value'] for a in terms if a['item']);scale=np.zeros(len(y));pred=self.base.mu+sum(a['coef'][a['code']]*a['value'] for a in terms);res=y-pred;self.objectives=[]
  for epoch in range(self.epochs):
   for a in terms:
    code,val=a['code'],a['value'];feature=val*(1+scale) if a['item'] else val
    old=a['coef'];res+=old[code]*feature
    new=np.bincount(code,weights=res*feature,minlength=len(old))/(np.bincount(code,weights=feature*feature,minlength=len(old))+a['penalty'])
    a['coef']=new;res-=new[code]*feature
    if a['item']:item+=(new-old)[code]*val
   for a in self.scales:
    code,val=a['code'],a['value'];feature=val*item;old=a['coef'];res+=old[code]*feature
    new=np.bincount(code,weights=res*feature,minlength=len(old))/(np.bincount(code,weights=feature*feature,minlength=len(old))+a['penalty'])
    a['coef']=new;res-=new[code]*feature;scale+=(new-old)[code]*val
   objective=float(res@res+sum(a['penalty']*float(a['coef']@a['coef']) for a in terms+self.scales));self.objectives.append(objective)
  assert np.all(np.diff(self.objectives)<=1e-6*np.maximum(1,np.asarray(self.objectives[:-1])))
  for old,new in zip(self.base.terms,terms):old['coef']=new['coef']
  for a in self.scales:del a['code'],a['value']
  self.training_prediction=y-res
  return self
 def parts(self,u,i,t):
  u=np.asarray(u,dtype=np.int64);i=np.asarray(i,dtype=np.int64)
  keys=self.base.keys(u,i,t);item=np.zeros(len(u));add=np.full(len(u),self.base.mu);scale=np.zeros(len(u))
  for a in self.base.terms:
   lab,val=keys[a['name']];ix=a['index'].get_indexer(lab);ok=ix>=0;v=np.zeros(len(u));v[ok]=a['coef'][ix[ok]]*val[ok];add+=v
   if a['name'].startswith('item'):item+=v
  for a in self.scales:
   lab,val=keys[a['name']];ix=a['index'].get_indexer(lab);ok=ix>=0;scale[ok]+=a['coef'][ix[ok]]*val[ok]
  return add+item*scale,1+scale,item
 def save(self,path):
  # Indexes/centers come from the immutable accepted base checkpoint.
  np.savez_compressed(path,**{'base_'+str(j):a['coef'].astype('f') for j,a in enumerate(self.base.terms)},**{'scale_'+str(j):a['coef'].astype('f') for j,a in enumerate(self.scales)})
 def restore(self,path):
  z=np.load(path)
  for j,a in enumerate(self.base.terms):a['coef']=z['base_'+str(j)].astype(float)
  self.scales=[]
  for j,(name,pen) in enumerate(self.specs):
   index=next(a['index'] for a in self.base.terms if a['name']==('user' if name=='drift' else name));self.scales.append(dict(name=name,penalty=pen,index=index,coef=z['scale_'+str(j)].astype(float)))
