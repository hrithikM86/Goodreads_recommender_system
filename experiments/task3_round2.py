"""Joint temporal SVD++ and direct rating learners; zero ratings excluded."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,ctypes,json,pickle,subprocess,sys,time,hashlib
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix,save_npz
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostRegressor
from task3_optimize import dataset,clip,mse,train_tree,baseline_validation
from task3_temporal_model import TemporalBias


def log(*a):print(*a,flush=True)
def codes(train,query):
 c,index=pd.factorize(train);return np.ascontiguousarray(c,np.int32),pd.Index(index).get_indexer(query),np.asarray(index)
def time_arrays(u,i,t,qu,qi,qt,U):
 good=(t>=0)&(t<=4102444800);qgood=(qt>=0)&(qt<=4102444800);day=np.where(good,np.floor(np.clip(t,0,4102444800)/86400),-1).astype(np.int64);qd=np.where(qgood,np.floor(np.clip(qt,0,4102444800)/86400),-1).astype(np.int64)
 center=np.bincount(u[good],weights=t[good],minlength=U)/np.maximum(np.bincount(u[good],minlength=U),1);fallback=np.mean(t[good]);center[np.bincount(u[good],minlength=U)==0]=fallback
 def dev(uu,tt,gg):
  d=(tt-center[np.maximum(uu,0)])/(86400*365.25);return np.ascontiguousarray(np.where(gg&(uu>=0),np.clip(np.sign(d)*np.abs(d)**.4,-5,5),0),np.float32)
 fields=[((u.astype(np.int64)+1)*1000000000+day,(qu.astype(np.int64)+1)*1000000000+qd),((i.astype(np.int64)+1)*1000000000+np.where(good,day//90,-1),(qi.astype(np.int64)+1)*1000000000+np.where(qgood,qd//90,-1)),((u.astype(np.int64)+1)*1000000000+np.where(good,day//90,-1),(qu.astype(np.int64)+1)*1000000000+np.where(qgood,qd//90,-1)),(np.where(good,day//30,-1),np.where(qgood,qd//30,-1))]
 return [codes(a,b) for a,b in fields],dev(u,t,good),dev(qu,qt,qgood),center

def joint_worker(data,out,full):
 # Existing completed configurations can be reused when expanding this fixed grid.
 p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);U,I=len(users),len(items);qt=q.timestamp.to_numpy(float);order=np.argsort(u,kind='stable');u=u[order];i=np.ascontiguousarray(i[order],np.int32);t=t[order];y=np.ascontiguousarray(y[order],np.float32);count=np.bincount(u,minlength=U);ptr=np.ascontiguousarray(np.r_[0,np.cumsum(count)],np.int32);ic=np.ascontiguousarray(np.bincount(i,minlength=I),np.int32)
 fields,dv,qd,center=time_arrays(u,i,t,qu,qi,qt,U);cs=[c for c,_,_ in fields];qc=[c for _,c,_ in fields];counts=[np.ascontiguousarray(np.bincount(c),np.int32) for c in cs];mu=float(np.mean(y));H=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=(U,I)).tocsr();so=out/'joint.so'
 if not so.exists():subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task3_joint_kernel.cpp')),'-o',str(so)],check=True)
 lib=ctypes.CDLL(str(so.resolve()));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.joint_train.argtypes=[ctypes.c_int,ctypes.c_int]+[ip]*6+[fp]*2+[ip]*5+[ctypes.c_float]+[fp]*11+[ctypes.c_int]*3+[ctypes.c_float]*5;lib.joint_train.restype=None;target=out/'joint_features.npz';features=dict(np.load(target)) if target.exists() else {}
 for name,k,reg,rday,rperiod,lr,seed,epochs in [('joint24',24,.025,10.,20.,.007,42,[40,80]),('joint48',48,.035,10.,30.,.006,17,[40,80]),('joint24weak',24,.01,30.,10.,.005,23,[40,80]),('joint24strong',24,.12,20.,40.,.007,31,[15,30,50]),('joint48strong',48,.08,30.,40.,.005,47,[15,30,50]),('joint16strong',16,.2,30.,60.,.008,11,[15,30,50])]:
  if all(name+'_e'+str(epoch) in features for epoch in epochs):continue
  rng=np.random.RandomState(seed);P=rng.normal(0,.1,(U,k)).astype(np.float32);Q=rng.normal(0,.1,(I,k)).astype(np.float32);Y=np.zeros((I,k),np.float32);A=np.zeros((U,k),np.float32);bu=np.zeros(U,np.float32);bi=np.zeros(I,np.float32);du,bt,ut,mt=[np.zeros(len(c),np.float32) for c in counts];alpha=np.zeros(U,np.float32);last=0
  for epoch in epochs:
   lib.joint_train(U,k,ptr,i,*cs,dv,y,ic,*counts,mu,P,Q,Y,A,bu,bi,du,bt,ut,mt,alpha,last,epoch-last,seed,lr,reg,rday,rperiod,10.);last=epoch
   assert all(np.isfinite(v).all() for v in [P,Q,Y,A,bu,bi,du,bt,ut,mt,alpha])
   effective=P+(H@Y)/np.sqrt(np.maximum(count,1))[:,None];pred=np.full(len(q),mu);vu=(qu>=0)&(count[np.maximum(qu,0)]>0);vi=(qi>=0)&(ic[np.maximum(qi,0)]>0);ok=vu&vi;pred[vu]+=bu[qu[vu]]+alpha[qu[vu]]*qd[vu];pred[vi]+=bi[qi[vi]]
   pred[ok]+=np.einsum('ij,ij->i',effective[qu[ok]]+qd[ok,None]*A[qu[ok]],Q[qi[ok]])
   for c,coeff in zip(qc,[du,bt,ut,mt]):valid=c>=0;pred[valid]+=coeff[c[valid]]
   features[name+'_e'+str(epoch)]=clip(pred);np.savez_compressed(out/(name+'_e'+str(epoch)+'.npz'),P=P,Q=Q,Y=Y,A=A,effective=effective,bu=bu,bi=bi,du=du,bt=bt,ut=ut,mt=mt,alpha=alpha,mu=mu,center=center,uc=count,ic=ic,**{'keys'+str(j):f[2] for j,f in enumerate(fields)})
   log(name,epoch,'complete')
 temp=out/'joint_features.partial.npz';np.savez_compressed(temp,**features);temp.replace(out/'joint_features.npz')


def encodings(codes,queries,y,rng):
 n=len(y);fold=rng.randint(5,size=n);train=[];test=[]
 for c,qc,penalty in zip(codes,queries,[5.,10.,5.,10.]):
  size=int(max(c.max(),qc.max(initial=-1))+1);total=np.bincount(c,minlength=size);s=np.bincount(c,weights=y,minlength=size);ss=np.bincount(c,weights=y*y,minlength=size);low=np.bincount(c,weights=y<=2,minlength=size);high=np.bincount(c,weights=y>=4,minlength=size);A=np.zeros((n,5),np.float32)
  def values(count,sums,squares,lo,hi,prior):
   mean=(sums+penalty*prior)/(count+penalty);sd=np.sqrt(np.maximum(0,(squares+penalty*(prior*prior+1))/(count+penalty)-mean*mean));return np.c_[mean,sd,np.log1p(count),(lo+1)/(count+5),(hi+3)/(count+5)]
  for k in range(5):
   rows=np.flatnonzero(fold==k);count=total-np.bincount(c[rows],minlength=size);sums=s-np.bincount(c[rows],weights=y[rows],minlength=size);squares=ss-np.bincount(c[rows],weights=y[rows]*y[rows],minlength=size);lo=low-np.bincount(c[rows],weights=y[rows]<=2,minlength=size);hi=high-np.bincount(c[rows],weights=y[rows]>=4,minlength=size);prior=np.mean(y[fold!=k]);A[rows]=values(count,sums,squares,lo,hi,prior)[c[rows]]
  V=values(total,s,ss,low,high,np.mean(y));B=np.tile(np.array([np.mean(y),1,0,.2,.6],np.float32),(len(qc),1));ok=qc>=0;B[ok]=V[qc[ok]];train.append(A);test.append(B)
 return np.c_[tuple(train)],np.c_[tuple(test)]

def raw_worker(data,out,full):
 if (out/'raw_features.npz').exists():return
 p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);U,I=len(users),len(items);X=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=(U,I)).tocsr();up=np.asarray(X.sum(1)).ravel();pop=np.asarray(X.sum(0)).ravel();idf=np.log1p(U/np.maximum(pop,1)).astype(np.float32);weighted=X.multiply(idf).multiply((1/np.sqrt(np.maximum(up,1)))[:,None]).tocsr();svd=TruncatedSVD(n_components=24,n_iter=5,random_state=42);svd.fit(weighted);Q=svd.components_.T.astype(np.float32);user_sum=X.multiply(idf)@Q
 qt=q.timestamp.to_numpy(float);good=(t>=0)&(t<=4102444800);qgood=(qt>=0)&(qt<=4102444800);bin=np.where(good,np.floor(np.clip(t,0,4102444800)/(365*86400)),-1).astype(np.int64);qb=np.where(qgood,np.floor(np.clip(qt,0,4102444800)/(365*86400)),-1).astype(np.int64)
 uc,quc,ukey=codes((u.astype(np.int64)+1)*1000000000+bin,(qu.astype(np.int64)+1)*1000000000+qb);ic,qic,ikey=codes((i.astype(np.int64)+1)*1000000000+bin,(qi.astype(np.int64)+1)*1000000000+qb)
 te,ve=encodings([u,i,uc,ic],[qu,qi,quc,qic],y,np.random.RandomState(19))
 def design(uu,ii,tt,bins,enc,training):
  knownu=uu>=0;knowni=ii>=0;profile=np.zeros((len(uu),24),np.float32);book=np.zeros_like(profile);profile[knownu]=user_sum[uu[knownu]];book[knowni]=Q[ii[knowni]]
  present=np.zeros(len(uu),bool);ok=knownu&knowni
  if training:present[:]=True
  else:present[ok]=np.asarray(X[uu[ok],ii[ok]]).ravel()>0
  ix=np.flatnonzero(present);profile[ix]-=idf[ii[ix],None]*Q[ii[ix]];profile/=np.maximum(np.linalg.norm(profile,axis=1,keepdims=True),1e-8);book/=np.maximum(np.linalg.norm(book,axis=1,keepdims=True),1e-8)
  days=np.clip(tt,0,4102444800)/86400;numeric=np.c_[enc,profile,book,profile*book,days/365.25,np.sin(days/365.25*2*np.pi),np.cos(days/365.25*2*np.pi)].astype(np.float32);frame=pd.DataFrame(numeric,columns=['x'+str(k) for k in range(numeric.shape[1])]);frame['user']=[str(v) for v in uu];frame['item']=[str(v) for v in ii];frame['user_year']=[str(v)+':'+str(z) for v,z in zip(uu,bins)];frame['item_year']=[str(v)+':'+str(z) for v,z in zip(ii,bins)];return frame
 A=design(u,i,t,bin,te,True);B=design(qu,qi,qt,qb,ve,False)
 import lightgbm as lgb
 cats=['user','item','user_year','item_year']
 for key in cats:
  A[key]=A[key].astype('category');B[key]=pd.Categorical(B[key],categories=A[key].cat.categories)
 log('LightGBM raw fit',A.shape)
 m=lgb.LGBMRegressor(n_estimators=600,num_leaves=31,max_bin=63,learning_rate=.06,min_child_samples=150,reg_lambda=15,colsample_bytree=.85,subsample=.85,subsample_freq=1,cat_smooth=20,max_cat_threshold=32,path_smooth=10,n_jobs=2,random_state=42,verbosity=-1)
 A=A.drop(columns=cats);B=B.drop(columns=cats)
 m.fit(A,y,eval_set=[(A,y)],eval_metric='l2',callbacks=[lgb.log_evaluation(100)])
 m.booster_.save_model(str(out/'raw_ratings.txt'));features={f'raw_lgbm_e{epoch}':clip(m.predict(B,num_iteration=epoch)) for epoch in [300,600]};np.savez_compressed(out/'raw_features.npz',**features)
 np.savez_compressed(out/'raw_query_design.npz',numeric=B.to_numpy(np.float32));np.savez_compressed(out/'unsupervised_embeddings.npz',Q=Q,idf=idf);log('Direct rating learner complete')


def learned_worker(data,out,full,prior,wide=False):
 prefix='wide_' if wide else '';target=out/('learnedwide_features.npz' if wide else 'learned_features.npz')
 if target.exists():return
 p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);U,I=len(users),len(items)
 with open(prior/('full' if full else 'validation')/'temporal_bias.pkl','rb') as handle:bias=pickle.load(handle)
 residual=y-bias.training_prediction;bq=bias.predict(qu.astype(np.int64),qi.astype(np.int64),q.timestamp.to_numpy(float));X=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=(U,I)).tocsr();R=coo_matrix((residual,(u,i)),shape=(U,I)).tocsr();T=coo_matrix((t,(u,i)),shape=(U,I)).tocsr();C=R.tocsc();XT=X.T.tocsr();RT=R.T.tocsr();pop=np.asarray(X.sum(0)).ravel();norm=np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel())
 settings=[('rating',5),('rating',25),('rating',100),('presence',25),('presence',100)];result={f'{prefix}learned_{kind}{reg}_{tau}':bq.copy() for kind,reg in settings for tau in [0,365]};result[prefix+'learned_support']=np.zeros(len(q));groups={}
 for z,j in enumerate(qi):
  if j>=0 and qu[z]>=0:groups.setdefault(j,[]).append(z)
 keys=np.array(list(groups));rows=[[] for _ in settings];cols=[[] for _ in settings];vals=[[] for _ in settings]
 for offset in range(0,len(keys),64):
  batch=keys[offset:offset+64];co=(XT[batch]@X).toarray();dot=(RT[batch]@R).toarray()
  for z,j in enumerate(batch):
   co[z,j]=0;dot[z,j]=0;sim=co[z]/np.sqrt(np.maximum(pop[j]*pop,1));corr=np.abs(dot[z])/np.maximum(norm[j]*norm,1e-8)*co[z]/(co[z]+5)
   take=min(128 if wide else 32,len(sim));nb1=np.argpartition(sim,-take)[-take:];nb2=np.argpartition(corr,-take)[-take:];nb=np.unique(np.r_[nb1[sim[nb1]>0],nb2[corr[nb2]>0]]);nb=nb[nb!=j]
   if not len(nb):continue
   lo,hi=C.indptr[j:j+2];us=C.indices[lo:hi];yy=C.data[lo:hi];queries=np.array(groups[j]);rr=R[qu[queries]][:,nb].toarray();xx=X[qu[queries]][:,nb].toarray();times=T[qu[queries]][:,nb].toarray();tw=np.exp(-np.abs(times-q.timestamp.to_numpy(float)[queries,None])/86400/365)*((times>=0)&(times<=4102444800));result[prefix+'learned_support'][queries]=np.log1p(xx.sum(1))
   for kind in ['rating','presence']:
    A=(R if kind=='rating' else X)[us][:,nb].toarray();dual=wide and A.shape[0]<A.shape[1];gram=A@A.T if dual else A.T@A;eig,V=np.linalg.eigh(gram);rhs=V.T@yy if dual else V.T@(A.T@yy)
    for k,(label,reg) in enumerate(settings):
     if label!=kind:continue
     w=((A.T@V if dual else V)@(rhs/(np.maximum(eig,0)+reg))).astype(np.float32);design=rr if kind=='rating' else xx;result[f'{prefix}learned_{kind}{reg}_0'][queries]+=design@w;result[f'{prefix}learned_{kind}{reg}_365'][queries]+=(design*tw)@w;rows[k].extend(nb);cols[k].extend([j]*len(nb));vals[k].extend(w)
  if offset%2048==0:log('Learned rating neighborhoods',offset,'/',len(keys))
 for k,(kind,reg) in enumerate(settings):
  W=coo_matrix((np.array(vals[k],np.float32),(rows[k],cols[k])),shape=(I,I)).tocsr();assert np.count_nonzero(W.diagonal())==0;save_npz(out/f'{prefix}learned_{kind}{reg}.npz',W)
 for key in result:
  if key!=prefix+'learned_support':result[key]=clip(result[key])
 np.savez_compressed(target,**result);log('Learned rating/presence neighborhoods complete',prefix)


def pearson_worker(data,out,full,prior):
 target=out/'pearson_features.npz'
 if target.exists():return
 p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);U,I=len(users),len(items)
 with open(prior/('full' if full else 'validation')/'temporal_bias.pkl','rb') as h:bias=pickle.load(h)
 bq=bias.predict(qu.astype(np.int64),qi.astype(np.int64),q.timestamp.to_numpy(float));X=coo_matrix((np.ones(len(u),np.float32),(u,i)),shape=(U,I)).tocsr();Y=coo_matrix((y.astype(np.float32),(u,i)),shape=(U,I)).tocsr();R=coo_matrix(((y-bias.training_prediction).astype(np.float32),(u,i)),shape=(U,I)).tocsr();T=coo_matrix((t,(u,i)),shape=(U,I)).tocsc();RC=R.tocsc();up=np.asarray(X.sum(1)).ravel();XT=X.T.tocsr();assert np.array_equal(Y.indices,R.indices)
 so=out/'pearson.so'
 if not so.exists():subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task3_pearson_kernel.cpp')),'-o',str(so)],check=True)
 lib=ctypes.CDLL(str(so.resolve()));fn=lib.pair_pearson;ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');fn.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,ip,fp,fp,fp,fp,ip];fn.restype=None
 variants=['raw_positive','raw_signed','residual_positive','residual_signed'];result={f'pearson_{name}_{k}_{tau}':bq.copy() for name in variants for k in [20,50] for tau in [0,365]};result.update({f'pearson_{name}_support':np.zeros(len(q)) for name in variants});groups={}
 for z,user in enumerate(qu):
  if user>=0 and qi[z]>=0:groups.setdefault(user,[]).append(z)
 keys=np.array(list(groups));qt=q.timestamp.to_numpy(float)
 for offset in range(0,len(keys),64):
  batch=keys[offset:offset+64];co=(X[batch]@XT).toarray()
  for z,user in enumerate(batch):
   c=co[z];c[user]=0;sim=c/np.sqrt(np.maximum(up[user]*up,1));l1=min(1000,U);l2=min(500,U);n1=np.argpartition(sim,-l1)[-l1:];n2=np.argpartition(c,-l2)[-l2:];nb=np.unique(np.r_[n1[c[n1]>=3],n2[c[n2]>=5]]).astype(np.int32)
   wanted=np.zeros(U,bool)
   for queryrow in groups[user]:
    lo,hi=RC.indptr[qi[queryrow]:qi[queryrow]+2];wanted[RC.indices[lo:hi]]=True
   nb=nb[wanted[nb]]
   if not len(nb):continue
   raw=np.zeros(len(nb),np.float32);res=raw.copy();counts=np.zeros(len(nb),np.int32);fn(int(user),len(nb),nb,Y.indptr,Y.indices,Y.data,R.data,raw,res,counts);shrink=counts/(counts+20.);rw=raw*shrink;rr=res*shrink;weights={}
   for name,v in [('raw_positive',np.maximum(rw,0)),('raw_signed',rw),('residual_positive',np.maximum(rr,0)),('residual_signed',rr)]:
    w=np.zeros(U);w[nb]=v;weights[name]=w
   for row in groups[user]:
    j=qi[row];lo,hi=RC.indptr[j:j+2];other=RC.indices[lo:hi];values=RC.data[lo:hi];dates=T.data[lo:hi];distance=np.abs(dates-qt[row])/86400;time_weight=np.exp(-distance/365)*((dates>=0)&(dates<=4102444800))
    for name,weight in weights.items():
     w0=weight[other];active=np.flatnonzero(w0);result[f'pearson_{name}_support'][row]=np.log1p(len(active))
     if not len(active):continue
     vals=values[active];original=w0[active];tw=time_weight[active]
     for tau in [0,365]:
      w=original if tau==0 else original*tw;ranking=np.argsort(-np.abs(w),kind='stable')[:50]
      for k in [20,50]:
       chosen=ranking[:k];result[f'pearson_{name}_{k}_{tau}'][row]+=(w[chosen]@vals[chosen])/(np.abs(w[chosen]).sum()+.5)
  if offset%2048==0:log('Exact common-book Pearson users',offset,'/',len(keys))
 for name in result:
  if not name.endswith('_support'):result[name]=clip(result[name])
 np.savez_compressed(target,**result);log('Exact common-book Pearson complete')


def current_features(data,legacy,prior,full):
 if full:base=np.load(prior/'full/baseline_features.npz')['features']
 else:base,y,n=baseline_validation(data,legacy)
 zs=[np.load(prior/('full' if full else 'validation')/(k+'_features.npz')) for k in ['library','temporal']];F=np.c_[base,np.column_stack([z[k] for z in zs for k in z.files])];return F

def select(F,G,y,n,out,names,only_all=False):
 idx=np.random.RandomState(17).permutation(n);tr=idx[:int(.8*n)];va=idx[int(.8*n):];old=train_tree(F[tr],y[tr],8,530);po=clip(old.predict(F[va]));best=(mse(y[va],po),0,8,530,[],"reference");report={'current_inner_MSE':best[0]}
 groups={'all':list(range(G.shape[1])),'learned':[j for j,k in enumerate(names) if 'learned_' in k or k.startswith('pearson_')],'learned_raw':[j for j,k in enumerate(names) if 'learned_' in k or k.startswith(('raw_','pearson_'))]}
 if only_all:groups={'all':groups['all']}
 for group,keep in groups.items():
  X=np.c_[F,G[:,keep]]
  for depth in [6,8]:
   m=CatBoostRegressor(iterations=1600,depth=depth,learning_rate=.025,l2_leaf_reg=15,random_seed=42,thread_count=2,loss_function='RMSE',allow_writing_files=False,verbose=False);m.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=120);iters=m.get_best_iteration()+1;pn=clip(m.predict(X[va]))
   for weight in [.5,.75,1.]:
    score=mse(y[va],weight*pn+(1-weight)*po);report[f'{group}_depth{depth}_mix{weight}']=dict(inner_MSE=score,iterations=iters)
    if score<best[0]:best=(score,weight,depth,iters,keep,group)
   log('Stacker',group,depth,'inner MSE',mse(y[va],pn),'iterations',iters)
 score,weight,depth,iters,keep,group=best;X=np.c_[F,G[:,keep]];old=train_tree(F[:n],y[:n],8,530);old.save_model(str(out/'reference.cbm'));m=None
 if weight:
  m=CatBoostRegressor(iterations=iters,depth=depth,learning_rate=.025,l2_leaf_reg=15,random_seed=42,thread_count=2,loss_function='RMSE',allow_writing_files=False,verbose=False);m.fit(X[:n],y[:n]);m.save_model(str(out/'new_meta.cbm'))
 po=clip(old.predict(F[n:]));pn=clip(m.predict(X[n:])) if m else po;pred=weight*pn+(1-weight)*po;r=dict(selected=dict(inner_MSE=score,new_weight=weight,depth=depth,iterations=iters,feature_group=group),selected_features=[names[j] for j in keep],keep_indices=keep,search=report,reference_evaluation_MSE=mse(y[n:],po),new_evaluation_MSE=mse(y[n:],pn),selected_evaluation_MSE=mse(y[n:],pred),blend_evaluation_MSE=mse(y[n:],.5*pn+.5*po));np.savez_compressed(out/'evaluation.npz',baseline=po,new=pn,selected=pred,y=y[n:]);(out/'results.json').write_text(json.dumps(r,indent=2));log('SELECTED',r);return old,m,weight,r

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--prior-dir',type=Path,required=True);ap.add_argument('--baseline-dir',type=Path,required=True);ap.add_argument('--worker',choices=['joint','raw','learned','learnedwide','pearson']);ap.add_argument('--full',action='store_true');ap.add_argument('--predict',action='store_true');ap.add_argument('--only-all',action='store_true',help='Compatibility flag: each bank uses the all-feature stack');args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);os.environ.setdefault('SURPRISE_DATA_FOLDER',str(args.output_dir/'surprise_data'))
 manifest={k:hashlib.sha256((args.data_dir/k).read_bytes()).hexdigest() for k in ['tasks_3_4_train.csv','task_3_test.csv']}
 if args.worker:
  stage={'full':args.full,**manifest};guard=args.output_dir/'stage_manifest.json'
  if guard.exists() and json.loads(guard.read_text())!=stage:raise ValueError('Cache stage/data mismatch: use a fresh stage directory')
  temp=args.output_dir/('stage_'+str(os.getpid())+'.json');temp.write_text(json.dumps(stage,indent=2));temp.replace(guard)
  if args.worker=='pearson':return pearson_worker(args.data_dir,args.output_dir,args.full,args.prior_dir)
  if args.worker in ['learned','learnedwide']:return learned_worker(args.data_dir,args.output_dir,args.full,args.prior_dir,wide=args.worker=='learnedwide')
  return {'joint':joint_worker,'raw':raw_worker}[args.worker](args.data_dir,args.output_dir,args.full)
 required=[args.prior_dir/'validation/library_features.npz',args.prior_dir/'validation/temporal_features.npz',args.prior_dir/'full/baseline_features.npz',args.prior_dir/'full/library_features.npz',args.prior_dir/'full/temporal_features.npz']
 if not all(path.exists() for path in required):subprocess.run([sys.executable,str(Path(__file__).with_name('task3_optimize.py')),'--data-dir',str(args.data_dir),'--output-dir',str(args.prior_dir),'--baseline-dir',str(args.baseline_dir),'--predict'],check=True)
 if not (args.baseline_dir/'split.npz').exists():args.baseline_dir=Path(json.loads((args.baseline_dir/'run.json').read_text())['cache']).resolve()
 mf=args.output_dir/'manifest.json'
 if mf.exists() and json.loads(mf.read_text())!=manifest:raise ValueError('Data changed; use fresh output directory')
 mf.write_text(json.dumps(manifest,indent=2));prior_manifest=args.prior_dir/'input_manifest.json'
 if prior_manifest.exists() and json.loads(prior_manifest.read_text())!=manifest:raise ValueError('Reference data differ; regenerate reference caches')
 kinds=['joint','raw','learned','learnedwide','pearson']
 def workers(part,full,needed):
  jobs=[]
  for kind in needed:
   h=open(part/(kind+'.log'),'a');cmd=[sys.executable,str(Path(__file__).resolve()),'--data-dir',str(args.data_dir),'--output-dir',str(part),'--prior-dir',str(args.prior_dir),'--baseline-dir',str(args.baseline_dir),'--worker',kind]+(['--full'] if full else []);jobs.append((subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT),h,kind))
  for proc,handle,kind in jobs:
   rc=proc.wait();handle.close()
   if rc:raise RuntimeError(f'{kind} failed; inspect {part/(kind+".log")}')
 cfg={'reference':[8,530,.03,10,42],'depths':[6,8],'iterations':1600,'learning_rate':.025,'l2_leaf_reg':15,'seed':42,'inner_seed':17,'weights':[.5,.75,1.]}
 part=args.output_dir/'validation';part.mkdir(exist_ok=True);workers(part,False,kinds);F=current_features(args.data_dir,args.baseline_dir,args.prior_dir,False);values={}
 for kind in kinds:
  z=np.load(part/(kind+'_features.npz'));values.update({k:z[k] for k in z.files})
 p,tr,q,*_=dataset(args.data_dir,False);y=q.rating.to_numpy();n=int(.95*len(p))-int(.9*len(p));runs={}
 for variant in ['plain','common']:
  names=[k for k in values if variant=='common' or not k.startswith('pearson_')];G=np.column_stack([values[k] for k in names]);modeldir=args.output_dir/'models'/variant;modeldir.mkdir(parents=True,exist_ok=True);rp=modeldir/'results.json';cached=json.loads(rp.read_text()) if rp.exists() else {}
  valid=cached.get('model_config')==cfg and cached.get('component_features')==names and cached.get('input_manifest')==manifest
  if valid:
   r=cached;old=CatBoostRegressor();old.load_model(str(modeldir/'reference.cbm'));weight=r['selected']['new_weight'];model=CatBoostRegressor() if weight else None
   if model:model.load_model(str(modeldir/'new_meta.cbm'))
  else:
   old,model,weight,r=select(F,G,y,n,modeldir,names,only_all=True);r.update(model_config=cfg,component_features=names,input_manifest=manifest);rp.write_text(json.dumps(r,indent=2))
  runs[variant]=(old,model,weight,r)
 primary=min(runs,key=lambda key:runs[key][3]['selected']['inner_MSE']);evals={key:np.load(args.output_dir/'models'/key/'evaluation.npz') for key in runs};assert np.array_equal(evals['plain']['y'],evals['common']['y']);mixed=.5*evals['plain']['selected']+.5*evals['common']['selected'];summary={'primary_variant':primary,'variants':{k:v[3] for k,v in runs.items()},'reference_evaluation_MSE':runs[primary][3]['reference_evaluation_MSE'],'primary_evaluation_MSE':runs[primary][3]['selected_evaluation_MSE'],'blend_evaluation_MSE':mse(evals['plain']['y'],mixed),'input_manifest':manifest};(args.output_dir/'results.json').write_text(json.dumps(summary,indent=2));log('LOCAL',primary,summary['primary_evaluation_MSE'],'blend',summary['blend_evaluation_MSE'])
 if not args.predict:return
 part=args.output_dir/'full';part.mkdir(exist_ok=True);needed=[]
 for _,_,_,r in runs.values():
  for name in r['selected_features']:
   kind='joint' if name.startswith('joint') else 'raw' if name.startswith('raw') else 'learnedwide' if name.startswith('wide_') else 'pearson' if name.startswith('pearson_') else 'learned'
   if kind not in needed:needed.append(kind)
 workers(part,True,needed);F=current_features(args.data_dir,args.baseline_dir,args.prior_dir,True);values={}
 for kind in needed:
  z=np.load(part/(kind+'_features.npz'));values.update({k:z[k] for k in z.files})
 test=pd.read_csv(args.data_dir/'task_3_test.csv');predictions={};references={}
 for variant,(old,model,weight,r) in runs.items():
  G=np.column_stack([values[k] for k in r['selected_features']]) if r['selected_features'] else np.empty((len(F),0));po=clip(old.predict(F));pn=clip(model.predict(np.c_[F,G])) if model else po;pred=clip(weight*pn+(1-weight)*po);assert len(pred)==20000 and np.isfinite(pred).all();predictions[variant]=pred;references[variant]=po;frame=test.copy();frame['rating']=pred;frame.to_csv(args.output_dir/'models'/variant/'task_3_predictions.csv',index=False)
 assert np.allclose(references['plain'],references['common'],atol=1e-12);blend=clip(.5*predictions['plain']+.5*predictions['common']);frame=test.copy();frame['rating']=predictions[primary];frame.to_csv(args.output_dir/'task_3_predictions.csv',index=False);frame['rating']=blend;frame.to_csv(args.output_dir/'task_3_alternative_predictions.csv',index=False);np.savez_compressed(args.output_dir/'test_predictions.npz',reference=references[primary],primary=predictions[primary],blend=blend,**predictions);log('SAVED',args.output_dir/'task_3_predictions.csv')

if __name__=='__main__':main()
