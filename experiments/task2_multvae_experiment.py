"""Train Mult-VAE, add its scores to all accepted Task2 components, fit XGBoost.
Run: PYTHONPATH=/workspace/work/task4_xgb OPENBLAS_NUM_THREADS=2 python experiments/task2_multvae_experiment.py
Uses only provided data; includes rating-zero events. Existing caches required.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import time,json,hashlib,gc
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix
from task2_multvae_numpy import MultVAE,normalize,gradient_check
from task2_xgboost_refresh import W,DATA,BASE,load_features,estimator,mirrored,labels,margin,accuracy,catmargin
from catboost import CatBoostClassifier
import xgboost as xgb
OUT=W/'task2_multvae';OUT.mkdir(exist_ok=True)
PREV=W/'task2_xgboost_refresh'
def log(*args):print(*args,flush=True)
def matrix(u,i,rows,shape):
 X=coo_matrix((np.ones(len(rows),dtype='f'),(u[rows],i[rows])),shape=shape).tocsr();X.data[:]=1;return X

def train_vae(X,users,items,inner=None,epochs=40):
 model=MultVAE(X.shape[1]);xn=normalize(X);active=np.flatnonzero(np.diff(X.indptr)>0);rng=np.random.RandomState(44);best=(-1,None,None);stale=0;history=[];t=time.time()
 for epoch in range(1,epochs+1):
  order=rng.permutation(active);loss=[]
  for j in range(0,len(order),256):
   ix=order[j:j+256];loss.append(model.step(X[ix],xn[ix]))
  row={'epoch':epoch,'training_loss':float(np.mean(loss)),'elapsed_seconds':round(time.time()-t,1)}
  if inner is not None and (epoch%2==0 or epoch==epochs):
   raw,lp=model.scores(X,users,items);n=len(raw)//2;acc=accuracy(raw[:n][inner]-raw[n:][inner]);row['selection_train_accuracy']=acc
   if acc>best[0]+1e-5:
    best=(acc,epoch,(raw,lp));stale=0;np.savez_compressed(OUT/'validation_scores.npz',raw=raw,logprob=lp,epoch=epoch)
   else:stale+=1
  history.append(row);log('VAE',json.dumps(row));(OUT/('validation_history.json' if inner is not None else 'production_history.json')).write_text(json.dumps(history,indent=2))
  if inner is not None and stale>=5:break
 if inner is not None:return best[1],best[2]
 model.save(OUT/'production_vae.npz');model.restore(OUT/'production_vae.npz');raw,lp=model.scores(X,users,items)
 np.savez_compressed(OUT/'test_scores.npz',raw=raw,logprob=lp)
 return epochs,(raw,lp)

def add_scores(d,s,scores):
 n=len(d);extra=np.column_stack(scores);assert len(extra)==2*n and np.isfinite(extra).all()
 return np.c_[d,extra[:n]-extra[n:]].astype('f'),np.c_[s,extra[:n]+extra[n:]].astype('f')

def main():
 log('GRADIENT_CHECK',gradient_check());prev=json.loads((PREV/'results.json').read_text());manifest=prev['input_manifest']
 for f in ['tasks_1_2_train.csv','task_2_test.csv']:assert hashlib.sha256((DATA/f).read_bytes()).hexdigest()==manifest[f]
 guard={'input_manifest':manifest,'architecture':[128,64],'seed':3181,'dropout':.5,'maximum_epochs':40,'kernel_sha256':hashlib.sha256(Path(__file__).with_name('task2_multvae_numpy.py').read_bytes()).hexdigest()}
 gp=OUT/'cache_guard.json'
 if gp.exists():assert json.loads(gp.read_text())==guard,'Changed inputs or VAE configuration: use a fresh output folder'
 else:gp.write_text(json.dumps(guard,indent=2))
 accepted_path=Path('/workspace/output/task2_xgboost_blend/task_2_predictions.csv');ah=hashlib.sha256(accepted_path.read_bytes()).hexdigest();assert ah=='9dd469b074af077ca895b1e1ed53b3b2fae456b3a98c9b8dde603d0169816880'
 df=pd.read_csv(DATA/'tasks_1_2_train.csv');u,users=pd.factorize(df.user);i,items=pd.factorize(df.item);shape=(len(users),len(items));order=np.random.RandomState(42).permutation(len(df));a=int(.9*len(df));hold=order[a:];mask=np.zeros(len(df),bool);mask[order[:a]]=True;eligible=np.flatnonzero(mask)
 X=matrix(u,i,eligible,shape);rng=np.random.RandomState(43);neg=i[eligible[rng.randint(len(eligible),size=len(hold))]];same=neg==i[hold]
 while same.any():neg[same]=i[eligible[rng.randint(len(eligible),size=int(same.sum()))]];same=neg==i[hold]
 qu=np.r_[u[hold],u[hold]];qi=np.r_[i[hold],neg];assert np.asarray(X[u[hold],i[hold]]).max()==0,'Held-out positive present in training history'
 split=np.load(BASE/'validation/validation_margins.npz');sel,ev=split['selection'],split['evaluation'];idx=np.random.RandomState(17).permutation(np.flatnonzero(sel));cut=int(.8*len(idx));tr,va=idx[:cut],idx[cut:]
 log('DATA',shape,'interactions',X.nnz,'with rating-zero events; positive holdouts excluded')
 if (OUT/'validation_scores.npz').exists():
  z=np.load(OUT/'validation_scores.npz');epochs=int(z['epoch']);vs=(z['raw'],z['logprob'])
 else:epochs,vs=train_vae(X,qu,qi,inner=tr)
 log('VAE_SELECTED_EPOCH',epochs);names,d,s=load_features('validation');ad,ass=add_scores(d,s,vs)
 model=estimator(1600,True);model.fit(mirrored(ad[tr],ass[tr]),labels(len(tr)),eval_set=[(mirrored(ad[va],ass[va]),labels(len(va)))],verbose=200);count=int(model.best_iteration)+1;model.save_model(OUT/'selection.ubj');ni=margin(model,ad[va],ass[va])
 oldx=xgb.XGBClassifier();oldx.load_model(PREV/'selection.ubj');cat=CatBoostClassifier();cat.load_model(str(PREV/'reference_inner.cbm'));w0=prev['xgboost_weight'];ci=w0*margin(oldx,d[va],s[va])+(1-w0)*catmargin(cat,d[va],s[va])
 search={str(float(w)):accuracy(w*ni+(1-w)*ci) for w in np.linspace(0,1,21)};weight=max(map(float,search),key=lambda w:(search[str(w)],-w))
 report={'vae':{'hidden':128,'latent':64,'input_dropout':.5,'learning_rate':.001,'batch_size':256,'KL_cap':.2,'KL_schedule':'min(.2,updates/10000)','epochs':epochs,'full_vocabulary':shape[1],'posterior_mean_inference':True,'production_parameter_storage':'float16 reloaded before scoring'},'features':names+['multvae_logit','multvae_logprob'],'iterations':count,'new_weight':weight,'inner_accepted_accuracy':accuracy(ci),'inner_augmented_accuracy':accuracy(ni),'inner_blend_accuracy':accuracy(weight*ni+(1-weight)*ci),'blend_search':search,'protocol':'Mult-VAE fit90% observed rows only. Positive holdouts absent from all input histories. Epoch selected on inner training pairs only; meta early stopping/weight use inner validation. Final5% evaluation not used for selection. All original16 components retained. Rating-zero events included. Production VAE refit all raw train rows; expanded meta fits all known holdout pairs. Reused cohorts and synthetic negatives limit independence.'}
 (OUT/'locked_selection.json').write_text(json.dumps(report,indent=2));log('LOCKED',json.dumps({k:v for k,v in report.items() if k not in ['features','blend_search']}))
 model=estimator(count);model.fit(mirrored(ad[sel],ass[sel]),labels(int(sel.sum())),verbose=False);model.save_model(OUT/'evaluation.ubj');ne=margin(model,ad[ev],ass[ev]);ce=np.load(PREV/'evaluation.npz')['blend'];blend=weight*ne+(1-weight)*ce
 report.update(evaluation_accepted_accuracy=accuracy(ce),evaluation_augmented_accuracy=accuracy(ne),evaluation_blend_accuracy=accuracy(blend),evaluation_vae_alone_accuracy=accuracy(vs[0][:len(d)][ev]-vs[0][len(d):][ev]))
 np.savez_compressed(OUT/'evaluation.npz',reference=ce,augmented=ne,blend=blend);log('EVALUATION',json.dumps({k:v for k,v in report.items() if k.startswith('evaluation')}))
 test=pd.read_csv(DATA/'task_2_test.csv');tu=pd.Index(users).get_indexer(test.user);t1=pd.Index(items).get_indexer(test.item1);t2=pd.Index(items).get_indexer(test.item2);assert min(tu.min(),t1.min(),t2.min())>=0
 del X;gc.collect();FX=matrix(u,i,np.arange(len(df)),shape)
 if (OUT/'test_scores.npz').exists():
  z=np.load(OUT/'test_scores.npz');tscores=(z['raw'],z['logprob'])
 else:_,tscores=train_vae(FX,np.r_[tu,tu],np.r_[t1,t2],epochs=epochs)
 fn,td,ss=load_features('full');assert fn==names;atd,ats=add_scores(td,ss,tscores)
 model=estimator(count);model.fit(mirrored(ad,ass),labels(len(ad)),verbose=False);model.save_model(OUT/'production.ubj');tn=margin(model,atd,ats)
 oldx=xgb.XGBClassifier();oldx.load_model(PREV/'production.ubj');cat.load_model(str(BASE/'validation/stacker.cbm'));tc=w0*margin(oldx,td,ss)+(1-w0)*catmargin(cat,td,ss)
 accepted=pd.read_csv(accepted_path);pd.testing.assert_frame_equal(accepted.iloc[:,:-1],test.iloc[:,:-1]);assert np.array_equal(accepted.which.to_numpy(),np.where(tc>=0,test.item1,test.item2))
 replay=xgb.XGBClassifier();replay.load_model(OUT/'production.ubj');assert np.array_equal(tn,margin(replay,atd,ats)) and np.allclose(tn,-margin(model,-atd,ats),atol=1e-7)
 report['outputs']={}
 for label,v in [('primary',tn),('blend',weight*tn+(1-weight)*tc)]:
  dest=Path('/workspace/output')/('task2_multvae_'+label);dest.mkdir(exist_ok=True);path=dest/'task_2_predictions.csv';frame=test.copy();frame['which']=np.where(v>=0,test.item1,test.item2);frame.to_csv(path,index=False);check=pd.read_csv(path)
  pd.testing.assert_frame_equal(check.iloc[:,:-1],test.iloc[:,:-1]);assert len(check)==20000 and np.isfinite(v).all() and ((check.which==check.item1)|(check.which==check.item2)).all();report['outputs'][label]={'path':str(path),'rows':len(check),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'changed_vs_accepted':int(np.sum(check.which.to_numpy()!=accepted.which.to_numpy()))}
 assert hashlib.sha256(accepted_path.read_bytes()).hexdigest()==ah
 report['input_manifest']=manifest;(OUT/'results.json').write_text(json.dumps(report,indent=2));log('EXPORTED',json.dumps(report['outputs']))
if __name__=='__main__':main()
