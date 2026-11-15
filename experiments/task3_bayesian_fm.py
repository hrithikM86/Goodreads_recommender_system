"""Bayesian time-aware FM using myfm; five-draw checkpoint averages."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,hashlib,time,pickle
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,save_npz
from task3_fm import dataset,design,clip,mse

def sparse(ids,x,F):
    rows=np.repeat(np.arange(len(ids)),ids.shape[1]);cols=ids.ravel();data=x.ravel();ok=(cols>=0)&(data!=0);return coo_matrix((data[ok].astype(float),(rows[ok],cols[ok])),shape=(len(ids),F)).tocsr()

def main():
    from myfm import MyFMRegressor
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--full',action='store_true');ap.add_argument('--selection',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    epochs=[30,60,90] if not a.full else [json.loads(a.selection.read_text())['checkpoint']];manifest={'full':a.full,'epochs':epochs,'rank':16,'seed':83,'inputs':{f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['tasks_3_4_train.csv','task_3_test.csv']}};guard=a.out/'manifest.json'
    if guard.exists():assert json.loads(guard.read_text())==manifest
    guard.write_text(json.dumps(manifest,indent=2))
    if (a.out/'results.json').exists():return
    p,tr,q,u,i,t,y,qu,qi,users,items,aa,bb=dataset(a.data_dir,a.full);A,x,B,qx,F,maps,center=design(u,i,t,qu,qi,q.timestamp.to_numpy(float),len(users),False);X=sparse(A,x,F);Q=sparse(B,qx,F);save_npz(a.out/'query.npz',Q);window=set(e for n in epochs for e in range(n-4,n+1));preds={};draws={};start=time.time()
    def callback(iteration,fm,hyper,history):
        e=iteration+1
        if e in window:
            n=next(n for n in epochs if n-4<=e<=n);preds.setdefault(n,[]).append(np.asarray(fm.predict_score(Q,[])));draws.setdefault(n,[]).append((float(fm.w0),np.array(fm.w,copy=True),np.array(fm.V,copy=True)))
        if e%15==0:print('BAYES_FM',e,'seconds',round(time.time()-start),flush=True)
        return False,None
    model=MyFMRegressor(rank=16,init_stdev=.05,random_seed=83);model.fit(X,y,n_iter=max(epochs),n_kept_samples=1,group_shapes=[len(f['labels']) for f in maps.values()],callback=callback)
    features={f'bayes16_e{e}':clip(np.mean(preds[e],axis=0)) for e in epochs};screen={}
    if not a.full:
        n=bb-aa;idx=np.random.RandomState(17).permutation(n)[:int(.8*n)];screen={e:mse(q.rating.to_numpy()[idx],features[f'bayes16_e{e}'][idx]) for e in epochs};selected=min(epochs,key=lambda e:screen[e])
    else:selected=epochs[0]
    states=draws[selected];np.savez_compressed(a.out/'model.npz',w0=np.array([d[0] for d in states]),w=np.stack([d[1] for d in states]),V=np.stack([d[2] for d in states]));np.savez_compressed(a.out/'features.npz',**features)
    with open(a.out/'maps.pkl','wb') as h:pickle.dump(maps,h)
    result={'checkpoint':int(selected),'inner_training_MSE':screen,'selected_feature':f'bayes16_e{selected}','manifest':manifest};(a.out/'results.json').write_text(json.dumps(result,indent=2));print('BAYES_SELECTED',result,flush=True)
if __name__=='__main__':main()
