"""Task 6 date-aware gap ensemble. Predictions are nonnegative seconds."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,hashlib,json,sys
from pathlib import Path
from importlib.metadata import version
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from task6_history_features import history_features
from task6_item_time import build_item_features


def load_data(data_dir):
    p=pd.read_csv(data_dir/'task_6_train.csv');test=pd.read_csv(data_dir/'task_6_test.csv')
    u,users=pd.factorize(p.user);i,items=pd.factorize(p.item)
    if np.any(np.diff(u)<0):raise ValueError('Expected contiguous user histories')
    t=p.timestamp.to_numpy(dtype=float);ratings=p.rating.to_numpy(dtype=float);starts=np.searchsorted(u,np.arange(len(users)+1));early=[];v=[];e=[]
    for user in range(len(users)):
        lo,hi=starts[user:user+2];n=hi-lo-1
        if n<3 or np.any(np.diff(t[lo:hi])<0):raise ValueError('Expected chronological histories with at least four interactions')
        v.append(hi-2);e.append(hi-1)
        for k in np.unique(np.linspace(1,n-3,min(5,max(0,n-3)),dtype=int)):early.append(lo+k+1)
    early=np.asarray(early);v=np.asarray(v);e=np.asarray(e);rows=np.r_[early,v,e]
    return p,test,u,users,i,items,t,ratings,starts,early,v,e,rows


def core_matrix(t,ratings,u,starts,rows):
    out=[]
    for idx,row in enumerate(rows):
        lo=starts[u[row]];out.append(history_features(t[lo:row],ratings[lo:row],int(row-lo-1)))
        if idx%100000==0:print('History feature rows',idx,'/',len(rows),flush=True)
    return np.asarray(out,dtype=np.float32)


def frame(core,book,p,rows=None,query=None):
    x=pd.DataFrame(np.column_stack([core,book]));x.columns=['f'+str(k) for k in range(x.shape[1])]
    source=p.iloc[rows] if query is None else query
    x['user']=source.user.to_numpy();x['next_item']=source.item.to_numpy()
    return x


def model_for(cfg):
    return CatBoostRegressor(loss_function=cfg['loss'],iterations=cfg['iterations'],depth=cfg['depth'],learning_rate=.05,l2_leaf_reg=3,random_seed=cfg['seed'],thread_count=2,verbose=300,allow_writing_files=False,cat_features=['user','next_item'],max_ctr_complexity=1)


def run_models(config,cache,p,rows,core,book,y,query_core,query_book,query,phase):
    X=frame(core,book,p,rows=rows);Q=frame(query_core,query_book,p,query=query);predictions={}
    for name,cfg in config['models'].items():
        path=cache/(phase+'_'+name+'.cbm');m=model_for(cfg)
        if path.exists():m.load_model(str(path))
        else:
            print('Fit',phase,name,'examples',len(y),flush=True)
            m.fit(X,np.log1p(y) if cfg['log'] else y);m.save_model(str(path))
        pred=m.predict(Q)
        if cfg['log']:pred=np.expm1(pred)
        predictions[name]=np.maximum(pred,0)
    xr=np.expm1(core[:,[0,19,22]]).astype(float);qr=np.expm1(query_core[:,[0,19,22]]).astype(float)
    ridge=Ridge(alpha=1.).fit(xr,y);predictions['ridge']=np.maximum(0,ridge.predict(qr))
    return predictions


def main():
    here=Path(__file__).resolve().parent;ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=here.parent/'Assignment 1/assignment1_2026');ap.add_argument('--output-dir',type=Path,default=here/'artifacts'/'task6');ap.add_argument('--evaluate',action='store_true');ap.add_argument('--predict',action='store_true');args=ap.parse_args()
    if not args.evaluate and not args.predict:ap.error('Specify --evaluate, --predict, or both')
    config=json.loads((here/'task6_config.json').read_text());args.output_dir.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256()
    for file in ['task_6_train.csv','task_6_test.csv']:h.update((args.data_dir/file).read_bytes())
    for file in ['task6_pipeline.py','task6_history_features.py','task6_item_time.py','task6_config.json']:h.update((here/file).read_bytes())
    for package in ['numpy','pandas','catboost','scikit-learn']:h.update((package+version(package)).encode())
    h.update(sys.version.encode());cache=args.output_dir/h.hexdigest()[:16];cache.mkdir(exist_ok=True)
    p,test,u,users,i,items,t,ratings,starts,early,v,e,rows=load_data(args.data_dir)
    if not (cache/'core.npz').exists():np.savez_compressed(cache/'core.npz',core=core_matrix(t,ratings,u,starts,rows))
    core=np.load(cache/'core.npz')['core'];y=(t[rows]-t[rows-1])/86400
    a=len(early);b=a+len(v)
    if args.evaluate:
        allowed=np.ones(len(p),dtype=bool);allowed[v]=False;allowed[e]=False
        if not (cache/'fold_book.npz').exists():np.savez_compressed(cache/'fold_book.npz',book=build_item_features(i,t,allowed,i[rows],t[rows-1],rows))
        book=np.load(cache/'fold_book.npz')['book'];query=p.iloc[np.r_[v,e]]
        preds=run_models(config,cache,p,early,core[:a],book[:a],y[:a],core[a:],book[a:],query,'validation')
        result={}
        for name,pred in preds.items():result[name]={'selection_MAE':float(np.mean(abs(pred[:len(v)]-y[a:b]))*86400),'evaluation_MAE':float(np.mean(abs(pred[len(v):]-y[b:]))*86400)}
        combined=sum(weight*preds[name] for name,weight in config['weights'].items())
        result['ensemble']={'selection_MAE':float(np.mean(abs(combined[:len(v)]-y[a:b]))*86400),'evaluation_MAE':float(np.mean(abs(combined[len(v):]-y[b:]))*86400)}
        (args.output_dir/'validation_results.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
    if args.predict:
        if not (cache/'full_features.npz').exists():
            book=build_item_features(i,t,np.ones(len(p),dtype=bool),i[rows],t[rows-1],rows)
            um=dict(zip(users,range(len(users))));qlast=[];qcore=[]
            for user in test.user:
                lo,hi=starts[um[user]:um[user]+2];qcore.append(history_features(t[lo:hi],ratings[lo:hi],int(hi-lo-1)));qlast.append(t[hi-1])
            qi=pd.Index(items).get_indexer(test.item);qbook=build_item_features(i,t,np.ones(len(p),dtype=bool),qi,np.asarray(qlast))
            np.savez_compressed(cache/'full_features.npz',book=book,query_core=np.asarray(qcore,dtype=np.float32),query_book=qbook)
        f=np.load(cache/'full_features.npz');preds=run_models(config,cache,p,rows,core,f['book'],y,f['query_core'],f['query_book'],test,'full')
        delta=sum(weight*preds[name] for name,weight in config['weights'].items())*86400
        if not np.isfinite(delta).all() or (delta<0).any():raise ValueError('Invalid gap prediction')
        test['delta']=delta;test.to_csv(args.output_dir/'task_6_predictions.csv',index=False)
        print('Saved',len(test),'Task 6 predictions in seconds',flush=True)

if __name__=='__main__':main()
