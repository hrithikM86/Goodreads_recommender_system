"""Task 6 chronological validation and ensemble experiments.
Run from repository root: python experiments/task6.py --cache /workspace/work/task6_features.npz
Feature cache must be produced using the documented chronology; see README.md.
"""
import argparse, json
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor

def features(times,k):
    # Predict gap k (times[k+1]-times[k]); only times through k are known.
    gaps=np.diff(times[:k+1]);recent=gaps[-5:]
    last=list(gaps[-5:][::-1]);last += [np.nan]*(5-len(last))
    return last+[len(gaps),float(np.median(gaps)),float(np.mean(gaps)),float(np.std(gaps)),float(np.min(gaps)),float(np.max(gaps)),float(np.quantile(gaps,.25)),float(np.quantile(gaps,.75)),float(np.median(recent)),float(np.mean(recent)),float(np.std(recent)),float(times[k]),float(times[k]%604800),float(times[k]-times[0])]

def metrics(pred, target):
    errors = (np.maximum(pred, 0) - target) * 86400
    return {'MAE': float(np.mean(abs(errors))), 'MSE': float(np.mean(errors ** 2))}


def build_cache(data_dir, cache):
    import csv
    from collections import defaultdict
    users = defaultdict(list)
    with (data_dir / 'task_6_train.csv').open() as f:
        reader = csv.reader(f)
        next(reader)
        for user, item, timestamp, rating in reader:
            users[user].append(float(timestamp))
    X, y, V, vy, T, ty = [], [], [], [], [], []
    for times in users.values():
        times = np.asarray(times)
        gaps = np.diff(times)
        if (gaps < 0).any():
            raise ValueError('Training histories must be chronological')
        n = len(gaps)
        if n < 3:
            continue
        T.append(features(times, n - 1)); ty.append(gaps[-1] / 86400)
        V.append(features(times, n - 2)); vy.append(gaps[-2] / 86400)
        indices = np.unique(np.linspace(1, n - 3, min(5, max(0, n - 3)), dtype=int))
        for k in indices:
            X.append(features(times, k)); y.append(gaps[k] / 86400)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=np.asarray(X, dtype=np.float32), y=y,
                        V=np.asarray(V, dtype=np.float32), vy=vy,
                        T=np.asarray(T, dtype=np.float32), ty=ty)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=Path('experiments/artifacts/features.npz'))
    parser.add_argument('--data-dir', type=Path, default=Path('Assignment 1/assignment1_2026'))
    parser.add_argument('--output', type=Path, default=Path('experiments/artifacts'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.cache.exists():
        build_cache(args.data_dir, args.cache)
    z = np.load(args.cache)
    X,y,V,vy,T,ty = [z[k] for k in ['X','y','V','vy','T','ty']]
    vp, tp, results = {}, {}, {}
    trials=[('mae_seed42','MAE',42,False),('mae_seed17','MAE',17,False),('log_mae','MAE',42,True),('rmse','RMSE',42,False)]
    for name,loss,seed,log_target in trials:
        model=CatBoostRegressor(loss_function=loss,iterations=1000,depth=6,learning_rate=.05,random_seed=seed,thread_count=4,verbose=250,allow_writing_files=False)
        yy=np.log1p(y) if log_target else y
        vv=np.log1p(vy) if log_target else vy
        model.fit(X,yy,eval_set=(V,vv),early_stopping_rounds=75)
        pv,pt=model.predict(V),model.predict(T)
        if log_target:pv,pt=np.expm1(pv),np.expm1(pt)
        vp[name],tp[name]=np.maximum(0,pv),np.maximum(0,pt)
        results[name]={'selection':metrics(vp[name],vy),'evaluation':metrics(tp[name],ty),'iterations':model.get_best_iteration()+1}
        model.save_model(str(args.output/(name+'.cbm')))
        print(name,results[name],flush=True)
    vp['median'],tp['median']=V[:,6]/86400,T[:,6]/86400
    vp['seed_average']=(vp['mae_seed42']+vp['mae_seed17'])/2
    tp['seed_average']=(tp['mae_seed42']+tp['mae_seed17'])/2
    # Select weights using penultimate-gap targets only.
    best=(float('inf'),None)
    for a in np.linspace(0,1,11):
        for b in np.linspace(0,1-a,int(round((1-a)*10))+1):
            c=1-a-b
            pv=a*vp['seed_average']+b*vp['log_mae']+c*vp['rmse']
            score=metrics(pv,vy)['MAE']
            if score<best[0]:best=(score,(float(a),float(b),float(c)))
    a,b,c=best[1]
    pv=a*vp['seed_average']+b*vp['log_mae']+c*vp['rmse']
    pt=a*tp['seed_average']+b*tp['log_mae']+c*tp['rmse']
    results['selected_ensemble']={'weights_seed_average_log_mae_rmse':[a,b,c],'selection':metrics(pv,vy),'evaluation':metrics(pt,ty)}
    results['median']={'selection':metrics(vp['median'],vy),'evaluation':metrics(tp['median'],ty)}
    results['seed_average']={'selection':metrics(vp['seed_average'],vy),'evaluation':metrics(tp['seed_average'],ty)}
    (args.output/'results.json').write_text(json.dumps(results,indent=2))
    np.savez_compressed(args.output/'predictions.npz',**{'validation_'+k:v for k,v in vp.items()},**{'evaluation_'+k:v for k,v in tp.items()})
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
