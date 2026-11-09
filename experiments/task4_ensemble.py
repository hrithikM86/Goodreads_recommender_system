"""Task 4 temporal implicit-feedback ensemble; run from any directory with --data-dir."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,argparse,json,time,hashlib
from types import SimpleNamespace
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from implicit.bpr import BayesianPersonalizedRanking
from implicit.als import AlternatingLeastSquares


def train_models(X, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_hash=hashlib.sha256(X.indptr.tobytes()+X.indices.tobytes()+X.data.tobytes()).hexdigest()
    trials=[('bpr32',BayesianPersonalizedRanking(factors=32,iterations=100,regularization=.01,learning_rate=.01,random_state=42,num_threads=1)),
            ('bpr64',BayesianPersonalizedRanking(factors=64,iterations=200,regularization=.01,learning_rate=.03,random_state=17,num_threads=1)),
            ('als64',AlternatingLeastSquares(factors=64,iterations=20,regularization=.1,alpha=20,random_state=42,num_threads=2)),
            ('als128',AlternatingLeastSquares(factors=128,iterations=20,regularization=.1,alpha=40,random_state=42,num_threads=2))]
    models={}
    for name,model in trials:
        params={k:getattr(model,k,None) for k in ['factors','iterations','regularization','learning_rate','alpha','random_state','verify_negative_samples','num_threads']}
        params['implicit_version']=__import__('implicit').__version__
        signature=hashlib.sha256((matrix_hash+json.dumps(params,sort_keys=True)).encode()).hexdigest()[:24]
        cache=cache_dir/(name+'_'+signature+'.npz')
        if cache.exists():
            with np.load(cache) as saved:
                models[name]=SimpleNamespace(user_factors=saved['user_factors'],item_factors=saved['item_factors'])
            print('Loaded cached factors',name,flush=True)
            continue
        print('Training',name,flush=True);start=time.time();model.fit(X,show_progress=False)
        models[name]=model
        np.savez_compressed(cache,user_factors=model.user_factors,item_factors=model.item_factors)
        print(name,'seconds',round(time.time()-start,1),flush=True)
    return models


def time_period(t,days):
    good=(t>=0)&(t<=4102444800)
    return np.where(good,np.floor(np.clip(t,0,4102444800)/(days*86400)),-1).astype(np.int64)


def train_time_models(u,i,t,shape,cache_dir):
    output=[]
    for days in [180,365]:
        keys=(u+1)*1000000000+time_period(t,days);code,index=pd.factorize(keys)
        B=coo_matrix((np.ones(len(u),dtype=np.float32),(code,i)),shape=(len(index),shape[1])).tocsr();B.data[:]=1
        signature=hashlib.sha256(B.indptr.tobytes()+B.indices.tobytes()+B.data.tobytes()+str(B.shape).encode()+('factors64_iter20_reg0.1_alpha20_seed42_threads2_'+__import__('implicit').__version__).encode()).hexdigest()[:24]
        path=cache_dir/('timeALS'+str(days)+'_'+signature+'.npz')
        if path.exists():
            with np.load(path) as f:m=SimpleNamespace(user_factors=f['user_factors'],item_factors=f['item_factors'])
            print('Loaded temporal ALS',days,flush=True)
        else:
            print('Training temporal ALS',days,'days; user-periods',len(index),flush=True)
            m=AlternatingLeastSquares(factors=64,iterations=20,regularization=.1,alpha=20,random_state=42,num_threads=2)
            m.fit(B,show_progress=False)
            np.savez_compressed(path,user_factors=m.user_factors,item_factors=m.item_factors)
        output.append(('als_user'+str(days),days,pd.Index(index),m))
    return output


def features(X,models,qu,qi,Times,qt,time_models=None):
    pop=np.asarray(X.sum(axis=0)).ravel();U,I=X.shape
    out={"popularity":np.log1p(pop[np.maximum(qi,0)])}
    out['popularity'][qi<0]=0
    valid=(qu>=0)&(qi>=0)
    seen=np.zeros(len(qu))
    seen[valid]=np.asarray(X[qu[valid],qi[valid]]).ravel()>0
    out['already_seen']=seen
    for name,m in models.items():
        score=np.zeros(len(qu));score[valid]=np.einsum('ij,ij->i',m.user_factors[qu[valid]],m.item_factors[qi[valid]])
        out[name]=score
    # Item-item cosine similarities, considering only observed training interactions.
    G=X.T.tocsr();groups={}
    for q,key in enumerate(qi):
        if valid[q]:groups.setdefault(key,[]).append(q)
    keys=np.array(list(groups));top=np.zeros(len(qu));total=np.zeros(len(qu));jacc=np.zeros(len(qu))
    temporal={name:np.zeros(len(qu)) for name in ['cosine_time30','cosine_time180','cosine_time365','cosine_top10_time180','item_nearest_time','item_age']}
    for days in [30,90,365]:
        temporal['item_count'+str(days)]=np.zeros(len(qu));temporal['item_fraction'+str(days)]=np.zeros(len(qu))
    TC=Times.tocsc()
    for item,qs in groups.items():
        lo,hi=TC.indptr[item:item+2];dates=TC.data[lo:hi];dates=np.sort(dates[(dates>=0)&(dates<=4102444800)])
        if not len(dates):continue
        dates_query=qt[qs]
        pos=np.searchsorted(dates,dates_query)
        nearest=np.minimum(abs(dates_query-dates[np.clip(pos,0,len(dates)-1)]),abs(dates_query-dates[np.clip(pos-1,0,len(dates)-1)]))/86400
        temporal['item_nearest_time'][qs]=-np.log1p(nearest)
        diff=(dates_query-dates[0])/86400;temporal['item_age'][qs]=np.sign(diff)*np.log1p(abs(diff))
        for days in [30,90,365]:
            count=np.searchsorted(dates,dates_query+days*86400,side='right')-np.searchsorted(dates,dates_query-days*86400,side='left')
            temporal['item_count'+str(days)][qs]=np.log1p(count)
            temporal['item_fraction'+str(days)][qs]=count/max(pop[item],1)
    if time_models:
        for name,periods,index,m in time_models:
            keys_query=(qu+1)*1000000000+time_period(qt,periods)
            code=index.get_indexer(keys_query);ok=valid&(code>=0)
            score=out['als64'].copy()
            score[ok]=np.einsum('ij,ij->i',m.user_factors[code[ok]],m.item_factors[qi[ok]])
            out[name]=score
    for offset in range(0,len(keys),64):
        batch=keys[offset:offset+64];co=(G[batch]@X).toarray()
        for row,key in enumerate(batch):
            cosine=co[row]/np.sqrt(np.maximum(pop[key]*pop,1));cosine[key]=0
            jac=co[row]/np.maximum(pop[key]+pop-co[row],1);jac[key]=0
            for q in groups[key]:
                lo,hi=X.indptr[qu[q]:qu[q]+2];hist=X.indices[lo:hi];w=cosine[hist]
                if len(w):
                    top[q]=w.max();total[q]=np.sort(w)[-10:].sum();jacc[q]=jac[hist].max()
                    distance=abs(Times.data[lo:hi]-qt[q])/86400
                    for days in [30,180,365]:
                        tw=w*np.exp(-distance/days)
                        temporal['cosine_time'+str(days)][q]=tw.max()
                        if days==180:temporal['cosine_top10_time180'][q]=np.sort(tw)[-10:].sum()
        if offset%10000<64:print('Similarity items',offset,'/',len(keys),flush=True)
    out.update(cosine_max=top,cosine_top10=total,jaccard_max=jacc)
    out.update(temporal)
    return list(out),np.column_stack(list(out.values()))


def fit_blend(differences):
    idx=np.random.RandomState(17).permutation(len(differences));cut=int(.8*len(idx));best=None
    for C in [.01,.1,1.]:
        x=differences[idx[:cut]];scale=np.std(x,axis=0);scale[scale<1e-8]=1
        m=LogisticRegression(C=C,fit_intercept=False,max_iter=1000)
        m.fit(np.vstack([x/scale,-x/scale]),np.r_[np.ones(len(x)),np.zeros(len(x))])
        accuracy=float(np.mean((differences[idx[cut:]]/scale)@m.coef_[0]>0))
        if best is None or accuracy>best[0]:best=(accuracy,C)
    scale=np.std(differences,axis=0);scale[scale<1e-8]=1
    m=LogisticRegression(C=best[1],fit_intercept=False,max_iter=1000)
    m.fit(np.vstack([differences/scale,-differences/scale]),np.r_[np.ones(len(differences)),np.zeros(len(differences))])
    return m.coef_[0]/scale,best


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/task2'))
    parser.add_argument('--predict',action='store_true')
    parser.add_argument('--stack',action='store_true',help='Compare a shallow antisymmetric pair stacker with the linear combiner')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(args.data_dir/'tasks_3_4_train.csv',usecols=['user','item','timestamp'])
    timestamps=data.timestamp.to_numpy(dtype=float)
    u,users=pd.factorize(data.user);i,items=pd.factorize(data.item);U,I=len(users),len(items)
    full=coo_matrix((np.ones(len(u),dtype=np.float32),(u,i)),shape=(U,I)).tocsr();full.data[:]=1
    order=np.random.RandomState(42).permutation(len(u));a=int(.9*len(u));b=int(.95*len(u));idx=order[:a];hold=order[a:]
    X=coo_matrix((np.ones(len(idx),dtype=np.float32),(u[idx],i[idx])),shape=(U,I)).tocsr();X.data[:]=1
    rng=np.random.RandomState(43);negative=[]
    for positive in i[hold]:
        # Draw an interaction uniformly, making item probability proportional to training popularity.
        # A negative may already be in the user's history, as occurs in the supplied test pairs.
        item=i[idx[rng.randint(len(idx))]]
        while item==positive:item=i[idx[rng.randint(len(idx))]]
        negative.append(item)
    qu=np.r_[u[hold],u[hold]];qi=np.r_[i[hold],negative]
    models=train_models(X,args.output_dir/'factors')
    time_models=train_time_models(u[idx],i[idx],timestamps[idx],X.shape,args.output_dir/'factors')
    Times=coo_matrix((timestamps[idx],(u[idx],i[idx])),shape=X.shape).tocsr()
    query_time=np.r_[timestamps[hold],timestamps[hold]]
    names,scores=features(X,models,qu,qi,Times,query_time,time_models)
    d=scores[:len(hold)]-scores[len(hold):];v=d[:b-a];t=d[b-a:]
    # Ties count as half credit for local pairwise metrics.
    def accuracy(m):return float(np.mean((m>0)+.5*(m==0)))
    results={name:{'selection_accuracy':accuracy(v[:,k]),'evaluation_accuracy':accuracy(t[:,k])} for k,name in enumerate(names)}
    weights,selection=fit_blend(v)
    results['ensemble']={'selection_accuracy':accuracy(v@weights),'evaluation_accuracy':accuracy(t@weights),'internal_selection_accuracy':selection[0],'C':selection[1]}
    report={'split_sizes':[a,b-a,len(u)-b],'negative_sampling':'popularity weighted using training-partition interactions; previously seen negatives allowed','feature_names':names,'weights':weights.tolist(),'results':results,'seed':42}
    print(json.dumps(report,indent=2),flush=True)
    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(args.output_dir/'validation_margins.npz',selection=v,evaluation=t)
    np.savez_compressed(args.output_dir/'pair_features.npz',scores=scores,users=qu,items=qi)
    report['seen_negative_rows']=int(np.count_nonzero(scores[len(hold):,names.index('already_seen')]))
    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2))
    stacker=None
    if args.stack:
        from task4_stacking import train_stacker,margin as stack_margin
        sums=scores[:len(hold)]+scores[len(hold):]
        candidate,stack_report=train_stacker(v,sums[:b-a],args.output_dir)
        stack_report['linear_internal_selection_accuracy']=report['results']['ensemble']['internal_selection_accuracy']
        stack_report['linear_evaluation_accuracy']=report['results']['ensemble']['evaluation_accuracy']
        stack_report['selected']='stacker' if stack_report['internal_selection_accuracy']>stack_report['linear_internal_selection_accuracy'] else 'linear'
        sm=stack_margin(candidate,t,sums[b-a:])
        stack_report['evaluation_accuracy']=accuracy(sm)
        if stack_report['selected']=='stacker':stacker=candidate
        report['stacking']=stack_report
        report['selected_model']=stack_report['selected']
        report['selected_evaluation_accuracy']=stack_report['evaluation_accuracy'] if stacker is not None else stack_report['linear_evaluation_accuracy']
        (args.output_dir/'stacking.json').write_text(json.dumps(stack_report,indent=2))
        (args.output_dir/'results.json').write_text(json.dumps(report,indent=2))
        print('Selected Task 4 combiner:',stack_report,flush=True)
    if args.predict:
        models=train_models(full,args.output_dir/'factors')
        test=pd.read_csv(args.data_dir/'task_4_test.csv')
        um=dict(zip(users,range(U)));im=dict(zip(items,range(I)))
        testu=np.array([um.get(x,-1) for x in test.user]);i1=np.array([im.get(x,-1) for x in test.item1]);i2=np.array([im.get(x,-1) for x in test.item2])
        time_models=train_time_models(u,i,timestamps,full.shape,args.output_dir/'factors')
        Times=coo_matrix((timestamps,(u,i)),shape=full.shape).tocsr()
        query_time=np.r_[test.timestamp.to_numpy(),test.timestamp.to_numpy()]
        newnames,f=features(full,models,np.r_[testu,testu],np.r_[i1,i2],Times,query_time,time_models);assert names==newnames
        np.savez_compressed(args.output_dir/'test_pair_features.npz',scores=f,users=np.r_[testu,testu],items=np.r_[i1,i2])
        margin=(f[:len(test)]-f[len(test):])@weights
        if stacker is not None:
            margin=stack_margin(stacker,f[:len(test)]-f[len(test):],f[:len(test)]+f[len(test):])
        chosen=np.where(margin>=0,test.item1,test.item2)
        # Deterministic popularity fallback whenever the user or either item is unseen.
        missing=(testu<0)|(i1<0)|(i2<0)
        pop=np.asarray(full.sum(axis=0)).ravel();p1=np.where(i1>=0,pop[np.maximum(i1,0)],0);p2=np.where(i2>=0,pop[np.maximum(i2,0)],0)
        chosen[missing]=np.where(p1[missing]>=p2[missing],test.item1.to_numpy()[missing],test.item2.to_numpy()[missing])
        test['which']=chosen
        assert len(test)==20000 and ((test.which==test.item1)|(test.which==test.item2)).all()
        test.to_csv(args.output_dir/'task_4_predictions.csv',index=False)
        print('Saved',len(test),'predictions; fallback rows',int(missing.sum()),flush=True)

if __name__=='__main__':main()
