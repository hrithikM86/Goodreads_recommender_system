"""Task 2 implicit-feedback ensemble; run from any directory with --data-dir."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from implicit.bpr import BayesianPersonalizedRanking
from implicit.als import AlternatingLeastSquares


def train_models(X):
    trials=[('bpr32',BayesianPersonalizedRanking(factors=32,iterations=100,regularization=.01,learning_rate=.01,random_state=42,num_threads=1)),
            ('bpr64',BayesianPersonalizedRanking(factors=64,iterations=200,regularization=.01,learning_rate=.03,random_state=17,num_threads=1)),
            ('als64',AlternatingLeastSquares(factors=64,iterations=20,regularization=.1,alpha=20,random_state=42,num_threads=2)),
            ('als128',AlternatingLeastSquares(factors=128,iterations=20,regularization=.1,alpha=40,random_state=42,num_threads=2))]
    models={}
    for name,model in trials:
        print('Training',name,flush=True);start=time.time();model.fit(X,show_progress=False)
        models[name]=model;print(name,'seconds',round(time.time()-start,1),flush=True)
    return models


def features(X,models,qu,qi):
    pop=np.asarray(X.sum(axis=0)).ravel();U,I=X.shape
    out={"popularity":np.log1p(pop[np.maximum(qi,0)])}
    out['popularity'][qi<0]=0
    valid=(qu>=0)&(qi>=0)
    for name,m in models.items():
        score=np.zeros(len(qu));score[valid]=np.einsum('ij,ij->i',m.user_factors[qu[valid]],m.item_factors[qi[valid]])
        out[name]=score
    # Item-item cosine similarities, considering only observed training interactions.
    G=X.T.tocsr();groups={}
    for q,key in enumerate(qi):
        if valid[q]:groups.setdefault(key,[]).append(q)
    keys=np.array(list(groups));top=np.zeros(len(qu));total=np.zeros(len(qu));jacc=np.zeros(len(qu))
    for offset in range(0,len(keys),64):
        batch=keys[offset:offset+64];co=(G[batch]@X).toarray()
        for row,key in enumerate(batch):
            cosine=co[row]/np.sqrt(np.maximum(pop[key]*pop,1));cosine[key]=0
            jac=co[row]/np.maximum(pop[key]+pop-co[row],1);jac[key]=0
            for q in groups[key]:
                lo,hi=X.indptr[qu[q]:qu[q]+2];hist=X.indices[lo:hi];w=cosine[hist]
                if len(w):
                    top[q]=w.max();total[q]=np.sort(w)[-10:].sum();jacc[q]=jac[hist].max()
        if offset%10000<64:print('Similarity items',offset,'/',len(keys),flush=True)
    out.update(cosine_max=top,cosine_top10=total,jaccard_max=jacc)
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
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(args.data_dir/'tasks_1_2_train.csv',usecols=['user','item'])
    u,users=pd.factorize(data.user);i,items=pd.factorize(data.item);U,I=len(users),len(items)
    full=coo_matrix((np.ones(len(u),dtype=np.float32),(u,i)),shape=(U,I)).tocsr();full.data[:]=1
    order=np.random.RandomState(42).permutation(len(u));a=int(.9*len(u));b=int(.95*len(u));idx=order[:a];hold=order[a:]
    X=coo_matrix((np.ones(len(idx),dtype=np.float32),(u[idx],i[idx])),shape=(U,I)).tocsr();X.data[:]=1
    rng=np.random.RandomState(43);negative=[]
    histories=[set(full.indices[full.indptr[k]:full.indptr[k+1]]) for k in range(U)]
    for user in u[hold]:
        item=rng.randint(I)
        while item in histories[user]:item=rng.randint(I)
        negative.append(item)
    qu=np.r_[u[hold],u[hold]];qi=np.r_[i[hold],negative]
    models=train_models(X);names,scores=features(X,models,qu,qi)
    d=scores[:len(hold)]-scores[len(hold):];v=d[:b-a];t=d[b-a:]
    # Ties count as half credit for local pairwise metrics.
    def accuracy(m):return float(np.mean((m>0)+.5*(m==0)))
    results={name:{'selection_accuracy':accuracy(v[:,k]),'evaluation_accuracy':accuracy(t[:,k])} for k,name in enumerate(names)}
    weights,selection=fit_blend(v)
    results['ensemble']={'selection_accuracy':accuracy(v@weights),'evaluation_accuracy':accuracy(t@weights),'internal_selection_accuracy':selection[0],'C':selection[1]}
    report={'split_sizes':[a,b-a,len(u)-b],'negative_sampling':'uniform item, excluding all known user interactions','feature_names':names,'weights':weights.tolist(),'results':results,'seed':42}
    print(json.dumps(report,indent=2),flush=True)
    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(args.output_dir/'validation_margins.npz',selection=v,evaluation=t)
    if args.predict:
        models=train_models(full)
        test=pd.read_csv(args.data_dir/'task_2_test.csv')
        um=dict(zip(users,range(U)));im=dict(zip(items,range(I)))
        testu=np.array([um.get(x,-1) for x in test.user]);i1=np.array([im.get(x,-1) for x in test.item1]);i2=np.array([im.get(x,-1) for x in test.item2])
        newnames,f=features(full,models,np.r_[testu,testu],np.r_[i1,i2]);assert names==newnames
        margin=(f[:len(test)]-f[len(test):])@weights
        chosen=np.where(margin>=0,test.item1,test.item2)
        # Deterministic popularity fallback whenever the user or either item is unseen.
        missing=(testu<0)|(i1<0)|(i2<0)
        pop=np.asarray(full.sum(axis=0)).ravel();p1=np.where(i1>=0,pop[np.maximum(i1,0)],0);p2=np.where(i2>=0,pop[np.maximum(i2,0)],0)
        chosen[missing]=np.where(p1[missing]>=p2[missing],test.item1.to_numpy()[missing],test.item2.to_numpy()[missing])
        test['which']=chosen
        assert len(test)==20000 and ((test.which==test.item1)|(test.which==test.item2)).all()
        test.to_csv(args.output_dir/'task_2_predictions.csv',index=False)
        print('Saved',len(test),'predictions; fallback rows',int(missing.sum()),flush=True)

if __name__=='__main__':main()
