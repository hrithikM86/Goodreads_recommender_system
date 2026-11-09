"""Sequential next-book ensemble. Uses CSV order and user/item IDs, never timestamp values."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from implicit.bpr import BayesianPersonalizedRanking
from implicit.als import AlternatingLeastSquares


def cached_fit(name,model,X,folder):
    folder.mkdir(parents=True,exist_ok=True)
    params={k:getattr(model,k,None) for k in ['factors','iterations','regularization','learning_rate','alpha','random_state','num_threads','verify_negative_samples']}
    params['version']=__import__('implicit').__version__
    key=hashlib.sha256(X.indptr.tobytes()+X.indices.tobytes()+X.data.tobytes()+str(X.shape).encode()+json.dumps(params,sort_keys=True).encode()).hexdigest()[:24]
    path=folder/(name+'_'+key+'.npz')
    if path.exists():
        with np.load(path) as z:result=SimpleNamespace(user_factors=z['user_factors'],item_factors=z['item_factors'])
        print('Loaded',name,flush=True);return result
    print('Fitting',name,flush=True);start=time.time();model.fit(X,show_progress=False)
    np.savez_compressed(path,user_factors=model.user_factors,item_factors=model.item_factors)
    print(name,'seconds',round(time.time()-start,1),flush=True);return model


def prepare(history,I,folder):
    U=len(history);u=np.repeat(np.arange(U),[len(h) for h in history]);i=np.concatenate(history)
    X=coo_matrix((np.ones(len(i),dtype=np.float32),(u,i)),shape=(U,I)).tocsr();X.data[:]=1
    prev=np.concatenate([h[:-1] for h in history if len(h)>1]);nxt=np.concatenate([h[1:] for h in history if len(h)>1])
    T=coo_matrix((np.ones(len(prev),dtype=np.float32),(prev,nxt)),shape=(I,I)).tocsr()
    specs=[('bpr32',BayesianPersonalizedRanking(factors=32,iterations=100,learning_rate=.01,regularization=.01,random_state=42,num_threads=1)),('bpr64',BayesianPersonalizedRanking(factors=64,iterations=150,learning_rate=.03,regularization=.01,random_state=17,num_threads=1)),('als64',AlternatingLeastSquares(factors=64,iterations=20,alpha=20,regularization=.1,random_state=42,num_threads=2))]
    models={name:cached_fit(name,m,X,folder) for name,m in specs}
    models['transition_als32']=cached_fit('transition_als32',AlternatingLeastSquares(factors=32,iterations=20,alpha=10,regularization=.1,random_state=42,num_threads=2),T,folder)
    return X,T,models


def features(X,T,models,contexts,qu,qi):
    I=X.shape[1];N=len(contexts);assert len(qu)==2*N and len(qi)==2*N
    ctx=np.r_[np.arange(N),np.arange(N)];length=np.array([len(h) for h in contexts]);pop=np.asarray(X.sum(axis=0)).ravel();valid=(qi>=0)&(qu>=0);known=valid.copy();known[valid]&=pop[qi[valid]]>0
    row=np.repeat(np.arange(N),length);items=np.concatenate(contexts)
    C=coo_matrix((np.ones(len(items),dtype=np.float32),(row,items)),shape=(N,I)).tocsr();C.data[:]=1
    seen=np.zeros(len(qi));seen[valid]=np.asarray(C[ctx[valid],qi[valid]]).ravel()>0
    popularity=np.zeros(len(qi));popularity[valid]=np.log1p(pop[qi[valid]])
    out={'popularity':popularity,'already_seen':seen,'cold_item':(~known).astype(float),'history_count':np.log1p(length[ctx])}
    for name in ['bpr32','bpr64']:
        m=models[name];score=np.zeros(len(qi));score[known]=np.einsum('ij,ij->i',m.user_factors[qu[known]],m.item_factors[qi[known]]);out[name]=score
    # Fold in each available query history using frozen item factors. Evaluation histories
    # may include the penultimate event, while held-out final targets remain absent.
    base=models['als64'];als=AlternatingLeastSquares(factors=64,iterations=20,alpha=20,regularization=.1,random_state=42,num_threads=2);als.item_factors=base.item_factors
    uf=als.recalculate_user(np.arange(N),C)
    score=np.zeros(len(qi));score[known]=np.einsum('ij,ij->i',uf[ctx[known]],base.item_factors[qi[known]]);out['als_history']=score
    recent=[h[-5:] for h in contexts];rr=np.repeat(np.arange(N),[len(h) for h in recent]);ri=np.concatenate(recent);CR=coo_matrix((np.ones(len(ri),dtype=np.float32),(rr,ri)),shape=C.shape).tocsr()
    uf=als.recalculate_user(np.arange(N),CR);score=np.zeros(len(qi));score[known]=np.einsum('ij,ij->i',uf[ctx[known]],base.item_factors[qi[known]]);out['als_recent5']=score
    last=np.array([h[-1] for h in contexts]);prev=last[ctx];trans=models['transition_als32'];score=np.zeros(len(qi));score[known]=np.einsum('ij,ij->i',trans.user_factors[prev[known]],trans.item_factors[qi[known]]);out['factorized_markov_last']=score
    meanprev=np.zeros((N,trans.user_factors.shape[1]))
    for n,h in enumerate(contexts):
        past=h[-3:][::-1];w=.7**np.arange(len(past));meanprev[n]=w@trans.user_factors[past]/w.sum()
    score=np.zeros(len(qi));score[known]=np.einsum('ij,ij->i',meanprev[ctx[known]],trans.item_factors[qi[known]]);out['factorized_markov_recent3']=score
    fwd=np.zeros(len(qi));rev=np.zeros(len(qi));fwd[valid]=np.asarray(T[prev[valid],qi[valid]]).ravel();rev[valid]=np.asarray(T[qi[valid],prev[valid]]).ravel()
    out['markov_count']=np.log1p(fwd);out['markov_reverse']=np.log1p(rev)
    lift=np.zeros(len(qi));lift[valid]=np.log1p(fwd[valid]/(5*pop[qi[valid]]/max(pop.sum(),1)+1e-9));out['markov_lift']=lift
    # Global tail popularity uses only the ordered histories fitted to the graph.
    # Caller supplies these through models, without absolute time features.
    for k,counts in models['tail_popularity'].items():
        score=np.zeros(len(qi));fraction=np.zeros(len(qi));score[valid]=np.log1p(counts[qi[valid]]);fraction[valid]=counts[qi[valid]]/np.maximum(pop[qi[valid]],1)
        out['tail_pop'+str(k)]=score;out['tail_fraction'+str(k)]=fraction
    out.update(cosine_max=np.zeros(len(qi)),cosine_top10=np.zeros(len(qi)),cosine_last=np.zeros(len(qi)),cosine_recent3=np.zeros(len(qi)),cosine_recent5=np.zeros(len(qi)))
    groups={}
    for q,key in enumerate(qi):
        if known[q]:groups.setdefault(key,[]).append(q)
    keys=np.array(list(groups));G=X.T.tocsr()
    for offset in range(0,len(keys),64):
        batch=keys[offset:offset+64];co=(G[batch]@X).tocsr();co.sort_indices()
        for j,key in enumerate(batch):
            lo,hi=co.indptr[j:j+2];neighbors=co.indices[lo:hi];values=co.data[lo:hi]/np.sqrt(np.maximum(pop[key]*pop[neighbors],1))
            for q in groups[key]:
                h=contexts[ctx[q]];pos=np.searchsorted(neighbors,h);ok=pos<len(neighbors);ok[ok]&=neighbors[pos[ok]]==h[ok];w=np.zeros(len(h));w[ok]=values[pos[ok]];w[h==key]=0
                out['cosine_max'][q]=w.max();out['cosine_top10'][q]=np.sort(w)[-10:].sum();out['cosine_last'][q]=w[-1]
                for k in [3,5]:
                    rw=w[-k:][::-1];decay=.7**np.arange(len(rw));out['cosine_recent'+str(k)][q]=rw@decay/decay.sum()
        if offset%20000<64:print('Similarity candidate items',offset,'/',len(keys),flush=True)
    return list(out),np.column_stack(list(out.values()))


def fit_blend(v):
    ids=np.random.RandomState(17).permutation(len(v));cut=int(.8*len(ids));best=None
    for C in [.01,.1,1.]:
        x=v[ids[:cut]];scale=x.std(axis=0);scale[scale<1e-8]=1
        m=LogisticRegression(C=C,fit_intercept=False,max_iter=1000);m.fit(np.vstack([x/scale,-x/scale]),np.r_[np.ones(len(x)),np.zeros(len(x))]);score=float(np.mean((v[ids[cut:]]/scale)@m.coef_[0]>0))
        if best is None or score>best[0]:best=(score,C)
    scale=v.std(axis=0);scale[scale<1e-8]=1;m=LogisticRegression(C=best[1],fit_intercept=False,max_iter=1000);m.fit(np.vstack([v/scale,-v/scale]),np.r_[np.ones(len(v)),np.zeros(len(v))]);return m.coef_[0]/scale,best


def tail_pop(history,I):return {k:np.bincount(np.concatenate([h[-k:] for h in history]),minlength=I) for k in [1,5]}
def accuracy(x):return float(np.mean((x>0)+.5*(x==0)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,default=Path('artifacts/task5'));ap.add_argument('--predict',action='store_true');ap.add_argument('--stack',action='store_true');args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(args.data_dir/'task_5_train.csv',usecols=['user','item']);u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);U=len(users);I=len(items)
    if np.any(np.diff(u)<0):raise ValueError('Expected contiguous ordered user histories')
    starts=np.searchsorted(u,np.arange(U+1));history=[i[starts[j]:starts[j+1]] for j in range(U)];assert all(len(h)>=4 for h in history)
    train=[h[:-2] for h in history];contexts=train+[h[:-1] for h in history];positive=np.r_[[h[-2] for h in history],[h[-1] for h in history]];query_users=np.r_[np.arange(U),np.arange(U)]
    X,T,models=prepare(train,I,args.output_dir/'factors');models['tail_popularity']=tail_pop(train,I)
    rng=np.random.RandomState(43);pool=np.concatenate(train);negative=np.empty(len(positive),dtype=int)
    for k,pos in enumerate(positive):
        choice=pool[rng.randint(len(pool))]
        while choice==pos:choice=pool[rng.randint(len(pool))]
        negative[k]=choice
    qu=np.r_[query_users,query_users];qi=np.r_[positive,negative]
    names,scores=features(X,T,models,contexts,qu,qi);N=len(positive);d=scores[:N]-scores[N:];v=d[:U];e=d[U:];weights,choice=fit_blend(v)
    results={name:{'selection_accuracy':accuracy(v[:,j]),'evaluation_accuracy':accuracy(e[:,j])} for j,name in enumerate(names)};results['ensemble']={'selection_accuracy':accuracy(v@weights),'evaluation_accuracy':accuracy(e@weights),'internal_selection_accuracy':choice[0],'C':choice[1]}
    report={'users':U,'items':I,'training_interactions':int(X.nnz),'selection_pairs':U,'evaluation_pairs':U,'negative_sampling':'training-interaction popularity weighted; previously seen candidates allowed','feature_names':names,'weights':weights.tolist(),'results':results,'ordering_only':True}
    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2));np.savez_compressed(args.output_dir/'pair_features.npz',scores=scores,users=qu,items=qi);np.savez_compressed(args.output_dir/'validation_margins.npz',selection=v,evaluation=e)
    print(json.dumps(report,indent=2),flush=True)
    stacker=None
    if args.stack:
        from task4_stacking import train_stacker,margin as stack_margin
        sums=scores[:N]+scores[N:]
        candidate,stack_report=train_stacker(v,sums[:U],args.output_dir)
        stack_report['linear_internal_selection_accuracy']=report['results']['ensemble']['internal_selection_accuracy']
        stack_report['linear_evaluation_accuracy']=report['results']['ensemble']['evaluation_accuracy']
        stack_report['selected']='stacker' if stack_report['internal_selection_accuracy']>stack_report['linear_internal_selection_accuracy'] else 'linear'
        sm=stack_margin(candidate,e,sums[U:]);stack_report['evaluation_accuracy']=accuracy(sm)
        if stack_report['selected']=='stacker':stacker=candidate
        report['stacking']=stack_report;report['selected_model']=stack_report['selected']
        report['selected_evaluation_accuracy']=stack_report['evaluation_accuracy'] if stacker is not None else stack_report['linear_evaluation_accuracy']
        (args.output_dir/'stacking.json').write_text(json.dumps(stack_report,indent=2));(args.output_dir/'results.json').write_text(json.dumps(report,indent=2))
        print('Selected Task 5 combiner:',stack_report,flush=True)
    if args.predict:
        X,T,models=prepare(history,I,args.output_dir/'factors');models['tail_popularity']=tail_pop(history,I)
        test=pd.read_csv(args.data_dir/'task_5_test.csv');um=dict(zip(users,range(U)));im=dict(zip(items,range(I)));tu=np.array([um.get(x,-1) for x in test.user]);i1=np.array([im.get(x,-1) for x in test.item1]);i2=np.array([im.get(x,-1) for x in test.item2]);assert (tu>=0).all()
        ctx=[history[j] for j in tu];nn,f=features(X,T,models,ctx,np.r_[tu,tu],np.r_[i1,i2]);assert names==nn;n=len(test);margin=(f[:n]-f[n:])@weights
        if stacker is not None:margin=stack_margin(stacker,f[:n]-f[n:],f[:n]+f[n:])
        test['which']=np.where(margin>=0,test.item1,test.item2)
        assert ((test.which==test.item1)|(test.which==test.item2)).all()
        np.savez_compressed(args.output_dir/'test_pair_features.npz',scores=f,users=np.r_[tu,tu],items=np.r_[i1,i2]);test.to_csv(args.output_dir/'task_5_predictions.csv',index=False)
        print('Saved',n,'Task 5 predictions',flush=True)

if __name__=='__main__':main()
