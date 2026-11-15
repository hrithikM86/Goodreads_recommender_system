"""Independent optimization, leakage and exported prediction checks for Tasks 1/6."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,ctypes,json,hashlib,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix,load_npz
from catboost import CatBoostRegressor
from task6_improve import event_features

def numeric(out):
    out.mkdir(parents=True,exist_ok=True);so=out/'check.so';subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task1_svdpp_kernel.cpp')),'-o',str(so)],check=True);lib=ctypes.CDLL(str(so));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.svdpp_train.argtypes=[ctypes.c_int]*2+[ip,ip,fp,ip,ctypes.c_float]+[fp]*5+[ctypes.c_int]*3+[ctypes.c_float]*2+[ctypes.c_int];rng=np.random.RandomState(4);U,I,k=2,4,3;ptr=np.array([0,3,5],np.int32);items=np.array([0,1,2,1,3],np.int32);y=np.array([4,3,5,2,4],np.float32);count=np.bincount(items,minlength=I).astype(np.int32);P=rng.normal(0,.1,(U,k)).astype(np.float32);Q=rng.normal(0,.1,(I,k)).astype(np.float32);Y=rng.normal(0,.1,(I,k)).astype(np.float32);bu=np.zeros(U,np.float32);bi=np.zeros(I,np.float32);p,q,z,ub,ib=[v.copy() for v in [P,Q,Y,bu,bi]];eta=.007;reg=.1;mu=3.6
    for u in range(U):
        history=items[ptr[u]:ptr[u+1]];norm=1/np.sqrt(len(history))
        for row in range(ptr[u],ptr[u+1]):
            j=items[row];s=z[history].sum(0)*norm;eff=p[u]+s;oldq=q[j].copy();error=y[row]-(mu+ub[u]+ib[j]+eff@oldq);ub[u]+=eta*(error-10/len(history)*ub[u]);ib[j]+=eta*(error-10/count[j]*ib[j]);p[u]+=eta*(error*oldq-reg*p[u]);q[j]+=eta*(error*eff-reg*oldq);z[history]+=eta*(error*oldq*norm-reg*z[history])
    lib.svdpp_train(U,k,ptr,items,y,count,mu,P,Q,Y,bu,bi,0,1,42,eta,reg,0);assert all(np.allclose(a,b,atol=1e-6) for a,b in zip([P,Q,Y,bu,bi],[p,q,z,ub,ib]))
    item=np.array([0,0,0,0]);t=np.array([10,20,30,40.])*86400;allowed=np.ones(4,bool);args=(np.array([0]),np.array([0.]),np.array([np.log1p(10)]),np.array([1]));a=event_features(item,t,allowed,*args);changed=t.copy();changed[1]=200*86400;b=event_features(item,changed,allowed,*args);assert np.allclose(a,b,equal_nan=True);assert np.isclose(a[0,6],np.log1p(30));allowed[1]=False;assert np.allclose(event_features(item,t,allowed,*args),event_features(item,changed,allowed,*args),equal_nan=True);single=event_features(np.array([0]),np.array([20.*86400]),np.ones(1,bool),np.array([0]),np.array([0.]),np.array([1.]),np.array([0]));assert single[0,0]==0 and single[0,-1]==1 and np.isnan(single[0,3:10]).all()
    print('PASS: cached SVD++ updates match naive history updates; Task6 target perturbation/self-removal/singleton checks.',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--numerical-only',action='store_true');ap.add_argument('--task',choices=['1','6']);ap.add_argument('--data-dir',type=Path);ap.add_argument('--legacy',type=Path);ap.add_argument('--accepted',type=Path);a=ap.parse_args();numeric(a.out/'checks')
    if a.numerical_only:return
    if a.task=='1':check1(a)
    elif a.task=='6':check6(a)

def check1(a):
    from task1_improve import dataset,clip,WEIGHTS
    from task1_improve_stack import features
    summary=json.loads((a.out/'results.json').read_text());X,names,ref,test,n=features(a,True);assert names==summary['features'];preds={}
    for name in summary['weights']:
        model=CatBoostRegressor();model.load_model(str(a.out/(name+'_full.cbm')));preds[name]=clip(model.predict(X))
    new=clip(sum(summary['weights'][k]*v for k,v in preds.items()));blend=.5*new+.5*ref;audit={}
    for label,pred in [('new',new),('blend50',blend)]:
        path=a.out/label/'task_1_predictions.csv';frame=pd.read_csv(path);pd.testing.assert_frame_equal(frame.drop(columns='rating'),test.drop(columns='rating'));assert len(frame)==20000 and frame.rating.between(1,5).all() and np.isfinite(frame.rating).all();assert np.allclose(frame.rating,pred,atol=1e-10);audit[label]={'rows':20000,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'input_columns_preserved':True,'model_replay':True}
    for full,stage in [(False,'validation'),(True,'full')]:
        p,tr,q,u,i,qu,qi,users,items,n=dataset(a.data_dir,full);known=(qu>=0)&(qi>=0);sample=np.random.RandomState(19).choice(len(q),128,replace=False);lib=np.load(a.out/stage/'library.npz')
        for key in lib.files:
            if not key.startswith('svdpp'):continue
            m=dict(np.load(a.out/stage/(key+'_model.npz')));assert np.all(m['effective'][m['cu']==0]==0) and np.all(m['Q'][m['ci']==0]==0);out=np.full(len(sample),float(m['mu']));vu=qu[sample]>=0;vi=qi[sample]>=0;both=vu&vi;out[vu]+=m['bu'][qu[sample][vu]];out[vi]+=m['bi'][qi[sample][vi]];out[both]+=np.einsum('ij,ij->i',m['effective'][qu[sample][both]],m['Q'][qi[sample][both]]);assert np.allclose(clip(out),lib[key][sample],atol=1e-8)
        H=coo_matrix((np.ones(len(u)),(u,i)),shape=(len(users),len(items))).tocsr()
        if not full:assert np.all(np.asarray(H[qu,qi]).ravel()==0)
    p,tr,q,u,i,qu,qi,users,items,n=dataset(a.data_dir,True);U,I=len(users),len(items);b=dict(np.load(a.out/'full/bias.npz'));y=tr.rating.to_numpy(float);r=y-float(b['mu'])-b['bu'][u]-b['bi'][i];R=coo_matrix((r,(u,i)),shape=(U,I)).tocsr();H=coo_matrix((np.ones(len(u)),(u,i)),shape=(U,I)).tocsr();known=(qu>=0)&(qi>=0);sample=np.random.RandomState(29).choice(np.flatnonzero(known),64,replace=False);components=np.load(a.out/'full/neighbors.npz')
    for ratio in [.25,1,4]:
        for pen in [10,50]:
            name=f'joint_r{ratio}_p{pen}';W=load_npz(a.out/'full'/(name+'.npz')).tocsc();rr,cc=W.nonzero();assert np.all(rr%I!=cc)
            for row in sample:
                j=qi[row];lo,hi=W.indptr[j:j+2];ids=W.indices[lo:hi];nb=ids%I;block=ids//I;weights=W.data[lo:hi];value=np.where(block==0,np.asarray(H[qu[row],nb].toarray()).ravel(),np.asarray(R[qu[row],nb].toarray()).ravel());pred=np.clip(float(b['mu'])+b['bu'][qu[row]]+b['bi'][j]+weights@value,1,5);assert np.allclose(pred,components[name][row],atol=1e-6)
    summary['audit']=audit;summary['checks']={'naive_implicit_gradient_equivalence':True,'SVDpp_parameter_replay':True,'cold_factors_neutral':True,'learned_neighbor_replay':True,'self_book_predictors_excluded':True,'validation_targets_held_out':True,'full_ensemble_replay':True};summary['source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['task1_improve.py','task1_svdpp_kernel.cpp','task1_improve_stack.py','task1_6_checks.py']};(a.out/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'task':1,'audit':audit,'checks':summary['checks']},indent=2))

def check6(a):
    from task6_improve_models import framing
    import lightgbm as lgb
    summary=json.loads((a.out/'results.json').read_text());full=np.load(a.out/'full.npz');test=pd.read_csv(a.data_dir/'task_6_test.csv');reference=pd.read_csv(a.accepted);pd.testing.assert_frame_equal(reference.drop(columns='delta'),test.drop(columns='delta'));preds={'accepted':reference.delta.to_numpy()/86400}
    for name,r in summary['runs'].items():
        if summary['new_weights'][name]==0 and summary['blend_weights'][name]==0:preds[name]=np.zeros(len(test));continue
        if name=='cat':M=CatBoostRegressor();M.load_model(str(a.out/(name+'_full.model')));Q=framing(full['Q'],None,test=test)
        else:M=lgb.Booster(model_file=str(a.out/(name+'_full.model')));Q=full['Q']
        preds[name]=np.maximum(0,M.predict(Q))
    new=sum(summary['new_weights'][k]*preds[k] for k in summary['new_weights'])*86400;blend=sum(summary['blend_weights'][k]*preds[k] for k in summary['blend_weights'])*86400;audit={}
    for label,pred in [('new',new),('blend',blend)]:
        path=a.out/label/'task_6_predictions.csv';frame=pd.read_csv(path);assert len(frame)==55403 and np.isfinite(frame.delta).all() and (frame.delta>=0).all();pd.testing.assert_frame_equal(frame.drop(columns='delta'),test.drop(columns='delta'));assert np.allclose(frame.delta,pred,atol=1e-6,rtol=1e-10);audit[label]={'rows':55403,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'input_columns_preserved':True,'model_replay':True}
    from task6_pipeline import load_data
    p,query,u,users,i,items,t,r,starts,early,v,e,rows=load_data(a.data_dir);cached=np.load(a.out/'validation.npz');assert np.array_equal(cached['rows'],rows);core=cached['X'][:,:64];sample=np.random.RandomState(37).choice(len(rows),24,replace=False)
    for stage,allowed,expected in [('validation',np.ones(len(p),bool),cached['X']),('full',np.ones(len(p),bool),full['X'])]:
        if stage=='validation':allowed[v]=False;allowed[e]=False
        got=event_features(i,t,allowed,i[rows[sample]],t[rows[sample]-1],core[sample,19],rows[sample]);assert np.allclose(got,expected[sample,-29:],equal_nan=True,atol=1e-5)
        for index in sample[:8]:
            row=rows[index];mask=allowed&(i==i[row])&(t>=0)&(t<=4102444800)&(t>=t[row-1]);mask[row]=False;gaps=(t[mask]-t[row-1])/86400
            if len(gaps):assert np.allclose(np.log1p(np.quantile(gaps,[0,.1,.25,.5,.75,.9,1])),expected[index,-29:][3:10],atol=1e-5)
    summary['audit']=audit;summary['checks']={'target_timestamp_perturbation':True,'singleton_exclusion':True,'future_quantiles_independently_checked':True,'stage_feature_reconstruction':True,'row_alignment':True,'nonnegative_seconds':True,'full_ensemble_replay':True};summary['source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['task6_improve.py','task6_improve_models.py','task1_6_checks.py']};(a.out/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'task':6,'audit':audit,'checks':summary['checks']},indent=2))
if __name__=='__main__':main()
