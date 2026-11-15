"""Independent checks for FM equations, gradients and exported ensembles."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,itertools,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from task3_fm import library,clip,CONFIGS,design

def numerical(out):
    out.mkdir(parents=True,exist_ok=True);lib=library(out);rng=np.random.RandomState(7);linear=rng.normal(0,.1,10).astype(np.float32);V=rng.normal(0,.1,(10,4)).astype(np.float32);ids=np.array([[0,2,4,-1],[1,3,8,9]],np.int32);x=np.array([[1,-.5,2,0],[.25,1,0,-1]],np.float32);mean=.3;pred=np.zeros(2,np.float32);lib.fm_predict(2,4,4,ids,x,linear,V,mean,pred);expected=[]
    for row in range(2):
        active=[(int(j),float(v)) for j,v in zip(ids[row],x[row]) if j>=0 and v!=0];p=mean+sum(float(linear[j])*v for j,v in active)
        p+=sum(float(V[j].astype(float)@V[k].astype(float))*v*w for (j,v),(k,w) in itertools.combinations(active,2));expected.append(p)
    assert np.allclose(pred,expected,atol=1e-6)
    A=np.ascontiguousarray(ids[:1]);B=np.ascontiguousarray(x[:1]);target=np.array([1.2],np.float32);oldlinear=linear.copy();oldV=V.copy();error=target[0]-pred[0];sumV=np.sum(oldV[A[0,:3]]*B[0,:3,None],axis=0);el=oldlinear.copy();ev=oldV.copy()
    for j,value in zip(A[0],B[0]):
        if j<0 or value==0:continue
        el[j]+=.005*(error*value-.03*oldlinear[j]);ev[j]+=.005*(error*value*(sumV-oldV[j]*value)-.2*oldV[j])
    lib.fm_train(1,4,4,A,B,target,linear,V,mean,0,1,7,.005,.2,.03);assert np.allclose(linear,el,atol=1e-7) and np.allclose(V,ev,atol=1e-7)
    Q=rng.normal(size=(4,12));idf=np.array([1.,2.,3.,4.]);uu=np.array([0,0,1]);ii=np.array([0,1,2]);t=np.array([1e9,1e9+86400,1e9+172800]);_,features,_,query,_,_,_=design(uu,ii,t,np.array([0]),np.array([2]),np.array([1e9]),2,False,{'Q':Q,'idf':idf});expected=idf[1]*Q[1];expected/=np.linalg.norm(expected);assert np.allclose(features[0,4:16],expected,atol=1e-6);expected=idf[0]*Q[0]+idf[1]*Q[1];expected/=np.linalg.norm(expected);assert np.allclose(query[0,4:16],expected,atol=1e-6)
    print('PASS: explicit FM equation, SGD gradient, and own-book exclusion from training profiles.',flush=True)

def replay(path,indices=None):
    z=np.load(path);ids=z['ids'];x=z['x'];base=z['base']
    if indices is not None:ids=ids[indices];x=x[indices];base=base[indices]
    V=z['V'];linear=z['linear'];known=ids>=0;sidx=np.maximum(ids,0);part=V[sidx]*x[:,:,None]*known[:,:,None];pred=float(z['mean'])+np.sum(linear[sidx]*x*known,axis=1)+.5*np.sum(part.sum(1)**2-(part**2).sum(1),axis=1);return clip(base+pred)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--data-dir',type=Path);ap.add_argument('--legacy',type=Path);ap.add_argument('--prior',type=Path);ap.add_argument('--round2',type=Path);ap.add_argument('--round3',type=Path);ap.add_argument('--numerical-only',action='store_true');a=ap.parse_args();numerical(a.out/'checks')
    if a.numerical_only:return
    for stage in ['validation0','validation2','validation5','validation8','full']:
        part=a.out/stage
        for f in part.glob('*_features.npz'):
            features=np.load(f)
            for name in features.files:
                path=part/(name+'_model.npz');z=np.load(path);sample=np.random.RandomState(11).choice(len(z['ids']),min(128,len(z['ids'])),replace=False);pred=replay(path,sample);assert np.isfinite(pred).all() and np.allclose(pred,features[name][sample],atol=3e-5),name
    print('PASS: FM checkpoints independently replay from saved parameters.',flush=True)
    if not (a.out/'test_predictions.npz').exists():return
    from task3_fm_stack import values
    from task3_round3 import extended_design,dataset
    from catboost import CatBoostRegressor
    from scipy.sparse import coo_matrix
    s=json.loads((a.out/'results.json').read_text());X,_,names=extended_design(a.data_dir,a.legacy,a.prior,a.round2,a.round3,True);assert names==s['base_features'];fm=values(a.out,True);Z=np.c_[X,np.column_stack([fm[k] for k in s['selection']['fm_features']])].astype(np.float32);preds={}
    for name in s['selected']['depth_weights']:
        m=CatBoostRegressor();m.load_model(str(a.out/(name+'_full.cbm')));preds[name]=clip(m.predict(Z))
    augmented=clip(sum(s['selected']['depth_weights'][k]*v for k,v in preds.items()));ref=np.load(a.round3/'test_predictions.npz')['primary'];blend=.5*augmented+.5*ref;test=pd.read_csv(a.data_dir/'task_3_test.csv');audit={}
    for label,pred in [('augmented',augmented),('blend50',blend)]:
        path=a.out/label/'task_3_predictions.csv';frame=pd.read_csv(path);assert len(frame)==20000 and np.isfinite(frame.rating).all() and frame.rating.between(1,5).all();pd.testing.assert_frame_equal(frame.drop(columns='rating'),test.drop(columns='rating'));assert np.allclose(frame.rating,pred,atol=1e-12);audit[label]={'rows':20000,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_replay':True,'input_columns_preserved':True,'mean_absolute_change_from_accepted':float(np.mean(np.abs(pred-ref)))}
    p,tr,q,u,i,t,y,qu,qi,users,items,aa,bb=dataset(a.data_dir,True);order=np.random.RandomState(42).permutation(len(p));train=order[:int(.9*len(p))];held=order[int(.9*len(p)):];graph=coo_matrix((np.ones(len(train)),(u[train],i[train])),shape=(len(users),len(items))).tocsr();assert np.all(np.asarray(graph[u[held],i[held]]).ravel()==0)
    inputs={f:hashlib.sha256((a.data_dir/f).read_bytes()).hexdigest() for f in ['tasks_3_4_train.csv','task_3_test.csv']};assert inputs==s['input_manifest']
    for stage in ['validation0','validation2','validation5','validation8','full']:
        for manifest in (a.out/stage).glob('manifest_*.json'):
            guard=json.loads(manifest.read_text());assert guard['inputs']==inputs and guard['full']==(stage=='full')
    s['audit']=audit;s['checks']={'FM_pairwise_formula':True,'FM_SGD_gradient':True,'profile_target_exclusion':True,'checkpoint_parameter_replay':True,'validation_pairs_held_out':True,'stage_and_data_hashes':True,'full_ensemble_replay':True};s['source_sha256']={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['task3_fm.py','task3_fm_kernel.cpp','task3_fm_stack.py','task3_fm_checks.py']};(a.out/'results.json').write_text(json.dumps(s,indent=2));print(json.dumps({'audit':audit,'checks':s['checks']},indent=2))
if __name__=='__main__':main()
