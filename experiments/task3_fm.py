"""Time-aware general second-order factorization machines for positive ratings."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,json,hashlib,pickle,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from task3_optimize import dataset,clip,mse

CONFIGS=[
 dict(name='basic16_res',context=False,k=16,residual=True,reg=.05,lr=.008,seed=41),
 dict(name='context16_res',context=True,k=16,residual=True,reg=.1,lr=.005,seed=43),
 dict(name='basic32_raw',context=False,k=32,residual=False,reg=.1,lr=.005,seed=47),
 dict(name='context16_raw',context=True,k=16,residual=False,reg=.2,lr=.005,seed=53),
 dict(name='context32_res',context=True,k=32,residual=True,reg=.2,lr=.005,seed=59),
 dict(name='basic16_resstrong',context=False,k=16,residual=True,reg=.2,lr=.005,seed=61,epochs=[5,10,20,40]),
 dict(name='context16_resstrong',context=True,k=16,residual=True,reg=.4,lr=.004,seed=67,epochs=[5,10,20,40]),
 dict(name='context32_rawweak',context=True,k=32,residual=False,reg=.03,lr=.004,seed=71),
 dict(name='profile16_res',context=False,dense=True,k=16,residual=True,reg=.2,lr=.005,seed=73,epochs=[5,10,20,40]),
 dict(name='profile32_raw',context=False,dense=True,k=32,residual=False,reg=.05,lr=.005,seed=79)]
EPOCHS=[20,50,80]

def library(out):
    path=out/'fm.so'
    source=Path(__file__).with_name('task3_fm_kernel.cpp')
    if not path.exists() or path.stat().st_mtime<source.stat().st_mtime:subprocess.run(['g++','-O3','-shared','-fPIC',str(source),'-o',str(path)],check=True)
    lib=ctypes.CDLL(str(path.resolve()));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.fm_predict.argtypes=[ctypes.c_int]*3+[ip,fp,fp,fp,ctypes.c_float,fp];lib.fm_train.argtypes=[ctypes.c_int]*3+[ip]+[fp]*4+[ctypes.c_float]+[ctypes.c_int]*3+[ctypes.c_float]*3
    return lib

def design(u,i,t,qu,qi,qt,U,context,embedding=None):
    good=(t>=0)&(t<=4102444800);qgood=(qt>=0)&(qt<=4102444800);count=np.bincount(u[good],minlength=U);center=np.bincount(u[good],weights=t[good],minlength=U)/np.maximum(count,1);center[count==0]=np.mean(t[good]);days=np.floor(np.clip(t,0,4102444800)/86400).astype(np.int64);qd=np.floor(np.clip(qt,0,4102444800)/86400).astype(np.int64)
    fields=[('user',u,qu,np.ones(len(u),bool),(qu>=0)),('book',i,qi,np.ones(len(i),bool),(qi>=0)),('period',days//90,qd//90,good,qgood),('drift',u,qu,good,qgood&(qu>=0))]
    if context:fields += [('user180',(u.astype(np.int64)+1)*1000000+days//180,(qu.astype(np.int64)+1)*1000000+qd//180,good,qgood&(qu>=0)),('book180',(i.astype(np.int64)+1)*1000000+days//180,(qi.astype(np.int64)+1)*1000000+qd//180,good,qgood&(qi>=0))]
    A=[];B=[];va=[];vb=[];maps={};offset=0
    for name,labels,queries,valid,qvalid in fields:
        c,index=pd.factorize(labels[valid]);ids=np.full(len(u),-1,np.int32);ids[valid]=c+offset;qids=pd.Index(index).get_indexer(queries);qids=np.where(qvalid&(qids>=0),qids+offset,-1).astype(np.int32);x=np.ones(len(u),np.float32);qx=np.ones(len(qu),np.float32)
        if name=='drift':
            d=(t-center[u])/(86400*365.25);qdev=(qt-center[np.maximum(qu,0)])/(86400*365.25);x=np.where(valid,np.clip(np.sign(d)*np.abs(d)**.4,-5,5),0).astype(np.float32);qx=np.where(qvalid,np.clip(np.sign(qdev)*np.abs(qdev)**.4,-5,5),0).astype(np.float32)
        maps[name]={'offset':offset,'labels':np.asarray(index)};offset+=len(index);A.append(ids);B.append(qids);va.append(x);vb.append(qx)
    if embedding is not None:
        Q=embedding['Q'][:,:12].copy();idf=embedding['idf'];X=coo_matrix((np.ones(len(u)),(u,i)),shape=(U,len(Q))).tocsr();sums=X.multiply(idf)@Q
        def profiles(uu,ii,training):
            profile=np.zeros((len(uu),12));book=np.zeros_like(profile);ok=(uu>=0)&(ii>=0);profile[uu>=0]=sums[uu[uu>=0]];book[ii>=0]=Q[ii[ii>=0]]
            if training:profile-=idf[ii,None]*Q[ii]
            else:
                rows=np.flatnonzero(ok);present=np.asarray(X[uu[rows],ii[rows]]).ravel()>0;rows=rows[present];profile[rows]-=idf[ii[rows],None]*Q[ii[rows]]
            profile/=np.maximum(np.linalg.norm(profile,axis=1,keepdims=True),1e-8);book/=np.maximum(np.linalg.norm(book,axis=1,keepdims=True),1e-8);return np.c_[profile,book].astype(np.float32)
        dense=profiles(u,i,True);qdense=profiles(qu,qi,False)
        for j in range(24):
            maps['profile'+str(j)]={'offset':offset,'labels':np.array([0])};A.append(np.full(len(u),offset,np.int32));B.append(np.full(len(qu),offset,np.int32));va.append(dense[:,j]);vb.append(qdense[:,j]);offset+=1
    return np.ascontiguousarray(np.column_stack(A),np.int32),np.ascontiguousarray(np.column_stack(va),np.float32),np.ascontiguousarray(np.column_stack(B),np.int32),np.ascontiguousarray(np.column_stack(vb),np.float32),offset,maps,center

def worker(data,prior,out,which,full=False,selection=None,round2=None):
    out.mkdir(parents=True,exist_ok=True);configs=[CONFIGS[j] for j in which];manifest={'full':full,'configs':configs,'epochs':EPOCHS,'selection':selection,'inputs':{f:hashlib.sha256((data/f).read_bytes()).hexdigest() for f in ['tasks_3_4_train.csv','task_3_test.csv']}}
    guard=out/('manifest_'+str(which[0])+'.json')
    if guard.exists():assert json.loads(guard.read_text())==manifest
    guard.write_text(json.dumps(manifest,indent=2));p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);qt=q.timestamp.to_numpy(float)
    with open(prior/('full' if full else 'validation')/'temporal_bias.pkl','rb') as h:bias=pickle.load(h)
    lib=library(out);start=time.time()
    for cfg in configs:
        path=out/(cfg['name']+'_features.npz')
        if path.exists():continue
        embedding=None
        if cfg.get('dense'):
            assert round2 is not None and json.loads((round2/'manifest.json').read_text())==manifest['inputs'];embedding=dict(np.load(round2/('full' if full else 'validation')/'unsupervised_embeddings.npz'))
        A,x,B,qx,F,maps,center=design(u,i,t,qu,qi,qt,len(users),cfg['context'],embedding);target=np.ascontiguousarray(y-bias.training_prediction if cfg['residual'] else y,np.float32);base=bias.predict(qu.astype(np.int64),qi.astype(np.int64),qt) if cfg['residual'] else np.zeros(len(q));mean=float(target.mean());V=np.random.RandomState(cfg['seed']).normal(0,.025,(F,cfg['k'])).astype(np.float32);linear=np.zeros(F,np.float32);last=0;features={};epochs=cfg.get('epochs',EPOCHS)
        if selection is not None:epochs=sorted({int(name.rsplit('_e',1)[1]) for name in selection if name.startswith(cfg['name']+'_e')})
        for epoch in epochs:
            lib.fm_train(len(A),A.shape[1],cfg['k'],A,x,target,linear,V,mean,last,epoch-last,cfg['seed'],cfg['lr'],cfg['reg'],.03);last=epoch;assert np.isfinite(V).all() and np.isfinite(linear).all();pred=np.zeros(len(B),np.float32);lib.fm_predict(len(B),B.shape[1],cfg['k'],B,qx,linear,V,mean,pred);features[cfg['name']+f'_e{epoch}']=clip(base+pred)
            # Every checkpoint can be replayed directly without compiling the trainer.
            np.savez_compressed(out/(cfg['name']+f'_e{epoch}_model.npz'),linear=linear,V=V,mean=mean,ids=B,x=qx,base=base,center=center);print('FM',cfg['name'],epoch,'seconds',round(time.time()-start),flush=True)
        with open(out/(cfg['name']+'_maps.pkl'),'wb') as h:pickle.dump(maps,h)
        temp=out/(cfg['name']+'_partial.npz');np.savez_compressed(temp,**features);temp.replace(path)

def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','prior','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--which',type=int,nargs='+',required=True);ap.add_argument('--full',action='store_true');ap.add_argument('--selection',type=Path);ap.add_argument('--round2',type=Path);a=ap.parse_args();selection=json.loads(a.selection.read_text())['fm_features'] if a.selection else None;worker(a.data_dir,a.prior,a.out,a.which,a.full,selection,a.round2)
if __name__=='__main__':main()
