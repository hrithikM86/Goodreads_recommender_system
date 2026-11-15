"""Wider joint book neighborhoods with separate exposure/rating penalties."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,hashlib,pickle,time
import numpy as np
from scipy.sparse import coo_matrix,load_npz,save_npz
from task3_optimize import dataset,clip,mse

RATIOS=[.25,1.,4.,16.]
PENALTIES=[10.,50.,200.]
def config_name(width,ratio,penalty):return f'jointw{width}_ratio{ratio:g}_pen{penalty:g}'

def ridge_spectrum(A,y):
    dual=A.shape[0]<A.shape[1]
    eig,V=np.linalg.eigh(A@A.T if dual else A.T@A)
    return (A.T@V if dual else V),V.T@(y if dual else A.T@y),np.maximum(eig,0)

def fit_worker(data,prior,round2,out,width,full=False,selected=None):
    stage=out/('full' if full else 'validation');stage.mkdir(parents=True,exist_ok=True)
    cfg={'width':width,'ratios':RATIOS,'penalties':PENALTIES,'full':full,'selected':selected,'inputs':{f:hashlib.sha256((data/f).read_bytes()).hexdigest() for f in ['tasks_3_4_train.csv','task_3_test.csv']}}
    guard=stage/f'manifest{width}.json'
    if guard.exists():assert json.loads(guard.read_text())==cfg,'Changed configuration/data: use fresh output'
    guard.write_text(json.dumps(cfg,indent=2));target=stage/f'features{width}.npz'
    if target.exists():return
    p,tr,q,u,i,t,y,qu,qi,users,items,a,b=dataset(data,full);U,I=len(users),len(items);qt=q.timestamp.to_numpy(float)
    with open(prior/('full' if full else 'validation')/'temporal_bias.pkl','rb') as f:bias=pickle.load(f)
    residual=y-bias.training_prediction;base=bias.predict(qu.astype(np.int64),qi.astype(np.int64),qt);shape=(U,I)
    X=coo_matrix((np.ones(len(y)),(u,i)),shape=shape).tocsr();R=coo_matrix((residual,(u,i)),shape=shape).tocsr();T=coo_matrix((t,(u,i)),shape=shape).tocsr();C=R.tocsc()
    candidate=load_npz(round2/('full' if full else 'validation')/(('wide_' if width==256 else '')+'learned_rating25.npz')).tocsc();assert np.count_nonzero(candidate.diagonal())==0
    specs=[(r,pen) for r in RATIOS for pen in PENALTIES if selected is None or config_name(width,r,pen) in selected];result={config_name(width,r,pen)+f'_t{tau}':base.copy() for r,pen in specs for tau in [0,365]};result[f'jointw{width}_support']=np.zeros(len(q));groups={}
    for z,j in enumerate(qi):
        if j>=0 and qu[z]>=0:groups.setdefault(j,[]).append(z)
    saved={config_name(width,r,pen):[[],[],[]] for r,pen in specs} if full else {};start=time.time()
    for z,(j,queries) in enumerate(groups.items()):
        lo,hi=candidate.indptr[j:j+2];nb=candidate.indices[lo:hi];assert j not in nb
        if not len(nb):continue
        lo,hi=C.indptr[j:j+2];us=C.indices[lo:hi];yy=C.data[lo:hi];queries=np.array(queries);exposure=X[us][:,nb].toarray();rating=R[us][:,nb].toarray();bx=X[qu[queries]][:,nb].toarray();br=R[qu[queries]][:,nb].toarray();B=np.c_[bx,br];times=T[qu[queries]][:,nb].toarray();tw=np.exp(-np.abs(times-qt[queries,None])/86400/365)*((times>=0)&(times<=4102444800));BW=B*np.tile(tw,(1,2));result[f'jointw{width}_support'][queries]=np.log1p(bx.sum(1))
        for ratio in RATIOS:
            penalties=[pen for r,pen in specs if r==ratio]
            if not penalties:continue
            # Whitening makes isotropic ridge equal to separate diagonal penalties:
            # lambda_presence ||wx||² + lambda_presence*ratio ||wr||².
            scale=np.r_[np.ones(len(nb)),np.full(len(nb),1/np.sqrt(ratio))];A=np.c_[exposure,rating]*scale;left,rhs,eig=ridge_spectrum(A,yy)
            for pen in penalties:
                name=config_name(width,ratio,pen);w=((left@(rhs/(eig+pen)))*scale).astype(np.float32)
                result[name+'_t0'][queries]+=B@w;result[name+'_t365'][queries]+=BW@w
                if full:
                    rows,cols,vals=saved[name];rows.extend(np.r_[nb,nb+I]);cols.extend([j]*len(w));vals.extend(w)
        if z%2048==0:print('Joint split ridge',width,z,len(groups),'seconds',round(time.time()-start),flush=True)
    for name,(rows,cols,vals) in saved.items():save_npz(stage/(name+'.npz'),coo_matrix((np.asarray(vals,np.float32),(rows,cols)),shape=(2*I,I)).tocsr())
    temp=stage/f'features{width}.partial.npz';np.savez_compressed(temp,**{k:clip(v) if not k.endswith('support') else v for k,v in result.items()});temp.replace(target)

def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','prior','round2','out']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--width',type=int,choices=[64,256],required=True);ap.add_argument('--full',action='store_true');ap.add_argument('--selection',type=Path);a=ap.parse_args();selected=json.loads(a.selection.read_text())['component_configs'] if a.selection else None
    fit_worker(a.data_dir,a.prior,a.round2,a.out,a.width,a.full,selected)
if __name__=='__main__':main()
