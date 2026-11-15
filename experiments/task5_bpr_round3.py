"""Stronger global BPR preference models, plus recent-history latent cosines."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from implicit.bpr import BayesianPersonalizedRanking
from task5_improve import data,graph

def run(a):
 part=a.out/('full' if a.full else 'validation');part.mkdir(exist_ok=True,parents=True)
 if (part/'bpr.npz').exists():return
 h,contexts,qu,qi,U,I,q=data(a.data_dir,a.baseline,a.full);X=graph(h,U,I);pop=np.asarray(X.sum(0)).ravel();known=qi>=0;known[known]&=pop[qi[known]]>0;N=len(contexts);ctx=np.tile(np.arange(N),2);last=np.array([c[-1] for c in contexts]);profiles={}
 for cap in [3,20]:
  short=[c[-cap:] for c in contexts];lens=np.array(list(map(len,short)));profiles[cap]=coo_matrix((np.repeat(1/np.maximum(lens,1),lens),(np.repeat(np.arange(N),lens),np.concatenate(short))),shape=(N,I)).tocsr()
 result={};samples={};start=time.time();configs=[('bpr128',128,.03,.03,350,42,[50,150,350]),('bpr96',96,.003,.05,300,23,[50,150,300])]
 for name,k,reg,lr,iterations,seed,checkpoints in configs:
  model=BayesianPersonalizedRanking(factors=k,iterations=iterations,regularization=reg,learning_rate=lr,random_state=seed,num_threads=1)
  def callback(epoch,*extra):
   epoch+=1
   if epoch not in checkpoints:return
   Uv=model.user_factors;Iv=model.item_factors;pred=np.zeros(len(qi),'f');pred[known]=np.einsum('ij,ij->i',Uv[qu[known]],Iv[qi[known]]);result[f'{name}_e{epoch}_personal']=pred;V=Iv[:,:-1].copy();V[pop==0]=0;norm=np.linalg.norm(V,axis=1);valid=known&(pop[last[ctx]]>0);cos=np.zeros(len(qi),'f');cos[valid]=np.einsum('ij,ij->i',V[last[ctx[valid]]],V[qi[valid]])/np.maximum(norm[last[ctx[valid]]]*norm[qi[valid]],1e-9);result[f'{name}_e{epoch}_lastcos']=cos
   for cap,C in profiles.items():
    mean=C@V;length=np.linalg.norm(mean,axis=1);cos=np.zeros(len(qi),'f');cos[known]=np.einsum('ij,ij->i',mean[ctx[known]],V[qi[known]])/np.maximum(length[ctx[known]]*norm[qi[known]],1e-9);result[f'{name}_e{epoch}_recent{cap}cos']=cos
   if epoch==iterations:
    ix=np.random.RandomState(seed).choice(np.flatnonzero(known),min(512,known.sum()),replace=False);samples[name+'_rows']=ix;samples[name+'_users']=Uv[qu[ix]].copy();samples[name+'_items']=Iv[qi[ix]].copy();samples[name+'_scores']=pred[ix]
   print('GLOBAL_BPR',a.full,name,epoch,'seconds',round(time.time()-start),flush=True)
  model.fit(X,show_progress=False,callback=callback)
  assert name+'_e'+str(iterations)+'_personal' in result
 temp=part/'bpr.tmp.npz';np.savez_compressed(temp,**result);temp.replace(part/'bpr.npz');np.savez_compressed(part/'bpr_replay_sample.npz',**samples);(part/'bpr_config.json').write_text(json.dumps({'full':a.full,'configs':configs,'library_version':__import__('implicit').__version__,'factor_states':'replay sample retained; full models regenerate from deterministic fits'},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in ['data-dir','baseline','out']:ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--full',action='store_true');run(ap.parse_args())
