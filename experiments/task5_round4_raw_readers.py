"""Task5 full-history reader signals with the accepted raw-feature neural architecture."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,pickle
from pathlib import Path
import numpy as np,pandas as pd
from tasks245_round4 import load,pair,W,DATA,acc
from task5_round3 import nn_fit,nn_margin

def matrices(full=False):
 d,s,n=load(5,full);z=np.load(W/'task5_round4'/('readers_full.npz' if full else 'readers_validation.npz'));rd,rs=pair(np.column_stack([z[k] for k in z.files]));return np.c_[d,rd],np.c_[s,rs],n

def main():
 out=W/'task5_round4_raw_readers';out.mkdir(exist_ok=True);d,s,n=matrices();idx=np.random.RandomState(17).permutation(n);tr=idx[:int(.8*n)];va=idx[int(.8*n):];z=np.load(W/'task5_round4/reference.npz');ri,re=z['inner'],z['evaluation'];configs={'small':{'seed':79,'hidden':[64,32],'alpha':.1},'wide':{'seed':179,'hidden':[128,64],'alpha':1.}};inner={};reports={}
 for k,cfg in configs.items():
  path=out/(k+'_inner.pkl');rp=out/(k+'.json')
  if not rp.exists():
   _,r=nn_fit(d[tr],s[tr],35,(d[va],s[va],ri),path,**cfg);r.update(cfg);rp.write_text(json.dumps(r,indent=2))
  reports[k]=json.loads(rp.read_text());m,sc=pickle.loads(path.read_bytes());inner[k]=nn_margin(m,sc,d[va],s[va]);print('INNER',k,reports[k],flush=True)
 choices=[]
 for w in [0,.25,.5,.75,1]:
  p=w*inner['small']+(1-w)*inner['wide']
  for a in [0,.25,.5,.75,1]:choices.append((acc(a*p+(1-a)*ri),w,a))
 score,w,a=max(choices,key=lambda x:x[0]);weights={'small':a*w,'wide':a*(1-w),'reference':1-a};weights={k:v for k,v in weights.items() if v};print('CHOSEN',weights,score,flush=True);pred=weights.get('reference',0)*re;models={}
 for k,weight in weights.items():
  if k=='reference':continue
  r=reports[k];path=out/(k+'_evaluation.pkl')
  if path.exists():m,sc=pickle.loads(path.read_bytes())
  else:
   (m,sc),_=nn_fit(d[:n],s[:n],r['epochs'],seed=r['seed'],hidden=r['hidden'],alpha=r['alpha']);path.write_bytes(pickle.dumps((m,sc),protocol=5))
  models[k]=(m,sc);pred+=weight*nn_margin(m,sc,d[n:],s[n:])
 r={'task':5,'weights':weights,'runs':reports,'inner_accuracy':score,'reference_inner':acc(ri),'primary_evaluation':acc(pred),'reference_evaluation':acc(re),'blend50_evaluation':acc(.5*pred+.5*re),'transform':'raw StandardScaler on fitting rows','protocol':'same inner-only selection; never train on evaluation labels'};print('RESULT',r,flush=True);(out/'results.json').write_text(json.dumps(r,indent=2));np.savez_compressed(out/'evaluation.npz',primary=pred,reference=re)
 del d,s;fd,fs,_=matrices(True);ref=np.load(W/'task5_round3/preference/test_predictions.npz')['primary'];pred=weights.get('reference',0)*ref
 for k,(m,sc) in models.items():pred+=weights[k]*nn_margin(m,sc,fd,fs)
 np.savez_compressed(out/'test_margins.npz',primary=pred,reference=ref);test=pd.read_csv(DATA/'task_5_test.csv')
 for key,v in [('primary',pred),('conservative',.5*pred+.5*ref)]:
  dest=Path('/workspace/output')/('task5_round4_raw_readers_'+key);dest.mkdir(exist_ok=True);f=test.copy();f['which']=np.where(v>=0,f.item1,f.item2);f.to_csv(dest/'task_5_predictions.csv',index=False)
if __name__=='__main__':main()
