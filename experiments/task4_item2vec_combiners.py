"""Fresh XGBoost/LightGBM on all accepted features plus chronological Item2vec.
Reuse previous hyperparameters; no new tuning on evaluation labels.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from task4_tree_combiners import matrices,predict,fit,W,DATA,NSEL
OUT=W/'task4_item2vec';OLD=W/'task4_tree_combiners'
def augmented(full=False):
 d,s=matrices(full);z=np.load(OUT/('full_features.npz' if full else 'validation_features.npz'));f=np.column_stack([z[k] for k in z.files]);n=len(f)//2;assert n==len(d);d=np.c_[d,f[:n]-f[n:]].astype('f');s=np.c_[s,f[:n]+f[n:]].astype('f');assert np.isfinite(d).all() and np.isfinite(s).all();return d,s,list(z.files)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['xgboost','lightgbm'],required=True);a=ap.parse_args();kind=a.kind;config=json.loads((OLD/(kind+'_results.json')).read_text())['selected'];d,s,names=augmented();X=np.r_[np.c_[d[:NSEL],s[:NSEL]],np.c_[-d[:NSEL],s[:NSEL]]];y=np.r_[np.ones(NSEL),np.zeros(NSEL)];print('TRAIN',kind,config,'dimensions',X.shape,flush=True);m=fit(kind,config['params'],config['iterations'],X,y);ev=predict(m,kind,d[NSEL:],s[NSEL:]);old=np.load(OLD/(kind+'_evaluation.npz'))['margin'];acc=lambda p:float(np.mean((p>0)+.5*(p==0)));r={'kind':kind,'hyperparameters_reused':config,'evaluation_accuracy':acc(ev),'same_combiner_without_item2vec_evaluation':acc(old),'new_features':names,'total_component_scores':d.shape[1],'selection_pairs':NSEL,'evaluation_pairs':len(ev),'protocol':'All previous component features unchanged; add only training-fitted Item2vec scores. Fixed previous tree settings; no evaluation-based tuning. Cohorts reused across earlier experiments. Test labels unavailable.','input_feature_hashes':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['validation_features.npz','full_features.npz']}};print('EVALUATION',kind,r['evaluation_accuracy'],'reference',acc(old),flush=True)
 if kind=='xgboost':m.save_model(OUT/(kind+'.ubj'))
 else:m.booster_.save_model(str(OUT/(kind+'.txt')))
 np.savez_compressed(OUT/(kind+'_evaluation.npz'),margin=ev);fd,fs,fn=augmented(True);assert fn==names;p=predict(m,kind,fd,fs);assert np.isfinite(p).all() and np.allclose(p,-predict(m,kind,-fd,fs));q=pd.read_csv(DATA/'task_4_test.csv');f=q.copy();f['which']=np.where(p>=0,f.item1,f.item2);dest=Path('/workspace/output')/('task4_item2vec_'+kind);dest.mkdir(exist_ok=True);path=dest/'task_4_predictions.csv';f.to_csv(path,index=False);read=pd.read_csv(path);assert len(read)==20000;assert read.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert ((read.which==read.item1)|(read.which==read.item2)).all();np.savez_compressed(OUT/(kind+'_test.npz'),margin=p);accepted=pd.read_csv('/workspace/output/task4_lightgbm/task_4_predictions.csv');r['output_audit']={'rows':len(read),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'changed_vs_accepted_lightgbm':int((read.which!=accepted.which).sum()),'pair_swap_antisymmetry':True};(OUT/(kind+'_results.json')).write_text(json.dumps(r,indent=2));print('RESULT',r,flush=True)
if __name__=='__main__':main()
