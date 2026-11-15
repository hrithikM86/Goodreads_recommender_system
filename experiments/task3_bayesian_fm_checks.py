"""Reconstruct Bayesian posterior-window predictions from raw FM parameters."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

def replay(stage):
    info=json.loads((stage/'results.json').read_text());Q=load_npz(stage/'query.npz');Q2=Q.multiply(Q);z=np.load(stage/'model.npz');pred=np.zeros(Q.shape[0])
    for w0,w,V in zip(z['w0'],z['w'],z['V']):
        sums=Q@V;pred+=w0+Q@w+.5*np.sum(sums*sums-Q2@(V*V),axis=1)
    pred=np.clip(pred/len(z['w0']),1,5);stored=np.load(stage/'features.npz')[info['selected_feature']];assert np.allclose(pred,stored,atol=1e-8);return pred

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--round3',type=Path);ap.add_argument('--data-dir',type=Path);a=ap.parse_args();val=replay(a.out/'bayes_validation');print('PASS: Bayesian FM selected checkpoint replay.',flush=True)
    path=a.out/'bayes_blend_results.json'
    if not path.exists():return
    result=json.loads(path.read_text());w=result['weights'];result['checks']={'validation_posterior_window_replay':True}
    if (a.out/'bayes_blend/task_3_predictions.csv').exists():
        bayes=replay(a.out/'bayes_full');z=np.load(a.out/'test_predictions.npz');pred=np.clip(w['sgd_augmented']*z['augmented']+w['bayesian_fm']*bayes+w['accepted']*z['reference'],1,5);frame=pd.read_csv(a.out/'bayes_blend/task_3_predictions.csv');test=pd.read_csv(a.data_dir/'task_3_test.csv');assert len(frame)==20000 and frame.rating.between(1,5).all();pd.testing.assert_frame_equal(frame.drop(columns='rating'),test.drop(columns='rating'));assert np.allclose(frame.rating,pred,atol=1e-8);result['checks'].update(full_posterior_window_replay=True,full_blend_replay=True,input_columns_preserved=True);result['prediction_sha256']=hashlib.sha256((a.out/'bayes_blend/task_3_predictions.csv').read_bytes()).hexdigest()
    result['source_sha256']={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['task3_bayesian_fm.py','task3_fm_bayes_blend.py','task3_bayesian_fm_checks.py']};path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
