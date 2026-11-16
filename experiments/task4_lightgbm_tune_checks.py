"""Frozen-bank provenance, selected inner-model and exported classifier replay checks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from task4_lightgbm_tune import OUT,augmented,accuracy,W,DATA,NSEL,lgb

def main():
 r=json.loads((OUT/'results.json').read_text());raw=json.loads((W/'task4_item2vec/full_report.json').read_text())['input_hashes']
 for name,h in raw.items():assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==h
 manifest=json.loads((OUT/'input_manifest.json').read_text());old_base=json.loads((W/'task4_tree_combiners/lightgbm_results.json').read_text())['input_hashes'];old_item=json.loads((W/'task4_item2vec/lightgbm_results.json').read_text())['input_feature_hashes']
 for name,h in old_base.items():assert manifest['task4_two_approaches/'+name]==h
 for name,h in old_item.items():assert manifest['task4_item2vec/'+name]==h
 for name,h in manifest.items():assert hashlib.sha256((W/name).read_bytes()).hexdigest()==h
 d,s,_=augmented();assert d.shape[1]==142;idx=np.random.RandomState(17).permutation(NSEL);va=idx[int(.8*NSEL):];selected=r['selected']['settings']['name'];inner=lgb.Booster(model_file=str(OUT/(selected+'_inner.txt')));ip=inner.predict(np.c_[d[va],s[va]],num_threads=2)-inner.predict(np.c_[-d[va],s[va]],num_threads=2);assert np.allclose(ip,np.load(OUT/(selected+'_inner.npz'))['margin'],atol=1e-7);assert accuracy(ip)==r['selected']['inner_accuracy']
 model=lgb.Booster(model_file=str(OUT/'lightgbm.txt'));assert model.num_feature()==284 and model.num_trees()==r['selected']['trees'];ev=model.predict(np.c_[d[NSEL:],s[NSEL:]],num_threads=2)-model.predict(np.c_[-d[NSEL:],s[NSEL:]],num_threads=2);assert np.allclose(ev,np.load(OUT/'evaluation_margins.npz')['margin'],atol=1e-7);assert accuracy(ev)==r['evaluation_accuracy'];fd,fs,_=augmented(True);p=model.predict(np.c_[fd,fs],num_threads=2)-model.predict(np.c_[-fd,fs],num_threads=2);assert np.allclose(p,np.load(OUT/'test_margins.npz')['margin'],atol=1e-7);q=pd.read_csv(DATA/'task_4_test.csv');path=Path('/workspace/output/task4_lightgbm_tuned/task_4_predictions.csv');f=pd.read_csv(path);assert len(f)==20000 and f.drop(columns='which').equals(q.drop(columns='which',errors='ignore'));assert np.array_equal(f.which,np.where(p>=0,q.item1,q.item2));assert ((f.which==f.item1)|(f.which==f.item2)).all();assert hashlib.sha256(path.read_bytes()).hexdigest()==r['output_audit']['sha256'];old_report=json.loads((W/'task4_item2vec/lightgbm_results.json').read_text());old_path=Path('/workspace/output/task4_item2vec_lightgbm/task_4_predictions.csv');assert hashlib.sha256(old_path.read_bytes()).hexdigest()==old_report['output_audit']['sha256'];report={'bank_hashes_match_accepted':True,'raw_input_CSV_hashes_match_accepted':True,'component_scores':142,'selected_inner_model_replay':True,'selection_split_before_pair_mirroring':True,'evaluation_model_replay':True,'test_model_replay':True,'original_accepted_CSV_untouched':True,'all_rows_checked':len(f),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};(OUT/'audit.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
