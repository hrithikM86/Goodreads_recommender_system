from task3_paths import DATA,ART
import os,sys,json
os.environ['OPENBLAS_NUM_THREADS']='1'
# scikit-surprise is installed in the active environment.
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from task3_components import generate,extra_features
root=DATA
p=pd.read_csv(root/'tasks_3_4_train.csv');p=p[p.rating>0].reset_index(drop=True);test=pd.read_csv(root/'task_3_test.csv')
configs=json.load(open(str(ART/'configs.json')));selection=json.load(open(str(ART/'selection.json')))
components=generate(p,test,configs)
if selection['chosen']=='meta':
 metadata=json.load(open(str(ART/'meta_results.json')));A=np.column_stack([components[k] for k in metadata['base_columns']]);A=np.column_stack([A,extra_features(p,test)])
 model=CatBoostRegressor();model.load_model(str(ART/'meta_')+str(selection['meta_depth'])+'.cbm');pred=np.clip(model.predict(A),1,5)
else:
 weights=json.load(open(str(ART/'blend_results.json')))['weights'];pred=np.clip(sum(weight*components[name] for name,weight in weights.items()),1,5)
assert len(pred)==len(test)==20000 and np.isfinite(pred).all()
test['rating']=pred;test.to_csv(str(ART/'task_3_predictions.csv'),index=False)
print('Saved 20,000 Task 3 predictions',flush=True)
