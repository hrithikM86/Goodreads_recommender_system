import numpy as np
from catboost import CatBoostClassifier

def design(d,s):return np.column_stack([d,s])
def margin(model,d,s):
    return model.predict_proba(design(d,s))[:,1]-model.predict_proba(design(-d,s))[:,1]

def train_stacker(d,s,output):
    idx=np.random.RandomState(17).permutation(len(d));cut=int(.8*len(idx));tr=idx[:cut];va=idx[cut:];best=None;reports={}
    for depth in [3,5]:
        x=np.vstack([design(d[tr],s[tr]),design(-d[tr],s[tr])]);y=np.r_[np.ones(len(tr)),np.zeros(len(tr))]
        v=np.vstack([design(d[va],s[va]),design(-d[va],s[va])]);vy=np.r_[np.ones(len(va)),np.zeros(len(va))]
        m=CatBoostClassifier(iterations=700,depth=depth,learning_rate=.04,l2_leaf_reg=10,loss_function='Logloss',eval_metric='Accuracy',random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
        m.fit(x,y,eval_set=(v,vy),early_stopping_rounds=75)
        score=float(np.mean(margin(m,d[va],s[va])>0));iters=m.get_best_iteration()+1
        reports[str(depth)]={'internal_selection_accuracy':score,'iterations':iters}
        print('Stacker',depth,reports[str(depth)],flush=True)
        if best is None or score>best[0]:best=(score,depth,iters)
    score,depth,iters=best
    m=CatBoostClassifier(iterations=iters,depth=depth,learning_rate=.04,l2_leaf_reg=10,loss_function='Logloss',eval_metric='Accuracy',random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
    m.fit(np.vstack([design(d,s),design(-d,s)]),np.r_[np.ones(len(d)),np.zeros(len(d))]);m.save_model(str(output/'stacker.cbm'))
    return m,{'candidates':reports,'internal_selection_accuracy':score,'depth':depth,'iterations':iters}
