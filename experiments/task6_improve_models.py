"""Selection-only MAE model/blend tuning; separate final-event evaluation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize
from catboost import CatBoostRegressor

def model(name,iters=1800,threads=1):
    if name=='cat':return CatBoostRegressor(iterations=iters,depth=6,learning_rate=.04,l2_leaf_reg=5,random_seed=23,loss_function='MAE',thread_count=threads,verbose=300,allow_writing_files=False,cat_features=['user','next_item'],max_ctr_complexity=1)
    from lightgbm import LGBMRegressor
    return LGBMRegressor(objective='regression_l1',n_estimators=iters,num_leaves=31 if name=='lgb' else 63,learning_rate=.04,min_child_samples=200,reg_lambda=50,max_bin=127,colsample_bytree=.9,subsample=.9,subsample_freq=1,path_smooth=10,n_jobs=1,random_state=42,verbosity=-1)
def framing(X,p,rows=None,test=None):
    f=pd.DataFrame(X,columns=['f'+str(j) for j in range(X.shape[1])]);source=p.iloc[rows] if test is None else test;f['user']=source.user.to_numpy();f['next_item']=source.item.to_numpy();return f
def old_prediction(legacy):
    from sklearn.linear_model import Ridge
    z=np.load(legacy/'task6_rich.npz');old=np.load(legacy/'task6_final/predictions.npz');w=json.loads((legacy/'task6_final/blend_results.json').read_text())['weights'];ridge=Ridge(alpha=1.).fit(np.expm1(z['X'][:,[0,19,22]]).astype(float),z['y']);out=[]
    for prefix,Q in [('v_',z['V']),('t_',z['T'])]:
        preds={name:old[prefix+name] for name in ['book_mae','book_log']};preds['ridge']=np.maximum(0,ridge.predict(np.expm1(Q[:,[0,19,22]]).astype(float)));out.append(sum(w[name]*preds[name] for name in w))
    return out
def user_weights(p,rows):
    user=pd.factorize(p.user.iloc[rows])[0];position=p.groupby('user',sort=False).cumcount().to_numpy()[rows];last=np.zeros(user.max()+1);np.maximum.at(last,user,position);raw=1+4*(position/np.maximum(last[user],1))**2;total=np.bincount(user,weights=raw);return raw/total[user]*(len(rows)/len(total))
def full_fit(a,name,iterations,p,full,progress,test):
    M=model(name,iterations,threads=2);XX=full['X'];QQ=full['Q'];path=a.out/(name+'_full.model');weight=1+4*progress**2 if name=='recent' else (user_weights(p,full['rows']) if name=='balanced' else None)
    if name=='cat':XX=framing(XX,p,full['rows']);QQ=framing(QQ,p,test=test)
    if path.exists():
        if name=='cat':M.load_model(str(path))
        else:
            import lightgbm as lgb;M=lgb.Booster(model_file=str(path))
    else:
        if name=='cat':M.fit(XX,full['y']);M.save_model(str(path))
        else:M.fit(XX,full['y'],sample_weight=weight);M.booster_.save_model(str(path))
    return np.maximum(0,M.predict(QQ))
def main():
    ap=argparse.ArgumentParser()
    for key in ['data-dir','legacy','out','accepted']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--predict',action='store_true');ap.add_argument('--only-model',choices=['lgb','recent','balanced','cat']);a=ap.parse_args();z=np.load(a.out/'validation.npz');X=z['X'];y=z['y'];rows=z['rows'];n=int(z['n']);m=int(z['m']);p=pd.read_csv(a.data_dir/'task_6_train.csv');V=X[n:n+m];T=X[n+m:];vy=y[n:n+m];ty=y[n+m:];rv,rt=old_prediction(a.legacy);mae=lambda y,p:float(np.mean(np.abs(y-p))*86400);assert abs(mae(ty,rt)-7457904.6219744105)<.1;predv={};predt={};runs={}
    for name in ([a.only_model] if a.only_model else ['lgb','recent','balanced','cat']):
        path=a.out/(name+'_validation.model');info=a.out/(name+'_info.json');M=model(name)
        trainX=X[:n];queryV=V;queryT=T;weight=None
        if name=='recent':weight=1+4*z['progress'][:n]**2
        if name=='balanced':weight=user_weights(p,rows[:n])
        if name=='cat':trainX=framing(trainX,p,rows[:n]);queryV=framing(V,p,rows[n:n+m]);queryT=framing(T,p,rows[n+m:])
        if path.exists():
            r=json.loads(info.read_text())
            if name=='cat':M.load_model(str(path))
            else:
                import lightgbm as lgb;M=lgb.Booster(model_file=str(path))
        else:
            if name=='cat':M.fit(trainX,y[:n],eval_set=(queryV,vy),early_stopping_rounds=125);iters=M.tree_count_;M.save_model(str(path))
            else:
                import lightgbm as lgb;M.fit(trainX,y[:n],sample_weight=weight,eval_set=[(queryV,vy)],callbacks=[lgb.early_stopping(125,verbose=False),lgb.log_evaluation(300)]);iters=M.best_iteration_;M.booster_.save_model(str(path))
            r={'iterations':int(iters),'recent_weighted':name=='recent'};info.write_text(json.dumps(r,indent=2))
        predv[name]=np.maximum(0,M.predict(queryV));predt[name]=np.maximum(0,M.predict(queryT));r.update(selection_MAE=mae(vy,predv[name]),evaluation_MAE=mae(ty,predt[name]));runs[name]=r;print('TASK6',name,r,flush=True)
    if a.only_model:
        if a.predict:
            full=np.load(a.out/'full.npz');test=pd.read_csv(a.data_dir/'task_6_test.csv');pred=full_fit(a,a.only_model,runs[a.only_model]['iterations'],p,full,z['progress'],test);np.save(a.out/(a.only_model+'_full_predictions.npy'),pred)
        return
    # The new-only blend and accepted-reference blend are tuned on selection only.
    def fit(names,P):
        obj=lambda w:np.mean(np.abs(P@w-vy));best=None
        for init in [np.ones(len(names))/len(names)]+list(np.eye(len(names))):
            r=minimize(obj,init,method='SLSQP',bounds=[(0,1)]*len(names),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':200,'ftol':1e-8})
            if best is None or r.fun<best.fun:best=r
        w=np.maximum(best.x,0);w[w<1e-6]=0;w/=w.sum();return dict(zip(names,w.tolist()))
    names=list(runs);nw=fit(names,np.column_stack([predv[k] for k in names]));allnames=['accepted']+names;P=np.column_stack([rv]+[predv[k] for k in names]);weights=fit(allnames,P);newv=sum(nw[k]*predv[k] for k in nw);newt=sum(nw[k]*predt[k] for k in nw);dv={'accepted':rv,**predv};dt={'accepted':rt,**predt};blendv=sum(weights[k]*dv[k] for k in weights);blendt=sum(weights[k]*dt[k] for k in weights);summary={'runs':runs,'new_weights':nw,'blend_weights':weights,'reference_selection_MAE':mae(vy,rv),'reference_evaluation_MAE':mae(ty,rt),'new_selection_MAE':mae(vy,newv),'new_evaluation_MAE':mae(ty,newt),'blend_selection_MAE':mae(vy,blendv),'blend_evaluation_MAE':mae(ty,blendt),'reference_public_MAE':8958810.647357272,'input_manifest':json.loads((a.out/'manifest.json').read_text()),'train_rows':n,'selection_rows':m,'evaluation_rows':len(ty)};np.savez_compressed(a.out/'evaluation.npz',y=ty,reference=rt,new=newt,blend=blendt);(a.out/'results.json').write_text(json.dumps(summary,indent=2));print('TASK6_LOCAL',summary,flush=True)
    if not a.predict:return
    full=np.load(a.out/'full.npz');Q=full['Q'];test=pd.read_csv(a.data_dir/'task_6_test.csv');reference=pd.read_csv(a.accepted);pd.testing.assert_frame_equal(reference.drop(columns='delta'),test.drop(columns='delta'));preds={}
    for name in names:
        if nw[name]==0 and weights[name]==0:preds[name]=np.zeros(len(Q));continue
        preds[name]=full_fit(a,name,runs[name]['iterations'],p,full,z['progress'],test);print('TASK6 full',name,'ready',flush=True)
    new=sum(nw[k]*preds[k] for k in nw)*86400;preds['accepted']=reference.delta.to_numpy()/86400;blend=sum(weights[k]*preds[k] for k in weights)*86400
    for label,pred in [('new',new),('blend',blend)]:
        assert np.isfinite(pred).all() and np.min(pred)>=0;folder=a.out/label;folder.mkdir(exist_ok=True);frame=test.copy();frame['delta']=pred;frame.to_csv(folder/'task_6_predictions.csv',index=False)
    np.savez_compressed(a.out/'test_predictions.npz',new=new,blend=blend,**{k:val*86400 for k,val in preds.items()})
if __name__=='__main__':main()
