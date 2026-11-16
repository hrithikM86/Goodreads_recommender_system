"""Read-only Task5 validation audit. Test inputs only; no test labels inferred or used."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from task5_improve import data
W=Path('/workspace/work');DATA=W/'Goodreads_recommender_system/Assignment 1/assignment1_2026';OUT=W/'task5_validation_audit';OUT.mkdir(exist_ok=True)
def quant(x):return {str(p):float(np.quantile(x,p)) for p in [0,.1,.25,.5,.75,.9,.99,1]}
def dist(x):
 bins=[0,1,2,5,10,20,50,100,500,1000,np.inf];return {str(bins[k])+'..'+str(bins[k+1]):float(np.mean((x>=bins[k])&(x<bins[k+1]))) for k in range(len(bins)-1)}
def describe(pop,qi,contexts):
 known=qi>=0;freq=np.zeros(len(qi));freq[known]=pop[qi[known]];seen=np.array([bool(j>=0 and j in h) for j,h in zip(qi,contexts)]);length=np.array(list(map(len,contexts)));lastpop=np.array([pop[h[-1]] for h in contexts]);r={'n':len(qi),'unknown_ID_rate':float((~known).mean()),'cold_fitting_count_zero_rate':float((freq==0).mean()),'candidate_seen_in_query_history_rate':float(seen.mean()),'candidate_training_count':quant(freq),'count_bins':dist(freq),'history_length':quant(length),'last_context_book_cold_rate':float((lastpop==0).mean()),'last_context_book_count_le1_rate':float((lastpop<=1).mean())};return r,freq,seen,length,lastpop

def main():
 p=pd.read_csv(DATA/'task_5_train.csv',usecols=['user','item']);u,users=pd.factorize(p.user);i,items=pd.factorize(p.item);U,I=len(users),len(items);ptr=np.searchsorted(u,np.arange(U+1));h=[i[ptr[k]:ptr[k+1]].astype('i') for k in range(U)];train,ctx,qu,qi,UU,II,_=data(DATA,W/'task5/results',False);assert (UU,II)==(U,I);prefix=np.concatenate(train);vp=np.bincount(prefix,minlength=I);fp=np.bincount(i,minlength=I);q=pd.read_csv(DATA/'task_5_test.csv');tu=pd.Index(users).get_indexer(q.user);t1=pd.Index(items).get_indexer(q.item1);t2=pd.Index(items).get_indexer(q.item2);N=len(q);assert (tu>=0).all();tctx=[h[v] for v in tu];sections={};raw={}
 for name,candidates,contexts in [('selection_positive',qi[:U],train),('selection_negative',qi[2*U:3*U],train),('evaluation_positive',qi[U:2*U],ctx[U:]),('evaluation_negative',qi[3*U:],ctx[U:])]:sections[name],*raw[name]=describe(vp,candidates,contexts)
 for name,candidates in [('test_item1',t1),('test_item2',t2)]:sections[name],*raw[name]=describe(fp,candidates,tctx)
 # Label-free candidate mixtures and pair invariants.
 evalfreq=np.r_[raw['evaluation_positive'][0],raw['evaluation_negative'][0]];testfreq=np.r_[raw['test_item1'][0],raw['test_item2'][0]];evseen=np.r_[raw['evaluation_positive'][1],raw['evaluation_negative'][1]];tseen=np.r_[raw['test_item1'][1],raw['test_item2'][1]]
 def pairfeatures(f1,f2,s1,s2,ln,last):return np.column_stack([np.log1p(np.minimum(f1,f2)),np.log1p(np.maximum(f1,f2)),abs(np.log1p(f1)-np.log1p(f2)),(f1==0).astype(int)+(f2==0).astype(int),s1.astype(int)+s2.astype(int),np.log1p(ln),np.log1p(last),last==0])
 ef=pairfeatures(raw['evaluation_positive'][0],raw['evaluation_negative'][0],raw['evaluation_positive'][1],raw['evaluation_negative'][1],raw['evaluation_positive'][2],raw['evaluation_positive'][3]);tf=pairfeatures(raw['test_item1'][0],raw['test_item2'][0],raw['test_item1'][1],raw['test_item2'][1],raw['test_item1'][2],raw['test_item1'][3]);reader_split=np.random.RandomState(419).permutation(U);fitreaders=reader_split[:int(.8*U)];checkreaders=reader_split[int(.8*U):];efit=np.isin(np.arange(U),fitreaders);tfit=np.isin(tu,fitreaders);model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=10,random_state=419);model.fit(np.r_[ef[efit],tf[tfit]],np.r_[np.zeros(efit.sum()),np.ones(tfit.sum())]);prob=model.predict_proba(np.r_[ef[~efit],tf[~tfit]])[:,1];auc=roc_auc_score(np.r_[np.zeros((~efit).sum()),np.ones((~tfit).sum())],prob)
 # Compare observable candidate marginals to several negative samplers; no candidate is labeled in test.
 rng=np.random.RandomState(421);pos=qi[U:2*U];candidates={}
 for mode in ['event_popularity_allow_seen','uniform_active_allow_seen','event_popularity_reject_context_seen','uniform_active_reject_context_seen']:
  active=np.flatnonzero(vp>0);neg=np.empty(U,'i')
  for v in range(U):
   while True:
    j=prefix[rng.randint(len(prefix))] if mode.startswith('event') else active[rng.randint(len(active))]
    if j!=pos[v] and (not 'reject' in mode or j not in ctx[U+v]):break
   neg[v]=j
  freq=vp[neg];mix=np.r_[vp[pos],freq];candidates[mode]={'negative_seen_rate':float(np.mean([j in hh for j,hh in zip(neg,ctx[U:])])),'mixture_cold_rate':float((mix==0).mean()),'mixture_seen_rate':float(np.mean([j in hh for j,hh in zip(neg,ctx[U:])]))/2,'mixture_log_count_KS_vs_test':float(ks_2samp(np.log1p(mix),np.log1p(testfreq)).statistic)}
 # Paired local changes and approximate IID standard errors, not independent final-test guarantees.
 ref=np.load(W/'task5_round3/preference/evaluation.npz')['primary'];base=(ref>0).astype(float)+.5*(ref==0);comparisons={}
 for name,path,key,pub in [('catboost',W/'task5_boosting/catboost_evaluation.npz','margin',.9205862696748324),('xgboost',W/'task5_boosting/xgboost_evaluation.npz','margin',.9200101398843131),('lightgbm',W/'task5_boosting/lightgbm_evaluation.npz','margin',.920125365842417),('item2vec',W/'task5_item2vec/evaluation.npz','margin',.9217385292558707)]:
  m=np.load(path)[key];correct=(m>0).astype(float)+.5*(m==0);delta=correct-base;se=delta.std(ddof=1)/np.sqrt(U);comparisons[name]={'local_accuracy':float(correct.mean()),'local_delta':float(delta.mean()),'paired_discordant_rows':int((delta!=0).sum()),'approx_IID_95pct_delta_interval':[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)],'public_accuracy':pub,'public_delta':pub-.9219689811720785}
 # Audit inner split/mirroring and graph refresh state.
 split=np.random.RandomState(17).permutation(U);tr,va=split[:int(.8*U)],split[int(.8*U):];assert not np.intersect1d(tr,va).size;assert len(np.unique(q.user))==N;assert len(set(zip(p.user,p.item)))==len(p)
 leak=0
 for row in range(2*U):leak+=int(qi[row] in ctx[row])
 r={'users':U,'items_in_full_training':I,'test_pairs':N,'test_unique_users':q.user.nunique(),'train_rows':len(p),'prefix_graph_rows':len(prefix),'sections':sections,'mixture_comparison':{'validation_cold_rate':float((evalfreq==0).mean()),'test_cold_rate':float((testfreq==0).mean()),'validation_seen_rate':float(evseen.mean()),'test_seen_rate':float(tseen.mean()),'candidate_log_count_KS':float(ks_2samp(np.log1p(evalfreq),np.log1p(testfreq)).statistic),'domain_classifier_AUC':float(auc),'domain_classifier_protocol':'Test vs historical-evaluation INPUTS only. Reader-disjoint80/20 split, no raw IDs, no target labels. Features are pair popularity min/max/imbalance, cold/seen counts, history length and last-book training count.'},'alternative_sampler_diagnostics':candidates,'paired_model_comparisons':comparisons,'checks':{'own_target_in_query_history_rows':leak,'selection_inner_reader_overlap':0,'mirroring_after_split':True,'user_item_training_rows_unique':True,'test_user_coverage_exact':bool(set(tu)==set(range(U)))},'global_graph_protocol':{'selection_graph':'H[:-2]','selection_context':'H[:-2]','evaluation_graph':'H[:-2] (not refreshed with known penultimate events)','evaluation_context':'H[:-1]','test_graph':'H','test_context':'H'},'limits':'No test labels available. Marginal shifts and domain AUC do not identify the true test negative sampler or prove the cause of public changes. Paired intervals assume IID user-pairs and ignore repeated experiment/model selection, shared item dependence and public-subset composition.'};(OUT/'results.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='sections'},indent=2));print('SECTION_SUMMARY',json.dumps({k:{x:v[x] for x in ['cold_fitting_count_zero_rate','candidate_seen_in_query_history_rate','last_context_book_cold_rate']} for k,v in sections.items()},indent=2))
if __name__=='__main__':main()
