"""Check residual MF gradients, missing-ID fallback and rating context."""
import tempfile
from pathlib import Path
import numpy as np,pandas as pd
from task3_optimize import mf_library,context_features
with tempfile.TemporaryDirectory() as d:
 lib=mf_library(Path(d));u=np.array([0],np.int32);i=u.copy();period=u.copy();dev=np.array([.5],np.float32);scale=np.array([.7],np.float32);y=np.array([.9],np.float32)
 P=np.array([[.1,.2]],np.float32);Q=np.array([[.3,.4]],np.float32);A=np.array([[.01,-.02]],np.float32);S=np.array([[.02,.01]],np.float32)
 p,q,a,s=[x.astype(float).copy() for x in [P,Q,A,S]];eff=p+.5*a+.7*s;e=.9-float(np.sum(eff*q));rate=.01
 expected=[p+rate*(e*q-.06*p),a+rate*(e*q*.5-.3*a),s+rate*(e*q*.7-.2*s),q+rate*(e*eff-.06*q)]
 lib.train_mf(1,2,u,i,period,dev,scale,y,P,A,S,Q,1,0,1,42,rate,.06,.06,.3,.2)
 for actual,want in zip([P,A,S,Q],expected):assert np.allclose(actual,want,atol=1e-7)
 pred=np.zeros(1,np.float32);counts=np.ones(1,np.int32);lib.predict_mf(1,2,1,1,u,i,period,dev,scale,counts,counts,P,A,S,Q,1,pred);assert np.allclose(pred,np.sum((P+.5*A+.7*S)*Q),atol=1e-7)
 missing=np.array([-1],np.int32);lib.predict_mf(1,2,1,1,missing,i,missing,dev,scale,counts,counts,P,A,S,Q,1,pred);assert pred[0]==0
 train=pd.DataFrame(dict(user=['u']*3,item=['a','b','c'],timestamp=[0,86400,172800],rating=[1,3,5]));query=pd.DataFrame(dict(user=['u'],item=['d'],timestamp=[86400]))
 f=context_features(train,query,np.array([3.,3.,3.]),np.array([3.]));assert f['user_mean0'][0]==3 and np.allclose(f['user_count0'][0],np.log(2));assert f['user_mean1'][0]==3 and np.allclose(f['user_count1'][0],np.log(4));assert f['user_prev_rating'][0]==1 and f['user_next_rating'][0]==3
print('PASS: temporal MF SGD matches analytic gradients; prediction and missing-ID fallback correct; local rating context matches hand calculation.')
