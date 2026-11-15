"""Numerical checks for temporal density, context scores and target exclusion."""
from pathlib import Path
import tempfile
import numpy as np
from scipy.sparse import coo_matrix
from task4_optimize import kde_library,binary,history_features

with tempfile.TemporaryDirectory() as d:
    kde=kde_library(Path(d));x=np.array([0.,1.,1.,3.,10.]);q=np.array([-2.,0.,.5,1.,2.,10.,30.])
    for h in [.5,7.,365.]:
        expected=np.exp(-np.abs(q[:,None]-x[None,:])/h).mean(1)/(2*h)
        assert np.allclose(kde(x,q,h),expected,atol=1e-14)
        assert np.allclose(kde(x+12345,q+12345,h),expected,atol=1e-14)
    assert np.all(kde(np.array([]),q,7)==0)
    # One user, two known books at days0 and2; candidate book2 is unobserved.
    u=np.array([0,0]);i=np.array([0,1]);X=binary(u,i,(1,3));T=coo_matrix((np.array([0.,172800.]),(u,i)),shape=X.shape).tocsr()
    W=coo_matrix(([.4,.6],([0,1],[2,2])),shape=(3,3)).tocsr();E=np.zeros((3,3),np.float32);E[0,2]=.2;E[1,2]=-.1
    V=np.eye(3,dtype=np.float32);mapping=np.arange(3);f=history_features(X,T,np.array([0]),np.array([2]),np.array([86400.]),W,E,mapping,V,Path(d))
    for h in [30,180,365]:
        assert np.allclose(f['slim_time'+str(h)],[np.exp(-1/h)],atol=1e-7)
        assert np.allclose(f['ease_time'+str(h)],[.1*np.exp(-1/h)],atol=1e-7)
    assert f['history_has_prev'][0]==1 and f['history_has_next'][0]==1
    assert f['slim_prev'][0]==np.float32(.4) and f['slim_next'][0]==np.float32(.6)
    assert X[0,2]==0
print('PASS: KDE matches direct density; time translation invariant; temporal SLIM/EASE scores match manual calculation; absent target stays excluded.')
