"""Check custom SLIM optimization and ensemble candidate-order symmetry."""
import ctypes,subprocess,tempfile
from pathlib import Path
import numpy as np
from sklearn.linear_model import ElasticNet
from catboost import CatBoostClassifier
from task2_optimize import design,predict

with tempfile.TemporaryDirectory() as d:
    libpath=Path(d)/'slim.so'
    subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task2_slim_kernel.cpp')),'-o',str(libpath)],check=True)
    lib=ctypes.CDLL(str(libpath));fn=lib.slim_fit
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');fn.argtypes=[ctypes.c_int,ptr,ptr,ptr];fn.restype=ctypes.c_double
    rng=np.random.RandomState(4);X=(rng.rand(120,8)<.3).astype(float);y=(rng.rand(120)<.2).astype(float)
    G=np.ascontiguousarray(X.T@X);rhs=np.ascontiguousarray(X.T@y);B=np.zeros((8,2));fn(8,G,rhs,B)
    for v,l1 in enumerate([.2,2.]):
        model=ElasticNet(alpha=(l1+10)/len(y),l1_ratio=l1/(l1+10),positive=True,fit_intercept=False,tol=1e-10,max_iter=10000).fit(X,y)
        assert np.max(np.abs(B[:,v]-model.coef_))<2e-5
    # EASE closed form satisfies zero diagonal and conditional ridge optimum.
    gram=X.T@X;reg=50;P=np.linalg.inv(gram+reg*np.eye(8));E=-P/np.diag(P)[None,:];np.fill_diagonal(E,0)
    for j in range(8):
        nb=np.array([k for k in range(8) if k!=j]);exact=np.linalg.solve(gram[np.ix_(nb,nb)]+reg*np.eye(7),gram[nb,j]);assert np.allclose(E[nb,j],exact)
    d=rng.randn(100,5);s=rng.randn(100,5)
    model=CatBoostClassifier(iterations=10,depth=2,verbose=False,allow_writing_files=False,thread_count=1)
    model.fit(np.r_[design(d,s),design(-d,s)],np.r_[np.ones(100),np.zeros(100)])
    assert np.allclose(predict(model,d,s),-predict(model,-d,s))
print('PASS: SLIM matches ElasticNet; EASE satisfies ridge optimum; nonlinear scores reverse under candidate swap.')
