"""Check cached implicit SGD against naive history-vector updates; check OOF encodings."""
from pathlib import Path
import ctypes,subprocess,tempfile
import numpy as np
from task3_round2 import encodings
with tempfile.TemporaryDirectory() as d:
 path=Path(d)/'joint.so';subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task3_joint_kernel.cpp')),'-o',str(path)],check=True);lib=ctypes.CDLL(str(path));ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');fn=lib.joint_train;fn.argtypes=[ctypes.c_int]*2+[ip]*6+[fp]*2+[ip]*5+[ctypes.c_float]+[fp]*11+[ctypes.c_int]*3+[ctypes.c_float]*5
 ptr=np.array([0,3],np.int32);items=np.arange(3,dtype=np.int32);code=np.zeros(3,np.int32);dv=np.full(3,.2,np.float32);y=np.full(3,4.,np.float32);ic=np.ones(3,np.int32);counts=np.array([3],np.int32)
 P=np.array([[.1,.2]],np.float32);Q=np.tile(np.array([.05,-.03],np.float32),(3,1));Y=np.tile(np.array([.02,.01],np.float32),(3,1));A=np.array([[.01,-.02]],np.float32);biases=[np.zeros(n,np.float32) for n in [1,3,1,1,1,1,1]]
 p,q,yy,a=[v.astype(float).copy() for v in [P,Q,Y,A]];bu,bi,du,bt,ut,mt,alpha=[v.astype(float).copy() for v in biases];eta=.007;reg=.05
 for j in range(3):
  z=yy.sum(0)/np.sqrt(3);eff=p[0]+z+.2*a[0];e=4-(3.5+bu[0]+bi[j]+du[0]+bt[0]+ut[0]+mt[0]+alpha[0]*.2+eff@q[j]);oldq=q[j].copy()
  bu[0]+=eta*(e-bu[0]);bi[j]+=eta*(e-5*bi[j]);du[0]+=eta*(e-10/3*du[0]);bt[0]+=eta*(e-20/3*bt[0]);ut[0]+=eta*(e-10/3*ut[0]);mt[0]+=eta*(e-100/3*mt[0]);alpha[0]+=eta*(e*.2-10/3*alpha[0]);p[0]+=eta*(e*oldq-reg*p[0]);a[0]+=eta*(e*oldq*.2-.15*a[0]);q[j]+=eta*(e*eff-reg*oldq);yy+=eta*(e*oldq/np.sqrt(3)-reg*yy)
 fn(1,2,ptr,items,code,code,code,code,dv,y,ic,counts,counts,counts,counts,3.5,P,Q,Y,A,*biases,0,1,42,eta,reg,10.,20.,10.)
 assert np.allclose(P,p,atol=1e-6) and np.allclose(A,a,atol=1e-6) and np.allclose(Y,yy,atol=1e-6)
 assert np.allclose(np.sort(Q,axis=0),np.sort(q,axis=0),atol=1e-6)
 for actual,want in zip(biases,[bu,bi,du,bt,ut,mt,alpha]):assert np.allclose(np.sort(actual),np.sort(want),atol=1e-6)
 y=np.tile(np.arange(1,6),4).astype(float);c=np.repeat(np.arange(4),5);codes=[c]*4;queries=[np.array([0,1,2,3])]*4
 original,_=encodings(codes,queries,y,np.random.RandomState(19))
 for j in [0,7,13]:
  altered=y.copy();altered[j]=6-y[j];changed,_=encodings(codes,queries,altered,np.random.RandomState(19));assert np.allclose(original[j],changed[j],atol=1e-6)
print('PASS: fast joint SVD++ equals naive per-rating implicit updates; own target excluded from numeric fold encodings.')
with tempfile.TemporaryDirectory() as d:
 path=Path(d)/'pearson.so';subprocess.run(['g++','-O3','-shared','-fPIC',str(Path(__file__).with_name('task3_pearson_kernel.cpp')),'-o',str(path)],check=True);lib=ctypes.CDLL(str(path));fn=lib.pair_pearson;ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');fn.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,ip,fp,fp,fp,fp,ip]
 ptr=np.array([0,4,8,10],np.int32);items=np.array([0,1,2,4,0,1,2,5,0,2],np.int32);y=np.array([1,4,5,1,2,3,5,4,3,2],np.float32);r=np.array([.1,-.2,.9,3,.3,-.3,.5,-1,.2,.7],np.float32);candidates=np.array([1,2],np.int32);raw=np.zeros(2,np.float32);res=raw.copy();counts=np.zeros(2,np.int32);fn(0,2,candidates,ptr,items,y,r,raw,res,counts)
 assert np.array_equal(counts,[3,2]);assert np.allclose(raw[0],np.corrcoef(y[:3],y[4:7])[0,1],atol=1e-6);assert np.allclose(res[0],np.corrcoef(r[:3],r[4:7])[0,1],atol=1e-6);assert raw[1]==0 and res[1]==0
 y[4:7]=6-y[:3];fn(0,2,candidates,ptr,items,y,r,raw,res,counts);assert np.allclose(raw[0],-1,atol=1e-6)
print('PASS: shared-book Pearson matches numpy, preserves anti-correlation and rejects insufficient overlap.')
