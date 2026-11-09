"""Small leakage-sensitive tests for next-book date summaries."""
import numpy as np
from task6_item_time import build_item_features

def main():
    items=np.array([0,0,0]);dates=np.array([100.,200.,400.])*86400;query=np.array([150.])*86400
    f=build_item_features(items,dates,np.ones(3,dtype=bool),np.array([0]),query,np.array([1]))
    assert np.isclose(np.expm1(f[0,0]),2)
    assert np.isclose(np.expm1(f[0,10]),250,rtol=1e-6)
    assert np.isclose(np.expm1(f[0,4]),100,rtol=1e-6)
    # Globally held-out date must yield the same result without a self-removal hint.
    g=build_item_features(items,dates,np.array([True,False,True]),np.array([0]),query)
    assert np.allclose(f,g,equal_nan=True)
    singleton=build_item_features(np.array([0]),np.array([200.])*86400,np.array([True]),np.array([0]),query,np.array([0]))
    assert singleton[0,14]==1 and singleton[0,0]==0
    print('Item-time leakage checks passed: own target excluded, held-out mask honored, singleton unknown.')

if __name__=='__main__':main()
