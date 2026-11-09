import numpy as np
from scipy.optimize import minimize
from numpy.testing import assert_allclose
from proxlib.prox import proxBox

#=====================================================================#

def objective(x, v):
    return 0.5 * np.sum((x - v)**2)

#=====================================================================#

def test_proxBox(tol : float = 1e-2):
    v = np.random.randn(5)
    l = -1.0
    u = 2.0
    lam = 0.3

    result = minimize(
        objective,
        v.copy(),
        args=(v,),
        bounds=[(l, u)] * len(v),
        method="SLSQP",
    )
    test_value = proxBox(v, l, u, lam)
    assert_allclose(test_value, result.x, atol=tol)


