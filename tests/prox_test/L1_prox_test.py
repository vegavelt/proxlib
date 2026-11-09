import numpy as np
from scipy.optimize import minimize
from numpy.testing import assert_allclose
from proxlib.prox import proxL1

#=====================================================================#

def objective(x, v, lam):
    return lam * np.sum(np.abs(x)) + 0.5 * np.sum((x - v)**2)

#=====================================================================#

def conjugate_objective(x, v):
    return 0.5 * np.sum((x - v)**2)

#=====================================================================#

def test_proxL1(tol : float = 1e-2):
    v = np.random.randn(5)
    lam = 0.3

    result = minimize(
        objective,
        v.copy(),
        args=(v, lam),
        method="SLSQP",
    )
    test_value = proxL1(v, lam)
    assert_allclose(test_value, result.x, atol=tol)

#=====================================================================#

def test_proxL1_conjugate(tol : float = 1e-2):
    v = np.random.randn(5)
    lam = 0.3

    result = minimize(
        conjugate_objective,
        v.copy(),
        args=(v,),
        bounds=[(-1, 1)] * len(v),
        method="SLSQP",
    )
    test_value = proxL1(v, lam, conjugate=True)
    assert_allclose(test_value, result.x, tol)