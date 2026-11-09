import numpy as np
from scipy.optimize import minimize
from numpy.testing import assert_allclose
from proxlib.prox import proxProbSimplex

#=====================================================================#

def objective(x, v):
    return 0.5 * np.sum((x - v)**2)

#=====================================================================#

def simplex_constraint(x):
    return np.sum(x) - 1

#=====================================================================#

def test_proxProbSimplex(tol : float = 1e-2):
    v = np.random.randn(5)
    lam = 0.3

    result = minimize(
        objective,
        np.ones(len(v)) / len(v),
        args=(v,),
        bounds=[(0, None)] * len(v),
        constraints={
            "type": "eq",
            "fun": simplex_constraint,
        },
        method="SLSQP",
    )
    test_value = proxProbSimplex(v, lam)
    assert_allclose(test_value, result.x, atol=tol)