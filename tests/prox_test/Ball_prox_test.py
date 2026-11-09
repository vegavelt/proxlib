import numpy as np
from scipy.optimize import minimize
from numpy.testing import assert_allclose
from proxlib.prox import proxBall

#=====================================================================#

def objective(x, v):
    return 0.5 * np.sum((x - v)**2)

#=====================================================================#

def ball_constraint(x, c, r):
    return r**2 - np.sum((x - c)**2)

#=====================================================================#

def test_proxBall(tol : float = 1e-2):
    v = np.random.randn(3)
    c = np.random.randn(3)
    r = 1.0
    lam = 0.3

    result = minimize(
        objective,
        v.copy(),
        args=(v,),
        constraints={
            "type": "ineq",
            "fun": ball_constraint,
            "args": (c, r),
        },
        method="SLSQP",
    )
    test_value = proxBall(v, c, r, lam)
    assert_allclose(test_value, result.x, atol=tol)

#=====================================================================#
