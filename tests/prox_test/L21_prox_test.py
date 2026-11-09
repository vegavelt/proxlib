import numpy as np
from scipy.optimize import minimize
from numpy.testing import assert_allclose
from proxlib.prox import proxL21

#=====================================================================#

def objective(x, V, lam):
    X = x.reshape(V.shape)
    return lam * np.sum(np.linalg.norm(X, axis=0)) + 0.5 * np.sum((X - V)**2)

#=====================================================================#

def test_proxL21(tol : float = 1e-2):
    V = np.random.randn(3, 4)
    lam = 0.3

    result = minimize(
        objective,
        V.flatten(),
        args=(V, lam),
        method="SLSQP",
    )
    test_value = proxL21(V, lam)
    assert_allclose(test_value, result.x.reshape(V.shape), atol=tol)

