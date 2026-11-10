import numpy as np

def ISTA(A, b, lambd, L, F, grad_f, prox, max_iter = 1000, x0 = None):
    if x0 is None:
        x0 = np.zeros(len(A[0]))
    x = x0
    F_values = []
    F_values.append(F(x0,A,b,lambd))

    for i in range(max_iter):
        g = grad_f(x,A,b)
        x = prox(x - g/L, lambd/L)
        F_values.append(F(x,A,b,lambd))
    return x, np.array(F_values)
    
def FISTA(A, b, lambd, L, F, grad_f, prox, max_iter = 1000, x0 = None):
    if x0 is None:
        x0 = np.zeros(len(A[0]))
    x = x0
    x_prev = x0
    y = x0
    t = 1
    F_values = []
    F_values.append(F(x,A,b,lambd))

    for i in range(max_iter):
        g = grad_f(y,A,b)
        x = prox(y-g/L,lambd/L)
        t_n = (1 + np.sqrt(1+4*t**2))/2
        y = x + (t-1)/t_n * (x - x_prev)
        x_prev = x
        t = t_n
        F_values.append(F(x,A,b,lambd))
    return x, np.array(F_values)

def metrics(x_hat, x_true, tol = 1e-8):
    idx_hat = np.flatnonzero(np.abs(x_hat) > tol )
    idx_true = np.flatnonzero(np.abs(x_true) > tol)
    TP = len(set(idx_hat) & set(idx_true) )
    FP = len(set(idx_hat) - set(idx_true))
    FN = len(set(idx_true) - set(idx_hat))
    return TP, FP, FN