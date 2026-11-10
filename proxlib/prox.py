import numpy as np

#=====================================================================#

def proxL1(v : np.ndarray, lam : float, conjugate : bool = False) -> np.ndarray:
    '''
    Parameters:
        v (darray)
        lam (float)
        conjugate (bool) : If true will compute the proximal of the convex conjugate

    Returns:
        Proximal operator of the L1 norm computed at v
    '''
    if (conjugate): 
        return v - lam * proxL1(v/lam, 1/lam)

    return np.sign(v)*np.maximum(0,np.abs(v) - lam)

#=====================================================================#

def proxL21(V : np.ndarray, lam : float, conjugate : bool = False) -> np.ndarray:
    '''
    Parameters:
        V (darray)
        lam (float)
        conjugate (bool) : If true will compute the proximal of the convex conjugate

    Returns:
        Proximal operator of the mixed norm L2,1 norm computed at v
    '''
    if V.ndim != 2:
        raise ValueError("V must be a 2D darray")

    if (conjugate): 
        return V - lam * proxL21(V/lam, 1/lam)

    X = np.zeros_like(V)
    cols = V.shape[1]

    for i in range(cols):
        Vi = V[:, i]
        norm_Vi = np.linalg.norm(Vi, ord=2)

        if norm_Vi > lam:
            X[:,i] = np.maximum(1 - lam/norm_Vi,0)*Vi

    return X

#=====================================================================#

def proxBox(v : np.ndarray, l : int, u : int, lam : float, conjugate : bool = False) -> np.ndarray:
    '''
    Parameters:
        v (darray)
        u (int) : Lower bound of the indicator
        l (int) : upper bound of the indicator
        lam (float)
        conjugate (bool) : If true will compute the proximal of the convex conjugate

    Returns:
        Proximal operator of the indicator box [l,v] norm computed at v
    '''
    if (conjugate): 
        return v - lam * proxBox(v/lam,l,u,1/lam)

    if np.any(np.asarray(l) > np.asarray(u)):
        raise ValueError("Every coordinate of l must be smaller than their corresponding coordinate in u")

    return np.clip(v, l, u)

#=====================================================================#

def proxBall(v : np.ndarray, c : np.ndarray, r : int, lam : float, conjugate : bool = False) -> np.ndarray:
    '''
    Parameters:
        v (darray)
        c (darray) : Center of the ball
        r (int) : radius of the ball
        lam (float)
        conjugate (bool) : If true will compute the proximal of the convex conjugate

    Returns:
        Proximal operator of the ball centered at C, of radius r norm computed at v
    '''
    if r < 0:
        raise ValueError("r must be non negative")

    if c.shape != v.shape:
        raise ValueError("c and v must be of same dimension")

    if (conjugate): 
        return v - lam * proxBall(v/lam,c,r,1/lam)
    
    norm_v_minus_c = np.linalg.norm(v-c, ord=2)

    if (norm_v_minus_c <= r):
        return v
    else:
        return c + (r/norm_v_minus_c) * (v - c)
    
#=====================================================================#

def proxProbSimplex(v : np.ndarray, lam : float = 1, conjugate : bool = False) -> np.ndarray:
    '''
    Parameters:
        v (darray)
        lam (float) (Solution is independent of lam)
        conjugate (bool) : If true will compute the proximal of the convex conjugate

    Returns:
        Proximal operator of the probability simplex at v (Projection on the simplex)
        Algorithm taken from https://arxiv.org/pdf/1309.1541
    '''
    if (conjugate): 
        return v - lam * proxProbSimplex(v/lam,1/lam)

    v_sorted = np.sort(v)[::-1]

    # The first element is equal to 1 (> 0), so we can initialize it at 0 safely.
    rho = 0
    sum_vj = 0
    sum_vj_to_rho = 1

    for j in range(len(v)):
        vj = v_sorted[j]
        sum_vj += vj

        if (vj + 1/(j+1) * (1 - sum_vj) > 0):
            rho = j+1
            sum_vj_to_rho = sum_vj

    lam2 = 1/rho * (1 - sum_vj_to_rho)
    return np.maximum(v + lam2, 0)
