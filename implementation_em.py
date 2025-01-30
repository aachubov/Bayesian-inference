import numpy as np
from scipy.special import logsumexp
from scipy.signal import fftconvolve

def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """

    H, W, K = X.shape
    h, w = F.shape

    ones = np.ones((h, w))
    ones_like_X = np.ones((H,W,K))


    x = fftconvolve(X**2, ones[..., None], 'valid', axes = (0,1))
    fx = fftconvolve(X, np.flip(F)[..., None], 'valid', axes = (0,1))
    f_sq = fftconvolve(ones_like_X, (np.flip(F)[..., None])**2, 'valid', axes = (0,1))

    first_term = x - 2*fx + f_sq

    matr = (X - B[..., None])**2
    total_sum = matr.sum(axis = (0,1))
    complement_sums = total_sum - fftconvolve(matr, ones[..., None], 'valid', axes = (0, 1))
    second_term = complement_sums
    

    return -1/(2*s**2)*(first_term + second_term) - H*W*(0.5 * np.log(2 * np.pi) + np.log(s))



def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    H, W, K = X.shape
    h, w = F.shape

    if use_MAP:
        mask = np.zeros((H-h+1, W-w+1, K))
        for k in range(K):
            i, j = q[:, k]
            mask[i, j, k] = 1
        
        q = mask
        
    ones = np.ones([h,w])
    probs_fa = fftconvolve(q, ones[..., None], 'full', axes = (0,1))
    probs_not_fa = 1 - probs_fa
    f_q_term = fftconvolve(q, F[..., None], 'full', axes = (0,1))

    
    first_term = (probs_fa*((X)**2)).sum() - 2*(f_q_term*X).sum() + \
                fftconvolve(q, (F[..., None])**2, 'full', axes = (0,1)).sum()
    
    sec_term = (probs_not_fa*((X - B[..., None])**2)).sum()

    lb = -1/(2*s**2)*(first_term + sec_term) - H*W*K*np.log(s) + \
            (q*np.log(A[..., None]+1e-17)).sum() - \
            (q*np.log(q + 1e-17)).sum() - H*W*K*(0.5 * np.log(2 * np.pi))
    
    return lb


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """

    K = X.shape[2]

    nominator = calculate_log_probability(X, F, B, s) + np.log(np.maximum(A[..., None], 1e-12))

    max_a = np.max(nominator, axis = (0,1))
    nominator = np.exp(nominator - max_a)
    denominator = nominator.sum(axis = (0,1))
    q = nominator/denominator
    
    if not use_MAP:
        return q

    else:
        n, m = q.shape[:2] 
        flat_q = q.reshape(-1, K)
        max_indices = np.argmax(flat_q, axis=0)
        i, j = max_indices//m, max_indices%m
    
        return np.column_stack([i, j]).T
        
        


def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    H, W, K = X.shape
    

    if not use_MAP:

        A = q.sum(axis = 2)/K
        
        kernels = q

        # F:
        F_nom = fftconvolve(X, np.flip(kernels, axis = (0,1)), mode='valid', axes = (0,1))
        F = F_nom.sum(axis = 2)/K

        # B:
        ones = np.ones([h,w])
        windows = fftconvolve(kernels, ones[..., None], 'full', axes = (0,1))
        probs = np.maximum(1e-12, (1 - windows))
        B = (X*probs).sum(axis = 2)/(probs.sum(axis = 2))

        # s:
        t = fftconvolve(kernels, F[..., None], 'full', axes = (0,1))
        ftoroy = (t*X).sum()
        pervi = (windows*((X)**2)).sum()
        treti = fftconvolve(kernels, (F[..., None])**2, 'full', axes = (0,1)).sum()
        
        first_term = pervi - 2*ftoroy + treti
        sec_term = (probs*((X - B[..., None])**2)).sum()


    else:


        H, W, K = X.shape
        A = np.zeros([H-h+1, W-w+1])
        for k in range(K):
            A[q[..., k][0], q[..., k][1]] += 1
        A = A/K
        
        
        F = np.zeros([h,w, K])
        B = np.zeros([H,W, K])
        B_denom = np.ones([H,W,K])

        for k in range(K):
            dh, dw = q[..., k]
            F[..., k] = X[dh:dh+h, dw:dw+w, k]
            B[..., k] = np.copy(X[...,k])
            B_denom[dh:dh+h, dw:dw+w, k] = 0
            B[dh:dh+h, dw:dw+w, k] = 0

        B = B.sum(axis = 2)/np.maximum(B_denom.sum(axis = 2), 1e-12)
        F = F.sum(axis = 2)/K

        first_term = 0
        sec_term = 0

        for k in range(K):
            dh, dw = q[..., k]
            first_term += ((X[dh:dh+h, dw:dw+w, k] - F)**2).sum()
            tmp = (X[..., k] -  B)**2
            tmp[dh:dh+h, dw:dw+w] = 0
            sec_term += tmp.sum()

    s = np.sqrt((first_term + sec_term)/(H*W*K)) if first_term + sec_term > 0 else 1e-17

    return F, B, s, A


def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step); 
        number_of_iters is actual number of iterations that was done.
    """

    H, W, K = X.shape

    if F is None:
        F = np.random.rand(h, w)

    if B is None:
        B = np.random.rand(H, W)

    if s is None:
        s = 1

    if A is None:
        A = np.random.rand(H - h + 1, W - w + 1)
        A /= A.sum() 


    LL = [0]*(max_iter)
    
    for i in range(max_iter):

        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)

        lb = calculate_lower_bound(X, F, B, s, A, q, use_MAP)
        
        LL[i] = lb
        if i > 0:
            if abs(LL[i] - LL[i-1]) < tolerance:
                return F, B, s, A, np.array(LL[:i+1])
        

    return F, B, s, A, np.array(LL)

    


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """

    results = [0]*(n_restarts)
    best_lb = -float('inf')
    
    for i in range(n_restarts):

        res = run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,max_iter=50, use_MAP=False)
        results[i] = (res)
        
        lb = res[-1][-1]
        if lb > best_lb:
            best_lb = lb
            best_ind = i


    return results[best_ind]