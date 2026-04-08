import numpy as np
import ot as pot

def prior_ot_fn(
    a,
    b,
    M,
    reg=0.1,
    prior_method='to_first'
):
    '''
    Implement a prior-based optimal transport method. This is a placeholder
    function and should be implemented based on the specific prior method you want to use.

    The output is then a plan G which represents the optimal transport plan.
    This optimal transport plan should be of size (a, b), representing the OT plan.
    
    Use this: https://pythonot.github.io/_modules/ot/lp/_network_simplex.html#emd as an example.

    These should be taking as argument and outputting numpy arrays.
    '''
    Q = np.zeros((a.shape[0], b.shape[0])) 
    # Prior cost matrix, which can be used to encode the prior information about the transport plan.
    if prior_method == 'basic_entropic_ot':
        Q = basic_entropic_ot_plan(a, b)
    if prior_method == 'to_first':
        Q = to_first_ot_plan(a, b)
    
    #Ensure no zero entries in Q to avoid log(0) issues.
    Q = clip_matrix(Q)

    M_adjusted = M - reg * np.log(Q)
    M_adjusted = clip_matrix(M_adjusted)

    P =  pot.sinkhorn(a, b, M_adjusted, reg=reg)
    P = clip_matrix(P)
    #raise ValueError(f'Prior method {prior_method} not implemented')
    return P


def to_first_ot_plan(a, b):
    '''
    Example OT method where we simply force it all to the very first cell in b.
    i.e. this is [1, 0, ..., 0] where these are represented as column vectors.
    '''
    Q = np.zeros((a.shape[0], b.shape[0]))
    Q[:, 0] = 1
    return Q

def basic_entropic_ot_plan(a, b):
    '''
    OT method which should be equivalent to the standard entropic OT plan without any prior information.
    Qe = a ⊗ b, is the outer product of a and b. 
    '''
    Q = np.outer(a,b)
    return Q

def clip_matrix(M, eps=1e-8):
    '''
    Clip the cost matrix M to ensure numerical stability.
    This is important to avoid issues with log(0) when adjusting the cost matrix with the prior.
    '''
    M_clipped = np.maximum(M, eps)
    M_normalized = M_clipped / np.sum(M_clipped)
    return M_normalized