import numpy as np
import ot as pot

def prior_ot_fn(
    a,
    b,
    M,
    reg,
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
    
    #Ensure no zero entries in Q to avoid log(0) issues, and normalize Q to be a valid probability distribution.
    eps = 1e-8
    Q = np.maximum(Q, eps)
    Q = Q / Q.sum()
    
    M_adjusted = M - reg * np.log(Q)

    #raise ValueError(f'Prior method {prior_method} not implemented')
    return pot.sinkhorn(a, b, M_adjusted, reg=reg)

def to_first_ot_plan(a, b):
    '''
    Example OT method where we simply force it all to the very first cell in b.
    i.e. this is [1, 0, ..., 0] where these are represented as column vectors.
    '''
    Q = np.zeros((a.shape[0], b.shape[0]))
    Q[:, 0] = 1
    return Q

def basic_entropic_ot_plan(a, b):
    Q = np.outer(a,b)
    return Q
