import numpy as np

def prior_ot_fn(
    a,
    b,
    M,
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
    if prior_method == 'to_first':
        return to_first_ot_plan(a, b)

    raise ValueError(f'Prior method {prior_method} not implemented')

def to_first_ot_plan(a, b):
    '''
    Example OT method where we simply force it all to the very first cell in b.
    i.e. this is [1, 0, ..., 0] where these are represented as column vectors.
    '''
    G = np.zeros((a.shape[0], b.shape[0]))
    G[:, 0] = 1
    return G
