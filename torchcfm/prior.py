import numpy as np
import ot as pot
import torch
import scipy as sp

def prior_ot_fn(
    a,
    b,
    M,
    reg=0.1,
    prior_method='to_first',
    D=None,
    x0=None,
    x1=None,
    y0=None,
    y1=None,
    p0=None,
    p1=None,
    profile_sigma=None,
):
    '''
    Implement a prior-based optimal transport method. This is a placeholder
    function and should be implemented based on the specific prior method you want to use.

    The output is then a plan G which represents the optimal transport plan.
    This optimal transport plan should be of size (a, b), representing the OT plan.
    
    Use this: https://pythonot.github.io/_modules/ot/lp/_network_simplex.html#emd as an example.

    These should be taking as argument and outputting numpy arrays.

    Then, we have x0, x1 being the original data matrices for the two time points,
    and D being the spatial distance matrix between the cells in x0 and x1.
    '''
    Q = np.zeros((a.shape[0], b.shape[0])) 
    # Prior cost matrix, which can be used to encode the prior information about the transport plan.
    if prior_method == 'basic_entropic_ot':
        Q = basic_entropic_ot_plan(a, b)
    elif prior_method == 'to_first':
        Q = to_first_ot_plan(a, b)
    elif prior_method == 'spatial':
        #To do : add safeguard for nonetype for D
        Q = get_spatial_prior(D)
    elif prior_method == 'spatial_knn_expression':
        if p0 is None or p1 is None:
            raise ValueError(
                "spatial_knn_expression prior requires sampled neighborhood profiles p0 and p1."
            )
        Q = get_spatial_knn_expression_prior(p0, p1, sigma=profile_sigma)
    elif prior_method == 'pseudotime_sigmoid':
        Q = get_pseudotime_sigmoid_prior(y0, y1)
    elif prior_method == 'pseudotime_uniform':
        if y0 is None or y1 is None:
            raise ValueError(
                "pseudotime_uniform prior requires precomputed y0 and y1 labels."
            )
        Q = get_pseudotime_prior_uniform(y0, y1)
    elif prior_method == 'pseudotime_gaussian':
        if y0 is None or y1 is None:
            raise ValueError(
                "pseudotime_gaussian prior requires precomputed y0 and y1 labels."
            )
        Q = get_pseudotime_prior_gaussian(y0, y1)
    elif prior_method == 'pseudotime_gamma':
        if y0 is None or y1 is None:
            raise ValueError(
                "pseudotime_gamma prior requires precomputed y0 and y1 labels."
            )
        Q = get_pseudotime_prior_gamma(y0, y1)
    else:
        raise ValueError(f"Unknown prior method: {prior_method}")
    
    #Ensure no zero entries in Q to avoid log(0) issues.
    Q = clip_matrix(Q)

    M_adjusted = M - reg * np.log(Q)
    M_adjusted = clip_matrix(M_adjusted)

    P =  pot.sinkhorn(a, b, M_adjusted, reg=reg)
    P = clip_matrix(P)
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

def get_spatial_prior(D):
    '''
    OT method for calculating spatial matrix
    computes matrix Q given spatial matrix - gaussian kernel
    e^-D^2/2sigma
    '''
    sigma = np.median(D) # Why median here? - e^-median$2/2median^2 is not too small , gpt suggestion though, look into proper sources
    Q = np.exp(- (D ** 2) / (2 * sigma ** 2))
    return Q


def get_spatial_knn_expression_prior(p0, p1, sigma=None, eps=1e-8):
    '''
    Build a prior from pairwise Euclidean distances between sampled
    precomputed neighborhood-expression profiles.
    '''
    if isinstance(p0, torch.Tensor):
        p0_np = p0.detach().cpu().numpy()
    else:
        p0_np = np.asarray(p0, dtype=np.float32)

    if isinstance(p1, torch.Tensor):
        p1_np = p1.detach().cpu().numpy()
    else:
        p1_np = np.asarray(p1, dtype=np.float32)

    diff = p0_np[:, None, :] - p1_np[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    if sigma is None:
        sigma = float(np.median(dist))
    sigma = max(float(sigma), eps)

    Q = np.exp(-((dist ** 2) / (2.0 * sigma ** 2))) + eps
    row_sums = Q.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, eps)
    return Q / row_sums

def get_pseudotime_sigmoid_prior(y0, y1, eps=1e-8, alpha=10.0):
    '''
    Build a soft directional prior from source and target pseudotimes using a sigmoid
    over the pairwise pseudotime gap Delta_ij = t_tgt[j] - t_src[i].
    '''
    n0 = y0.shape[0]
    n1 = y1.shape[0]

    y0_t = y0.detach().to(dtype=torch.float32)
    y1_t = y1.detach().to(device=y0_t.device, dtype=torch.float32)

    diff = y1_t.unsqueeze(0) - y0_t.unsqueeze(1)
    forward_mask = diff >= 0

    Q = torch.full((n0, n1), eps, device=y0_t.device, dtype=torch.float32)
    Q_forward = eps + (1.0 - eps) * torch.sigmoid(alpha * diff)
    Q = torch.where(forward_mask, Q_forward, Q)

    # renormalize each row of Q to ensure it sums to 1
    Q_sum = Q.sum(dim=1, keepdim=True)
    Q_normalized = Q / Q_sum
    return Q_normalized.cpu().numpy()

    
def clip_matrix(M, eps=1e-8):
    '''
    Clip the cost matrix M to ensure numerical stability.
    This is important to avoid issues with log(0) when adjusting the cost matrix with the prior.
    '''
    M_clipped = np.maximum(M, eps)
    M_normalized = M_clipped / np.sum(M_clipped)
    return M_normalized


################################################## PSEUDOTIME PRIOR METHODS ##################################################


def get_pseudotime_prior_uniform(y0, y1, threshold=0.2):
    '''
    Based on the pseudotimes y0 and y1, compute the prior cost matrix Q for the OT plan.

    The initial idea is to create a prior matrix that has some initial minimum threshold
    which will be threshold / total_cells. Then, we calculate the uniform distribution
    over the other pseudotimes which are above, getting threshold / total_cells + (1 - (threshold / total_cells)) / (pseudotime_over_cells)

    We repeat this for every single cell in y0, and we get the final Q matrix which is of size (n0, n1) where n0 is the number of cells in y0 and n1 is the number of cells in y1.
    '''

    n0 = y0.shape[0]
    n1 = y1.shape[0]
    if n0 == 0 or n1 == 0:
        raise ValueError("y0 and y1 must each contain at least one pseudotime value.")

    y0_t = y0.detach().to(dtype=torch.float32)
    y1_t = y1.detach().to(device=y0_t.device, dtype=torch.float32)

    # Boolean mask of admissible pairs (cell in y1 occurs after cell in y0).
    pseudotime_greater_f = (y1_t.unsqueeze(0) > y0_t.unsqueeze(1)).to(dtype=torch.float32)

    # Row-wise counts via GEMM (BLAS-backed matmul).
    ones = torch.ones((n1, 1), device=y0_t.device, dtype=torch.float32)
    counts = pseudotime_greater_f @ ones
    counts_safe = torch.clamp(counts, min=1.0)

    threshold_value = float(threshold) / float(n1)
    extra_mass = (1.0 - threshold_value) / counts_safe  # shape: (n0, 1)

    Q = torch.full((n0, n1), threshold_value, device=y0_t.device, dtype=torch.float32)
    Q = Q + pseudotime_greater_f * extra_mass

    # renormalize each row of Q to ensure it sums to 1
    Q_sum = Q.sum(dim=1, keepdim=True)
    Q_normalized = Q / Q_sum
    return Q_normalized.cpu().numpy()

def get_pseudotime_prior_gaussian(y0, y1, sigma=0.1):
    '''
    Alternative pseudotime prior method where we use a Gaussian kernel based on the difference in pseudotime values.
    Q[i, j] = exp(- (y1[j] - (y0[i] + mu))^2 / (2 * sigma^2))
    where the mu is the expected time shift in pseudotime.
    
    We calculate mu as the shift between the time distributions, calculated using
    the Wasserstein distance between the two pseudotime distributions.
    '''
    y0_t = y0.detach().to(dtype=torch.float32)
    y1_t = y1.detach().to(device=y0_t.device, dtype=torch.float32)

    mu = sp.stats.wasserstein_distance(y0_t.cpu().numpy(), y1_t.cpu().numpy())
    sigma = sigma * mu # Scale sigma by the Wasserstein distance to adapt to the scale of the pseudotime distributions

    diff = y1_t.unsqueeze(0) - (y0_t.unsqueeze(1) + mu)
    Q = torch.exp(- (diff ** 2) / (2 * sigma ** 2))

    # add a small constant to ensure no zero entries in Q to avoid log(0) issues
    Q = Q + 1e-8

    # renormalize each row of Q to ensure it sums to 1
    Q_sum = Q.sum(dim=1, keepdim=True)
    Q_normalized = Q / Q_sum
    return Q_normalized.cpu().numpy()

def get_pseudotime_prior_gamma(y0, y1, shape=2.0, scale=1.0):
    '''
    Alternative pseudotime prior method where we use a Gamma distribution based on the difference in pseudotime values.
    Q[i, j] = gamma.pdf(y1[j] - y0[i], a=shape, scale=scale)
    where the shape and scale parameters can be tuned to control the skewness and spread of the distribution.
    '''
    y0_t = y0.detach().to(dtype=torch.float32)
    y1_t = y1.detach().to(device=y0_t.device, dtype=torch.float32)

    mu = sp.stats.wasserstein_distance(y0_t.cpu().numpy(), y1_t.cpu().numpy())

    diff = y1_t.unsqueeze(0) - (y0_t.unsqueeze(1) + mu)
    Q = torch.from_numpy(sp.stats.gamma.pdf(diff.cpu().numpy(), a=shape, scale=scale)).to(device=y0_t.device)

    # add a small constant to ensure no zero entries in Q to avoid log(0) issues
    Q = Q + 1e-8

    # renormalize each row of Q to ensure it sums to 1
    Q_sum = Q.sum(dim=1, keepdim=True)
    Q_normalized = Q / Q_sum
    return Q_normalized.cpu().numpy()
