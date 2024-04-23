import torch

# this might be a cruft thing to do?
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def barycenter(D, C, reg, weights, maxiter=100):
    """
    solves the problem min_p sum_i l_i W_reg(D_i, p_sharp) using iterative bregman projections.

    See https://arxiv.org/abs/1412.5154 for more details and derivation.

    Notation in code from https://arxiv.org/abs/1708.01955

    :param D: an (n x d) tensor of dictionary elements
    :param weights: the set of weights (should sum to 1) (a (d x n_barys) tensor where n_barys is the number of barycenters being computed)
    :param C: the ground cost matrix (should be a square matrix) (a tensor)
    :param maxiter: the max number of iterations
    :return p_sharp: the wasserstein barycenter
    """

    # TODO input validation

    # size of p_sharp
    n = D.shape[0]

    # num of dictionary elements
    d = D.shape[1]

    # ensure each weight vector is in the probability simplex
    weights = torch.divide(weights, torch.sum(weights, dim=0))

    # reshape weights if 1 dimensional
    if weights.shape == (d,):
        weights = torch.reshape(weights, (d, 1))

    # the number of barycenters being computed

    n_barys = weights.shape[1]

    # reshape to make batch matrix multiply work
    weights = torch.reshape(weights.T, (n_barys, d, 1))

    K = torch.exp(-C / reg)

    # TODO handle typing better
    dtype = K.dtype
    # initialize variables
    p = torch.zeros((n_barys, n, 1), dtype=dtype, device=dev)
    phi = torch.zeros((n_barys, D.shape[0], D.shape[1]), dtype=dtype, device=dev)
    b = torch.ones((n_barys, D.shape[0], D.shape[1]), dtype=dtype, device=dev)

    for i in range(maxiter):
        # update phi (phi = K^T a)
        # a = K b
        phi = torch.matmul(K, torch.div(D, torch.matmul(K, b) + 1e-12))

        # update p_sharp
        p = torch.exp(torch.bmm(torch.log(phi), weights))

        # update b
        b = torch.div(p, phi + 1e-12)

    p = torch.reshape(p.T, (n, n_barys))
    return p


def OT(A: torch.tensor, B: torch.tensor, C: torch.tensor, reg, maxiter=5, mode="plans", weights=None):
    """
    Compute the optimal transport distance between sets of histograms A and B elementwise.

    Space scales O(n^2) using this method

    :param A: the (n x d) tensor of source distributions
    :param B: the (n x d) tensor of target distributions
    :param C: the (n x n) cost matrix
    :param reg: the entropic regularization parameter
    :param maxiter: the max number of iterations
    :param mode: string to tell what the output should be:
        - "gradient" returns the gradient with respect to A
        - "plans" returns a (d x n x n) tensor where plans[i] is optimal plan i

    :return: a (n) dimensional tensor of that consist of the transport costs associated with each
    or if gradient return only the (n x d) tensor gradient (reg*log( u^(maxiter), where u is the left scaling vector)
    """

    # TODO input validation

    dtype = A.dtype

    # setup variables
    u = torch.ones(A.shape, dtype=dtype, device=dev)
    v = torch.ones(B.shape, dtype=dtype, device=dev)
    K = torch.exp(-C / reg)

    # sinkhorn loop
    for i in range(maxiter):
        # update u
        u = torch.divide(A, torch.matmul(K.T, v))

        # update v
        v = torch.divide(B, torch.matmul(K, u))

    # dirty hack to deal with division by 0 - doesn't work...
    u[u != u] = 0
    v[v != v] = 0

    if mode == "plans":
        return
    elif mode == "gradient":
        return -reg * torch.log(u)
    elif mode == "loss-sharp":
        # expensive version (C, diag(u) K diag(v))
        # TODO verify this code is correct
        return torch.dot(C.view(-1), (torch.mul(u, torch.mul(K, v.T))).view(-1))
    elif mode == "loss-smooth":
        # loss = (f,a) + (g,b)
        return (torch.dot(torch.log(u.reshape(-1)),
                          A.reshape(-1)) + torch.dot(torch.log(v.reshape(-1)), B.reshape(-1))) * reg
    elif mode == "weighted-loss-smooth":
        # loss = (f,a) + (g,b)
        u = torch.mul(weights, u)
        v = torch.mul(weights, v)
        return (torch.dot(-torch.log(u.reshape(-1)),
                          A.reshape(-1)) + torch.dot(-torch.log(v.reshape(-1)), B.reshape(-1))) * reg
    else:
        raise NotImplementedError(f"No mode {mode} implemented.")
