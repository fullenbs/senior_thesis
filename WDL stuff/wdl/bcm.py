import cvxopt
import numpy as np
import ot
import torch


def getBCMweights(D: torch.Tensor, x: torch.Tensor, embeddings: torch.Tensor, reg: float,
                  max_sinkhorn_iters: int = 100, return_val: bool = False, device: torch.device = torch.device("cpu")):
    """

    :param D:
    :param x:
    :return:
    """

    # assert distributions have same dimension
    assert D.shape[0] == x.shape[0]

    if len(x.shape) == 1:
        x = x.view(-1, 1)

    n_data = x.shape[1]
    weights = torch.zeros((D.shape[1], n_data))

    if return_val:
        return_vals = torch.zeros(n_data)

    # get inner product matrix to optimize
    for i in range(n_data):
        A = inner_products(embeddings.mT, x[:, i], D, reg, max_sinkhorn_iters)
        if return_val:
            tmp, return_vals[i] = solve(A, return_val=return_val)
            weights[:, i] = torch.tensor(tmp)
        else:
            weights[:, i] = torch.tensor(solve(A, return_val=return_val))

    if return_val:
        return weights, return_vals
    else:
        return weights


def inner_products(data, a, D, entropy=0.1, max_sinkhorn_iters: int = 100):
    '''
    Code modified from Matt Werenski's BCM paper code

    nlp_utilities.inner_products
        Computes the inner product matrix A

    parameters
        data - list of point cloud supports. data[idx1] is a matrix the support of the base cloud

        bow_x - bag-of-words vector. bow_x[idx1] is the word counts in the base document

        base_idx - index of the source document

        ref_idxs - indices of the reference documents

        entropy - amount of entropy to use in finding the map

    returns
        the estimated inner product matrix A
    '''

    base_idx = a != 0.0

    # compute the entropic maps
    ent_maps = [compute_ent_map(data, a, D[:, i], entropy=entropy, max_sinkhorn_iters=max_sinkhorn_iters) for i in
                range(D.shape[1])]

    # subtract the support out for comp below
    supp = data[:, base_idx].T
    adj_maps = [em - supp for em in ent_maps]

    # get the base distribution
    dist = a[base_idx]

    p = D.shape[1]
    A = np.zeros((p, p))
    # fill in the matrix A
    for i, map_i in enumerate(adj_maps):
        for j, map_j in enumerate(adj_maps):
            A[i, j] = torch.dot((map_i * map_j).sum(1), dist)

    return A


def solve(inner_products, return_val=False):
    '''
    Code taken from Matt Werenski's BCM paper code

    opt_utilities.solve
        Actually solves the minimization procedure we've defined, given the
        evaluation of the inner products in the tangent space.

    parameters
        inner_products - p by p matrix of inner products in the tangent space.

        return_val - whether or not to return the value of the objective

    returns
        the optimal mixture weight, and value if return_val = True
    '''

    p = inner_products.shape[0]

    P = cvxopt.matrix(inner_products)
    q = cvxopt.matrix(np.zeros(p))
    G = cvxopt.matrix(-np.eye(p))
    h = cvxopt.matrix(np.zeros(p))
    A = cvxopt.matrix(np.ones((1, p)))
    b = cvxopt.matrix(np.ones((1, 1)))

    cvxopt.solvers.options['show_progress'] = False

    soln = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
    lam = np.squeeze(np.array(soln['x']))

    if return_val:
        return [lam, soln['primal objective']]
    return lam


def compute_ent_map(data, a, b, entropy=0.1, max_sinkhorn_iters: int = 100):
    '''
    code modified from Matt Werenski's BCM paper code

    nlp_utilities.compute_ent_map
        Computes the estiamte of the map between two point clouds

    parameters
        data - list of point cloud supports. data[idx1] is a matrixthe support of the base cloud

        bow_x - bag-of-words vector. bow_x[idx1] is the word counts in the base document

        idx1 - index of the source document

        idx2 - index of the target document

        entropy - amount of entropy to use in finding the map

    returns
        a matrix representation of the map estimate
    '''

    idx1 = a != 0.0
    idx2 = b != 0.0

    a = a[idx1]
    b = b[idx2]

    # pull out the point clouds
    x1 = data[:, idx1].T
    x2 = data[:, idx2].T

    # compute the optimal coupling matrix
    gamma = ot.bregman.empirical_sinkhorn(x1, x2, entropy, a=a, b=b, stopThr=1e-8, numIterMax=max_sinkhorn_iters)

    # turn it into a map
    ent_map = (gamma @ x2) / a.view(-1, 1)

    return ent_map
