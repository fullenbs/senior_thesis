import math
from numpy.random import choice
import torch
from scipy.optimize import linear_sum_assignment


def gaussian1D(bin_points: torch.Tensor,
               mean: float,
               variance: float,
               ):
    """
    makes a 1D guassian supported on the bin_points and ensures that the entries sum to 1

    :param bin_points: the points at which the gaussian is supported
    :param mean: the mean of the gaussian
    :param variance: the variance of the distribution
    :return: an approximate guassian supported on bin_points
    """

    histogram = torch.exp(-torch.pow(bin_points - mean, 2) / (2 * variance))
    histogram /= histogram.sum()
    return histogram


def dirac1D(bin_points: torch.Tensor,
            x: float,
            epsilon: float = 0.0
            ):
    """
    makes a 1D dirac with mass at nearest point in bin_points to x
    
    :param bin_points: the grid of points
    :param x: the support of the dirac
    :param non_zero: whether or not to add a small epsilon to all bins to help numerical issues with 0s in distributions
    :return: the dirac
    """

    dirac = torch.zeros_like(bin_points)
    idx = torch.argmin(torch.abs(bin_points - x))
    dirac[idx] = 1

    # adjust by epsilon
    dirac += epsilon
    dirac /= dirac.sum()

    return dirac


def uniformDiracs(bin_points: torch.Tensor,
                  n_diracs: int,
                  padding: float = 0.0,
                  epsilon: float = 0.0
                  ):
    """
    generates uniformly spaced diracs

    :param bin_points: the support of the diracs
    :param n_diracs: how many diracs
    :param padding: fraction of the interval to pad the diracs away from the end on either side, must be < 0.5,
     - 0.25 is a good value for example
    :param epsilon: small epsilon to all bins in each dirac
    :return:
    """
    d = bin_points.shape[0]
    diracs = torch.zeros((d, n_diracs))

    intervals_size = bin_points[-1] - bin_points[0]

    pad = padding * intervals_size

    spacing = torch.linspace(bin_points[0] + pad, bin_points[-1] - pad, n_diracs)

    for i in range(n_diracs):
        diracs[:, i] = dirac1D(bin_points, spacing[i], epsilon=epsilon)

    return diracs


def syntheticGaussianDataset(bin_points: torch.Tensor,
                             n_gaussians: int,
                             n_points_per_gauassian: int,
                             interpolation: str = "uniform",
                             method: str = "exact"
                             ):
    """
    returns a (n_bins x n_gaussians * n_points_per_gaussian) tensor containing histograms formed as  barycenters of
    translated gaussians, ie for each gaussian we translate it so that it forms two "end point" distributions
    of which we use to to make barycenters of to be synthetic data points (if method=exact, then the datapoints are
    various translations of each data point

    :param bin_points: the support of the distributions
    :param n_gaussians: the number of gaussians to greate
    :param n_points_per_gauassian: how many barycenters to form of each gaussian
    :param interpolation: how to generate the barycenters between the end point distributions,
    uniform being uniformly space, and random being randomly positioned
    :param method: the way in which the barycenters should be computed
    - explicit is the literal translates between barycenters
    :return: the set of gaussians
    """

    d = bin_points.shape[0]
    n = n_points_per_gauassian * n_gaussians

    gaussians = torch.zeros((d, n))

    if method == "exact":
        for i in range(n_gaussians):
            mean = 0.45 * torch.rand(1)

            # pick variance so that at least 2 standard deviations lie completel within the interval [0,1]
            variance = (mean * torch.rand(1) / 2) ** 2

            if interpolation == "uniform":
                spacing = torch.linspace(bin_points[0], bin_points[-1], n_points_per_gauassian)

            for j in range(n_points_per_gauassian):
                # pick random translated mean so that new gaussian is a barycenter of the gaussians at
                # (mean, variance), (1-mean, variance)
                if interpolation == "uniform":
                    new_mean = (1 - 2 * mean) * spacing[j] + mean
                elif interpolation == "random":
                    new_mean = (1 - 2 * mean) * torch.rand(1) + mean

                # generate gaussian
                gaussians[:, i * n_points_per_gauassian + j] = gaussian1D(bin_points, new_mean, variance)

    else:
        raise NotImplementedError(f"No such method \"{method}\" currently implemented to compute barycenters \
        of gaussians")

    return gaussians


def gaussian2D(height: int, width: int, mean: torch.Tensor, cov: torch.Tensor, vec: bool = False):
    y = torch.tensor(range(height))
    x = torch.tensor(range(width))

    n = len(x) * len(y)

    ys, xs = torch.meshgrid(y, x, indexing='ij')

    points = torch.stack((ys.reshape(-1), xs.reshape(-1)), dim=1)

    # recenter points about mean
    x = (points - mean).type(torch.get_default_dtype())

    norm = torch.sqrt(2 * math.pi * torch.linalg.det(cov))

    invcov = torch.linalg.inv(cov)

    # get distribution values
    values = torch.exp(-x.view(n, 1, 2).bmm(invcov.mm(x.T).T.view(n, 2, 1)) / 2.).reshape(-1) / norm

    padding = 2e-4
    values += padding

    # normalize to sum to 1
    values /= values.sum()

    if vec:
        return values
    else:
        # reshape to grid
        return values.reshape((height, width))


def synthetic2DGaussianDataset(n_gaussians: int,
                               n_gaussian_sets: int,
                               n_samples_per_gaussian_set: int,
                               n_atoms_per_sample: int,
                               height: int,
                               width: int,
                               margin: float,
                               barySolver,
                               disjoint: bool = False):
    """
    makes a data set where the data are barycenters of some set of atoms

    :param n_gaussians: the number of gaussian atoms
    :param n_gaussian_sets: the number of sets of gaussians to use
    :param n_atoms_per_sample: the number of atoms in a set
    :param n_samples_per_gaussian_set: how many samples should be drawn from each set
    :param height: the grid height
    :param width: the grid width

    :param margin:
    :return:
    """

    # form 2D gaussians
    atoms = torch.zeros((n_gaussians, height, width))

    for i in range(n_gaussians):
        mean = sample2Dmean(height, width, margin)
        cov = sample2Dcov(mean, height, width)
        atoms[i] = gaussian2D(height, width, mean, cov)

    # turn gaussians atoms into a matrix dictionary (grids to vectors)
    vec_atoms = grid2vec(atoms)

    # form barycenters and track the coefficients
    X = torch.zeros((height * width, n_gaussian_sets * n_samples_per_gaussian_set))
    Lambda = torch.zeros((n_gaussians, n_gaussian_sets * n_samples_per_gaussian_set))

    if disjoint:
        feasible_idxes = list(range(n_gaussians))
    for i in range(n_gaussian_sets):
        # choose n_gaussians_per_set atoms
        idxs = choice(n_gaussians, n_atoms_per_sample, replace=False)

        if disjoint:
            # pick idxes
            idxs = [feasible_idxes[x] for x in idxs]

            # remove from feasible set for future consideration
            for idx in idxs:
                feasible_idxes.remove((idx))

            # update remaining num of gaussians to pick from
            n_gaussians -= n_atoms_per_sample

        for j in range(n_samples_per_gaussian_set):
            weights = simplexSample(n_atoms_per_sample)
            Lambda[idxs, i * n_samples_per_gaussian_set + j] = weights
            X[:, i * n_samples_per_gaussian_set + j] = barySolver(vec_atoms[:, idxs], weights).view(-1)

    # return atoms, dataset and coefficients
    return vec_atoms, X, Lambda


def grid2vec(grid_data: torch.Tensor):
    if len(grid_data.shape) > 2:
        data_dim = grid_data.shape[1] * grid_data.shape[2]
        vec = grid_data.reshape(-1, data_dim).T
    else:
        vec = grid_data.reshape(-1)

    return vec


def vec2grid(vec_data: torch.Tensor, height, width):
    if len(vec_data.shape) == 1 or vec_data.shape[1] == 1:
        return vec_data.view(height, width)
    else:
        raise NotImplementedError


def sample2Dcov(mean, height, width):
    maxy = 1.5 * max(height - mean[0], mean[0])
    maxx = 1.5 * max(width - mean[1], mean[1])

    vary = maxy * torch.rand(1)
    varx = maxx * torch.rand(1)

    # sample a small distance away from the square to avoid degenerate covariances
    rvalue = torch.rand(1) * 0.5

    covxy = torch.sqrt(vary * varx * (rvalue - 0.1 * rvalue))

    cov = torch.zeros((2, 2))
    cov[0, 0] = vary
    cov[1, 1] = varx

    # flip the sign randomly
    cov[0, 1] = cov[1, 0] = covxy * torch.sign(torch.bernoulli(torch.tensor([0.5])) - 0.5)

    return cov


def sample2Dmean(height, width, margin):
    """
    pick a mean within [margin*height , height - margin*height] and similar for width

    :param height: grid height
    :param width: grid width
    :param margin: margin in [0, 0.5)
    :return:
    """

    mean = torch.zeros((2))

    # sample a mean within the margined box
    mean[0] = height * margin + torch.rand(1) * (height - 2 * height * margin)
    mean[1] = width * margin + torch.rand(1) * (width - 2 * width * margin)

    return mean


def simplexSample(k: int, n_samples: int = 1):
    """
    return a vector of length k whose elements are nonnegative and sum to 1 - and in particularly the vector is sampled
    uniformly from this set via the bayesian bootstrap
    https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex

    :param k: the length of the vector to be sample from the simplex
    :return: a uniformly sampled vector from the probability simplex
    """

    samples = torch.zeros(k, n_samples)

    for i in range(n_samples):
        # sample k - 1 points
        weights = torch.rand((k - 1))

        # add 0 and 1 then sort
        new_weights = torch.zeros((k + 1))
        new_weights[0] = 0.0
        new_weights[1] = 1.0
        new_weights[2:] = weights

        new_weights, _ = torch.sort(new_weights)

        # differences between points to get the uniform sample
        samples[:, i] = new_weights[1:] - new_weights[:-1]

    return samples


def sampleBaryFromDict(D: torch.Tensor, n_samples: int, barySolver):
    X = torch.zeros((D.shape[0], n_samples))
    # Lambda = torch.zeros(D.shape[1], n_samples)
    Lambda = simplexSample(D.shape[1], n_samples)
    for i in range(n_samples):
        X[:, i] = barySolver(D, Lambda[:, i]).view(-1)

    return X, Lambda


def matchAtoms(D1, D2, weights, OTsolver):
    """
    aligns set of distributions D1 to D2 by finding the minimum assignment when comparing the distributions via OT

    :param D1: set of distributions
    :param D2: set of distributions to be aligned to
    :param OTsolver: function that takes two distributions as arguments
    :return:
    """
    assert (D1.shape == D2.shape)

    k = D1.shape[1]
    C = torch.zeros((k, k))
    for i in range(k):
        for j in range(k):
            C[i, j] = OTsolver(D1[:, i], D2[:, j])

    old_assignments, assignments = linear_sum_assignment(C)
    cost = C[old_assignments, assignments].sum()

    D1[:, assignments] = D1[:, old_assignments]

    if weights is not None:
        weights[assignments, :] = weights[old_assignments, :]

        return D1, weights, cost
    else:
        return D1, cost
