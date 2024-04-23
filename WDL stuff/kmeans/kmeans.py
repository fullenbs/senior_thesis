import math

from wdl.bregman import OT, barycenter
import torch
import numpy as np
from tqdm import tqdm
import warnings

def wassersteinKMeansInit_update(data: torch.Tensor,
                          k: int,
                          OTmethod,
                          idx_track: np.array, 
                          dev: torch.device = torch.device("cpu")):
    #torch.set_default_dtype(torch.float64) 

    d = data.shape[0]
    n = data.shape[1]
    C = torch.zeros((d, k), device=dev)

    idx_track = list(idx_track)
    idxes = list(range(n))
    idx_temp = np.zeros(k)
    # pick random
    a_1 = np.random.choice(n, 1)[0]
    print(a_1, idx_track[int(a_1)])
    idx_temp[0] = idx_track[int(a_1)]
    C[:, 0] = data[:, a_1].view(-1)

    print("#### KMEANS++ INIT LOOP ####")
    for i in tqdm(range(1, k)):
        p = np.zeros(n - i)

        # compute distances to centroids
        for j in range(n - i):
            idx = idxes[j]
            p[j] = torch.tensor(OTmethod(data[:, idx], C[:, :i])).min()

        # p might have negative entries from entropy term
        # hack fix that is not great
        if (p < 0).sum() > 0:
            p -= p.min()

        # normalize probability
        p /= p.sum()

        # if p contains nan use uniform weights
        if np.isnan(p).any():
            p = np.ones_like(p) / (n - i)
            warnings.warn(
                "K Means probabilites became nonzero - likely due to numerical instability in transport computation.",
                RuntimeWarning)

        # pick new centroid
        new_centroid_idx = np.random.choice(n - i, 1, p=p)[0]
        print(new_centroid_idx, idx_track[new_centroid_idx])
        idx_temp[i] = idx_track[new_centroid_idx]
        C[:, i] = data[:, new_centroid_idx].view(-1)
        del idxes[new_centroid_idx]
        del idx_track[new_centroid_idx]

    return (C, idx_temp)



def wassersteinKMeansInit(data: torch.Tensor,
                          k: int,
                          OTmethod,
                          dev: torch.device = torch.device("cpu")):

    d = data.shape[0]
    n = data.shape[1]
    C = torch.zeros((d, k), device=dev)

    idxes = list(range(n))

    C[:, 0] = data[:, np.random.choice(n, 1)[0]].view(-1)

    print("#### KMEANS++ INIT LOOP ####")
    for i in tqdm(range(1, k)):
        p = np.zeros(n - i)

        # compute distances to centroids
        for j in range(n - i):
            idx = idxes[j]
            p[j] = torch.tensor(OTmethod(data[:, idx], C[:, :i])).min()

        # p might have negative entries from entropy term
        # hack fix that is not great
        if (p < 0).sum() > 0:
            p -= p.min()

        # normalize probability
        p /= p.sum()

        # if p contains nan use uniform weights
        if np.isnan(p).any():
            p = np.ones_like(p) / (n - i)
            warnings.warn(
                "K Means probabilites became nonzero - likely due to numerical instability in transport computation.",
                RuntimeWarning)

        # pick new centroid
        new_centroid_idx = np.random.choice(n - i, 1, p=p)[0]
        C[:, i] = data[:, new_centroid_idx].view(-1)
        del idxes[new_centroid_idx]

    return C


def wassersteinKMeans(data: torch.Tensor,
                      k: int,
                      n_restarts: int,
                      ot_method: str,
                      bary_method: str,
                      reg: float,
                      height: int = None,
                      width: int = None,
                      max_iter: int = 10,
                      max_sink_iter: int = 7,
                      cost: torch.Tensor = None,
                      dev: torch.device = torch.device("cpu")):
    n = data.shape[1]
    #torch.set_default_dtype(torch.float64) 

    # select OT method
    OTsolver = OT(C=cost, reg=reg, maxiter=max_sink_iter, method=ot_method,
                  height=height, width=width, dev=dev)

    # select barycenter method
    barycenterSolver = barycenter(C=cost, reg=reg, maxiter=max_sink_iter, method=bary_method, dev=dev,
                                  height=height, width=width)
    best_opt = math.inf
    best_C = None
    best_assignments = None
    for restart in tqdm(range(n_restarts)):
        print(f"\n-------restart: {restart}--------")
        # initialize data points as centroids
        C = wassersteinKMeansInit(data, k, OTsolver)

        # main loop
        assignments = torch.zeros((n,), dtype=int, device=dev)
        old_assignments = torch.zeros_like(assignments, device=dev)

        for i in tqdm(range(max_iter + 1)):
            opt = 0.0
            for j in range(n):
                cost = np.asarray(OTsolver(data[:, j], C))
                assign = cost.argmin()
                assignments[j] = assign
                opt += cost[assign]

            # print(f"opt value at iteration {i}: {opt:.3f}")

            # check if there has not been any changes in assignments to terminate early
            if torch.abs(old_assignments - assignments).sum() == 0:
                break

            old_assignments = assignments.clone()

            # make sure new assignments done for last iteration if stopping before convergence
            if i == max_iter:
                break

            # form new centroids
            for j in range(k):
                cluster_idxes = assignments == j
                n_assigned = cluster_idxes.sum()
                # print(f"n assigned to {j} = {n_assigned}")
                w = torch.ones((n_assigned, 1)) / n_assigned
                C[:, j] = barycenterSolver(data[:, cluster_idxes], w).view(-1)

        if best_opt > opt:
            best_opt = opt
            best_C, best_assignments = C, assignments

    # return assignments and centroids
    return best_C, best_assignments