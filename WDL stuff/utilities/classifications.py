import torch
from utilities.assignment import relabelAccuracy
from utilities.SpectralClustering import spectralClustering
from utilities.graphGen import KDSGraph
from kmeans.kmeans import wassersteinKMeans
from utilities.cost import atomCost, gridCost
from ot.lp import emd2


class Classifications():
    """
    Classifications class takes weights and data points learned from training a dictionry and performs various classifications on them
    """

    def __init__(self, weights: torch.Tensor, Dictionary: torch.Tensor, OTSolver, BarySolver, labels: torch.Tensor,
                 device: torch.device = torch.device("cpu")):
        """
        Setup the variables for classification use

        :param weights: (k x n) tensor where columns correspond to the weights represent each of the data points as a
        barycenter
        :param Dictionary: (d x k) tensor of distributions learned from a wasserstein dictionary learning process
        :param OTSolver: the function to compute optimal transport costs with
        :param BarySolver: the method to compute barycenter costs
        :param labels:
        """

        self.weights = weights
        self.Dictionary = Dictionary
        self.OTSolver = OTSolver
        self.BarySolver = BarySolver
        self.labels = labels
        self.methods = {"kds": ["unnormalized", "normalized-symmetric", "normalized-randomwalk"]}
        # ,"atom_entropic_ot_cost": ["unnormalized", "normalized-symmetric", "normalized-randomwalk"]}
        self.device = device
        self.accuracies = {}

    def Classify(self):
        """

        :return results: a dictionary with the accuracy results, labels, and label assignments
        """

        k = len(self.labels.unique())

        # loop over various methods of doing classification
        for method in self.methods:
            # loop over each variant per method (eg different ways of doing spectral clustering
            for variant in self.methods[method]:
                print(f"------Classifying with {method} using variant {variant}------")
                D, W = KDSGraph(self.weights)
                pred_labels = spectralClustering(D, W, k, method=variant, device=self.device)[:self.weights.shape[1]]
                accuracy = relabelAccuracy(pred_labels, self.labels, "min-error")
                self.accuracies[method + "-" + variant] = accuracy
                print(f"Accuracy for method {method} with k={k}: {accuracy}")

        # WDL K means on atoms
        method = "wasserstein-kmeans"
        C_grid = gridCost(28, 28).type(torch.float64)
        trueOT = lambda a, b: emd2(a, b, C_grid)
        # for true OT using emd2 need to renormalize data when as 64 bit due to problems with the library
        # since floating point errors may change the sum
        D = self.Dictionary.type(torch.float64)
        torch.divide(D, torch.sum(D, dim=0), out=D)
        C_atoms = atomCost(D, trueOT)

        centroids, pred_labels = wassersteinKMeans(data=self.weights,
                                                   k=k,
                                                   cost=C_atoms,
                                                   n_restarts=1,
                                                   ot_method="bregman",
                                                   bary_method="bregman",
                                                   reg=0.003,
                                                   max_iter=50,
                                                   max_sink_iter=7,
                                                   dev=self.device)

        accuracy = relabelAccuracy(pred_labels, self.labels, "min-error")
        self.accuracies[method] = accuracy
        print(f"Accuracy for method {method} with k={k}: {accuracy}")
