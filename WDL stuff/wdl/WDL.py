import math
import torch
from wdl.bregman import OT, barycenter
from old.bregman import OT as legacyOT
from old.bregman import barycenter as legacyBarycenter
import numpy as np
from kmeans.kmeans import wassersteinKMeansInit
import warnings
from wdl.bcm import getBCMweights
from utilities.simpleDistributions import simplexSample
from torch.utils.data import DataLoader, TensorDataset


class WDL():
    """
    Class WDL is the dictionary learning object. Fitting it with a data set learns a dictionary of specified size
    such that the datapoints can be represented as wasserstein barycenters
    """

    def __init__(self,
                 n_atoms: int,
                 dir: str,
                 dev: torch.device = torch.device("cpu"),
                 ) -> None:
        """

        :param n_atoms: the number of atoms to learn
        :param reg: the entropic regularization parameter
        """

        self.n_atoms = n_atoms

        # the dictionary
        self.D = None
        self.dir = dir
        # the weights
        self.weights = None

        self.fitted = False

        self.dev = dev

    def fit(self, X: torch.Tensor,
            C: torch.Tensor,
            dictionaryUpdate: str = "joint",
            dictOptimizer: type = None,
            dictOptKWargs: dict = {},
            weightUpdate: str = "joint",
            weight_iters: int = 1,
            weightOptimizer: type = None,
            weightOptKWargs: dict = {},
            init_method: str = 'rand',
            init_indices: torch.Tensor = None,
            bary_method: str = 'bregman',
            loss_method: str = 'bregman',
            weight_init: str = "rand",
            weight_update_iters: int = 1,
            support: torch.Tensor = None,
            sharp: bool = False,
            max_sinkhorn_iters: int = 10,
            reg: float = 1.0,
            locality: bool = False,
            mu: float = 0.0,
            max_iters: int = 1000,
            jointOptimizer: type = torch.optim.SGD,
            jointOptimKWargs: {} = None,
            tol: float = 1e-6,
            legacy: bool = False,
            verbose: bool = False,
            log: bool = False,
            log_iters: int = 100,
            n_restarts=1,
            height=None,
            width=None,
            batch_size: int = None,
            ):
        """

        :param X: (d x n) tensor where d is the dimension of the histograms, and n is the number of histograms
        :param C: (d x d) symmetric cost matrix between the support of the histograms
        :param bary_method: how to compute the barycenters
        :param loss_method: how to compute the loss between reconstruction and original histogram
        :param weight_update: specifies how the weights should be updated:
        - "joint", joint optimization with dictionary atoms
        - "bcm", solve the quadratic program to estimate the coefficients
        - "regression", use histogram regression
        :param weight_update_iters: how many iterations to use to update the weights (only used if regression is selected)
        :param max_sinkhorn_iters: maximum number of iterations to use for inner sinkhorn loops
        :param reg: entropic regularization parameter if applicable
        :param locality: whether or not to use locality constrain in regularization
        :param mu: the amount of which to regularize by locality
        :param max_iters: maximum outer learning iterations
        :param update_method: method to use to update variables in optimization
        :param tol: the how small the residual error should be before exiting
        :param verbose: whether or not to print messages about loss at each iteration
        :param log: whether or not to return a log dictionary of variables stored to
        :param log_iters: how often to log information about learning - (total_iters/log_iters) log entries
        :param legacy: overrides method options and uses old tensorized code to compute everything at one
        limited by memory, not stabilized, and prone to errors when 0s are present in distributions
        :param n_restarts: the number of times to do learning with random intitialization, picking the result that
        minimizes the objective

        :return: weights, logs
        """

        # validate inputs

        # initialize variables
        self.d = X.shape[0]  # dimension of the data
        self.n = X.shape[1]  # number of histograms

        # Move data to appropriate device
        self.X = X.to(self.dev)

        # assign parameters
        self.max_sinkhorn_iters = max_sinkhorn_iters
        self.reg = reg
        self.support = support
        self.joint = False
        self.mu = mu

        jointVars = []

        # setup dictionary and weights for initializing based on update methods
        if dictionaryUpdate == "joint":
            dictClass = lambda x: ExpCovHist(x, True)
        elif dictionaryUpdate == "expcov":
            dictClass = lambda x: ExpCovHist(x, False, optimizer=dictOptimizer, optKWargs=dictOptKWargs)
        else:
            raise ValueError(f"Dictionary update: {dictionaryUpdate} not implemented")

        if weightUpdate == "joint":
            weightClass = lambda x: ExpCovHist(x, True)
        elif weightUpdate == "expcov":
            weightClass = lambda x: ExpCovHist(x, False, optimizer=weightOptimizer, optKWargs=weightOptKWargs)
        elif weightUpdate == "bcm":
            weightClass = lambda x: BCMHist(x, self)
        else:
            raise ValueError(f"weight update: {weightUpdate} not implemented")

        # setup logging dictionary if logging
        if log:
            log_dict = {}

            # get the number of logged steps based on max iteration count
            # TODO: may need to truncate arrays if stopping with a certain tolerance
            n_logs = math.ceil(max_iters / log_iters)
            log_dict["loss"] = torch.zeros(n_logs)
            log_dict["weights"] = torch.zeros((n_logs, self.n_atoms, self.n))
            log_dict["weight_grads"] = torch.zeros((n_logs, self.n_atoms, self.n))
            log_dict["atoms"] = torch.zeros((n_logs, self.d, self.n_atoms))
            log_dict["atom_grads"] = torch.zeros((n_logs, self.d, self.n_atoms))
            log_dict["entropic regularization"] = reg
            log_dict["locality regularization"] = mu
            log_dict["distribution support size"] = self.d
            log_dict["initial distribution"] = X  # this will cause space issues in the future
            log_dict["log iterations"] = log_iters

        # restart setup
        self.best_loss = math.inf
        best_D = None
        best_weights = None

        for restart in range(n_restarts):

            print(f"\n---Restart {restart}---")
            if init_method == "rand":
                # dictionary
                self.D = dictClass(simplexSample(k=self.d, n_samples=self.n_atoms).to(self.dev))

                # changed variables for weights, shape (n_atoms x n)

            elif init_method == "kmeans++-init":
                # init atoms by choosing initial data points as in the kmeans++ centroid initialization
                OTsolver = OT(C=C, reg=reg, maxiter=max_sinkhorn_iters, method=loss_method,
                              height=height, width=width, sharp=sharp, dev=self.dev)
                self.D = dictClass(
                    wassersteinKMeansInit(X, k=self.n_atoms, OTmethod=OTsolver, dev=self.dev).to(self.dev))
            elif init_method == "rand-data":
                # initialize the variables with randomly chosen data point
                # changed variables for dictionary atoms, shape (d x n_atoms)
                self.D = dictClass(X[:, np.random.choice(self.n, self.n_atoms, replace=False)])
            elif init_method == "data-index":
                # initialize based on specifed indices of the input data
                if init_indices.view(-1).shape[0] != self.n_atoms:
                    raise ValueError("Number of indices to initialize with must be equal to the number of atoms.")

                if init_indices.max() >= self.n:
                    raise ValueError("Indices must be within the size of the input dataset.")

                # changed variables for dictionary atoms, shape (d x n_atoms)
                self.D = dictClass(X[:, init_indices])
            else:
                raise NotImplementedError(f"No variable initialization method matches \"{init_method}\".")

            # learn weights from dict using bcm or choose randomly from the simplex
            if weight_init == "bcm":
                self.weights = weightClass(
                    getBCMweights(D=self.D.get(), x=self.X, embeddings=support, reg=reg).to(self.dev))
            else:
                self.weights = weightClass(simplexSample(k=self.n_atoms, n_samples=self.n).to(self.dev))

            if self.D.joint:
                jointVars.append(self.D.variable)
            if self.weights.joint:
                jointVars.append(self.weights.variable)

            if len(jointVars) > 0:
                self.joint = True
                self.jointOptimizer = jointOptimizer(jointVars, **jointOptimKWargs)

            # use old code if running in legacy mode
            # TODO: integrate "tensorized" as an option with batch sizes
            if legacy:
                pass
                """ defunct with variable refactor
                for iter in range(max_iters):
                    # todo REMOVE, don't want to mention specific access to an optimizer in the
                    # zero out gradients
                    optim.zero_grad()

                    ## TODO Remove, should not need to access variables in a specific way like this
                    self.D = changeOfVariables(alpha)
                    weights = changeOfVariables(beta)

                    p = legacyBarycenter(D, C, reg, weights, maxiter=max_sinkhorn_iters)

                    # compute loss
                    loss = legacyOT(p, self.X,
                                    C,
                                    reg,
                                    maxiter=max_sinkhorn_iters,
                                    mode='loss-smooth') / n

                    # locality constraint
                    if locality:
                        locality_loss = 0.0
                        for i in range(n):
                            for j in range(self.n_atoms):
                                locality_loss += weights[j, i] * legacyOT(self.X[:, i],
                                                                          D[:, j],
                                                                          C,
                                                                          reg,
                                                                          maxiter=max_sinkhorn_iters,
                                                                          mode='loss-smooth')
                        loss += mu * locality_loss / n

                    # compute gradients
                    loss.backward()

                    # update variables
                    optim.step()

                    if iter % log_iters == 0:
                        if verbose:
                            print(f"Iteration: {iter}, loss: {loss.detach()}")
                        if log:
                            log_dict["loss"][int(iter / log_iters)] = loss.detach()

                if loss < best_loss:
                    best_loss = loss
                    best_α = alpha
                    best_β = beta
                """
            else:
                # select OT method
                self.OTsolver = OT(C=C, reg=reg, maxiter=max_sinkhorn_iters, method=loss_method,
                                   height=height, width=width, sharp=sharp, dev=self.dev)

                # select barycenter method
                self.barycenterSolver = barycenter(C=C, reg=reg, maxiter=max_sinkhorn_iters, method=bary_method,
                                                   dev=self.dev,
                                                   height=height, width=width)

                # residual_error = math.inf
                # TODO: allow exiting after some tolerance criteria is met

                # setup dataloader for training
                train_ds = TensorDataset(self.X.mT, torch.arange(self.n))
                if batch_size is None:
                    batch_size = self.n
                train_dl = DataLoader(train_ds, batch_size=batch_size)

                for curr_iter in range(max_iters):
                    # TODO: this should be involved with the variable updates
                    # optim.zero_grad()
                    total_loss = 0.0
                    # update weights
                    if not self.joint:
                        exitVal = False
                        for sub_iter in range(weight_iters):
                            loss = self.computeLoss()
                            total_loss += loss.detach()
                            if self.validateLoss(loss):
                                # dummy variable to do the double break
                                # (definitely better ways of doing this ¯\_(ツ)_/¯)
                                exitVal = True
                                break
                            loss.backward()
                            self.weights.update()

                        if exitVal:
                            break

                    # should be automatically done when updating the variables
                    # alpha_prev[:, :] = alpha
                    # beta_prev[:, :] = beta

                    # update dictionary or do joint optimization
                    for batch, batch_idxs in train_dl:
                        loss = self.computeLoss(batch.mT, batch_idxs)
                        total_loss += loss.detach()

                        loss.backward()
                        self.updateVariables()
                    if self.validateLoss(total_loss, curr_iter):
                        break

                    if curr_iter % log_iters == 0:
                        # normalize loss
                        total_loss /= len(train_dl)
                        if verbose:
                            print(f"Iteration: {curr_iter}, loss: {total_loss}")
                        if log:
                            # TODO: update logging to reflect new variables. Also consider looking into proper python logging
                            curr_idx = int(curr_iter / log_iters)
                            log_dict["loss"][curr_idx] = total_loss
                            log_dict["weights"][curr_idx] = self.weights.get().detach()
                            # log_dict["weight_grads"][curr_idx] = beta.grad
                            log_dict["atoms"][curr_idx] = self.D.get().detach()
                            # log_dict["atom_grads"][curr_idx] = alpha.grad
                if loss < self.best_loss:
                    self.best_loss = loss
                    # TODO: change to use generic variables
                    best_D = self.D.get().detach().clone()
                    best_weights = self.weights.get().detach().clone()

        # Cleanup output if bad results happened across all restarts
        # use previous weights from not bad output
        if self.best_loss == math.inf:
            best_D = self.D.getPrev()
            best_weights = self.weights.getPrev()

        # don't really use this so might be unnecessary
        self.fitted = True

        # TODO: setup with generic variables
        #  (maybe should be a method that can also be called when exiting due to numerical instability)
        # return weights and logged info
        self.D = best_D
        weights = best_weights

        if log:
            return weights, log_dict
        else:
            return weights

    def updateVariables(self):
        """
        updates variables in joint optmiizer or just the dictionary since the coding step handles the weights
        :return:
        """
        if self.joint:
            self.jointOptimizer.step()
            self.jointOptimizer.zero_grad()
        else:
            self.D.update()

    def computeLoss(self, batch: torch.Tensor, batch_idxs):
        """

        :param batch: the indices to use for the batch
        :return:
        """
        loss = 0.0

        # change of variables
        # D = changeOfVariables(alpha)
        X = batch
        batch_size = batch.shape[1]

        # weights = getWeights(D)

        D = self.D.get()
        weights = self.weights.get()[:, batch_idxs]

        # compute barycenters

        p = self.barycenterSolver(D, weights)
        # compute loss
        loss += self.OTsolver(X, p).sum()
        # Locality constraint
        if self.mu != 0.0:
            losses = torch.zeros(batch_size, D.shape[1], device=self.dev)
            for i in range(D.shape[1]):
                intermediate_loss = 0.0

                # compute weighted loss between data points and the atoms (locality constraint)
                losses[:, i] = self.OTsolver(D[:, i], X)

            # add each data's loss to the overall loss
            loss += self.mu * torch.matmul(losses.view(batch_size, 1, D.shape[1]),
                                           weights.mT.view(batch_size, D.shape[1], 1)).sum()

        # normalize by num data points
        loss /= batch_size

        return loss

    def validateLoss(self, loss, curr_iter):
        if torch.isnan(loss):
            warnings.warn(f"loss become nan on iteration {curr_iter}, exiting update loop", RuntimeWarning)
            if curr_iter == 0:
                warnings.warn("No updates were made and the initial variables were returned",
                              RuntimeWarning)

            # ensure that initial variables are used
            if self.best_loss == math.inf:
                loss = 0.0

            return True
        else:
            return False


def changeOfVariables(X: torch.Tensor):
    """
    helper function to do the change of variables that makes each column of X sum to 1

    :param X: tensor in R^(d x n)
    :return A: the tensor now of n d-dimensional histograms
    """

    # intermediate exponential
    expX = torch.exp(X)

    # normalize columns
    out = torch.divide(expX, torch.sum(expX, dim=0))
    return out


def inverseCOV(X: torch.tensor):
    """
    :param X: the (n x d) tensor of d historgrams to be inverted through the change of variables
    :return:
    """
    return torch.log(X)


def histRegression(D: torch.Tensor, p: torch.Tensor, baryMethod, otMethod,
                   optimizer: str = "adam", lr=0.25,
                   maxIter: int = 100, verbose: bool = False, tol=1e-6, rho: float = 0.0,
                   device: torch.device = torch.device("cpu")):
    """
    learns the coefficients to form p as a barycenter of the distributions in D

    :param D: histograms represented as a (d x k) tensor where d is the dimension of the histogram and k is the number
    of histograms
    :param p: the distribution to be represented by D
    :param optimizer: the optimizer to use in the iterative search
    :return w: the (k x 1) dimensional tensor that has the coefficients to form p approximately as a barycenter of D
    """

    if device is None:
        device = D.device

    p = p.view(-1, 1)

    assert D.shape[0] == p.shape[0]

    # initialize with change of variables as in WDL

    w_cov = torch.rand((D.shape[1],), requires_grad=True, device=device)

    w = changeOfVariables(w_cov)

    if optimizer == "adam":
        optim = torch.optim.Adam([w_cov], lr=lr)
    else:
        raise ValueError(f"Optimizer '{optimizer}' is not implemented.")

    for i in range(maxIter):
        w_prev = w.detach().clone()
        w = changeOfVariables(w_cov)
        optim.zero_grad()
        p_proj = baryMethod(D, w)

        loss = otMethod(p_proj, p)

        # compute locality
        if rho != 0.0:
            locality = w.dot(otMethod(p, D))
            loss += rho * locality

        loss.backward()
        optim.step()

        # do printing and tolerance check every 10 iterations
        if i % 10 == 0:
            if verbose:
                print(f"Loss at iter {i}: {loss.detach()}")

            if torch.linalg.norm(w - w_prev) < tol and i > 0:
                if verbose:
                    print(f"Converged within tolerance after {i} iterations")
                break

    return w.detach()


class GenericVariable:
    """
    describes a generic variable that can be optimized with update func
    """

    def __init__(self, variable: torch.Tensor, joint: bool, optimizer: type = None, optKWargs={}) -> None:
        """
        variable is the variable to be optimized as represented in its "external state" some implementations of this
        class may represent the variable in a different view (eg change of variables) in which case init should handle
        that change of variables to store it in the right state
        """
        self.variable = variable

        if joint and optimizer is not None:
            warnings.warn("Optimizer is ignored if joint is set to true.")
        elif not joint and optimizer is None:
            raise ValueError("Optimizer needs to be specified if not doing joint optimation")

        # ensure variable has a gradient
        if not self.variable.requires_grad:
            self.variable.requires_grad = True

        if type(optimizer) is type:
            self.optimizer = optimizer([self.variable], **optKWargs)

        self.joint = joint

        # setup get previous value (for errors)
        self.prev = self.variable.detach().clone()

    def get(self) -> torch.Tensor:
        return self.variable

    def getPrev(self) -> torch.Tensor:
        return self.prev

    def setPrev(self):
        self.prev = self.variable.detach().clone()

    def update(self) -> None:
        self.setPrev()
        if not self.joint:
            self.optimizer.step()
            self.optimizer.zero_grad()


# TODO: bake in an optimizer if not doing joint optimization
# open question on whether or not to allow user access to choice of optimizer
class ExpCovHist(GenericVariable):
    def __init__(self, variable: torch.Tensor, joint: bool, optimizer: type = None, optKWargs={}) -> None:
        # variable needs to be put into inverted change of variables
        # ensure variables are sufficently large with clip
        super().__init__(torch.clip(inverseCOV(variable).detach().clone(), min=-1e1), joint, optimizer,
                         optKWargs=optKWargs)

    def get(self) -> torch.Tensor:
        return changeOfVariables(self.variable)

    def getPrev(self) -> torch.Tensor:
        return changeOfVariables(self.prev)


# TODO: figure out how to get the dictionary information as well as support/other criteria
class BCMHist(GenericVariable):
    def __init__(self, variable: torch.Tensor, WDL):
        # set optimizer as dummy 1 to not throw an error
        # ok since update gets overridden
        super().__init__(variable, False, 1)
        self.WDL = WDL

    def update(self) -> None:
        super().setPrev()
        self.variable = getBCMweights(D=self.WDL.D.get(),
                                      x=self.WDL.X,
                                      embeddings=self.WDL.support,
                                      reg=self.WDL.reg,
                                      max_sinkhorn_iters=self.WDL.max_sinkhorn_iters)