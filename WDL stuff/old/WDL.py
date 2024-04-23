import torch
from torch.autograd import Variable
import math

from old.bregman import barycenter, OT


class WDL():

    def __init__(self, num_elements):
        self.num_elements = num_elements
        pass

    def fit(self,
            X: torch.Tensor,
            C: torch.Tensor,
            reg: float,
            n_epochs: int,
            learning_rate: float = 0.1,
            maxsinkiter=5,
            fullAutoDiff=False,
            initialize="rand_bary",
            verbose=False,
            metric_steps=100,
            save_loss=False,
            ):
        """


        :param X: the tensor of data histograms to represent
        :param C: the cost matrix associated with moving mass from one histogram to another
        :param reg: the entropic regularization parameter
        :param n_epochs: how many iterations to do
        :param learning_rate: how much to scale the updates on the variables
        :param maxsinkiter: the max iterations for the barycenter and OT computation
        :return: Tensors D, weights such that the barycenters of D and weights approximate X
        """

        # initialize variables (in change of variable as described in section 3.1 of the WDL paper)
        if initialize == "rand":
            alpha = torch.rand((X.shape[0], self.num_elements), requires_grad=fullAutoDiff)

            beta = torch.rand((self.num_elements, X.shape[1]), requires_grad=fullAutoDiff)
        elif initialize == "rand_bary":
            weights = torch.rand((X.shape[1], self.num_elements))
            barys = barycenter(X, C, reg, weights, maxiter=maxsinkiter)

            alpha = Variable(inverseCOV(barys), requires_grad=fullAutoDiff)

            # still use random weights on the new barycenters
            # TODO fit each data point to the new barycenters
            beta = torch.rand((self.num_elements, X.shape[1]), requires_grad=fullAutoDiff)
        else:
            raise NotImplementedError(f"initialization method '{initialize}' is unknown")

        if fullAutoDiff:
            # setup optimizer
            optim = torch.optim.SGD([alpha, beta], lr=learning_rate)

        if not fullAutoDiff:
            for i in range(n_epochs):
                # change updated variables into histograms
                D = Variable(changeOfVariables(alpha), requires_grad=True)
                weights = Variable(changeOfVariables(beta), requires_grad=True)

                # compute barycenters
                # f(D) : R^(n x n_barys) -> (n x d)
                # f(w) : R^(n_barys x d) -> (n x d)
                p = barycenter(D, C, reg, weights, maxiter=maxsinkiter)

                # get gradients
                # TODO fix this to get correct jacobian
                p.backward(torch.ones(p.shape))

                gradD = D.grad
                gradw = weights.grad
                gradL = OT(p, X, C, reg, maxsinkiter, "gradient")

                # E(D): (n x n_barys) -> (1)
                # gradient of loss and barycenter with respect to D
                gradED = torch.matmul(gradD.T, gradL)

                # E(w) (2 x 10) -> 1
                # gradient of loss and barycenter with respect to weights
                gradEw = torch.matmul(gradw.T, gradL)

                # gradient through change of variables
                gradCOValpha = COVgrad(alpha)
                gradCOVbeta = COVgrad(beta)

                # apply gradients
                alpha -= learning_rate * torch.matmul(gradCOValpha, gradED)
                beta -= learning_rate * torch.matmul(gradCOVbeta, gradEw)
        else:
            if save_loss:
                loss_vec = torch.zeros(math.floor(n_epochs / metric_steps))
            for i in range(n_epochs):
                # zero out gradients
                optim.zero_grad()

                D = changeOfVariables(alpha)
                weights = changeOfVariables(beta)

                p = barycenter(D, C, reg, weights, maxiter=maxsinkiter)

                # compute loss
                loss = OT(p, X,
                          C,
                          reg,
                          maxiter=maxsinkiter,
                          mode='loss-smooth')

                # compute gradients
                loss.backward()

                # update variables
                optim.step()

                if i % metric_steps == 0:
                    if verbose:
                        print(f"epoch: {i}, loss: {loss.detach() / X.shape[1]}")
                    if save_loss:
                        loss_vec[int(i / metric_steps)] = loss.detach() / X.shape[1]

        # return dictionary and coefficients
        return changeOfVariables(alpha).detach(), changeOfVariables(beta).detach(), loss_vec


def changeOfVariables(X: torch.Tensor):
    """
    helper function to do the change of variables that makes each column of X sum to 1

    :param X: tensor to change variables of
    :param out: the tensor to return
    :return A: the transformed tensor
    """

    # intermediate exponential
    expX = torch.exp(X)

    # normalize columns
    out = torch.divide(expX, torch.sum(expX, dim=0))
    return out


def COVgrad(X: torch.Tensor):
    """
    The gradient of the change of variables

    :param X: The tensor to evaluate the derivative at
    :return : the derivative
    """

    COVX = changeOfVariables(X)
    return COVX - torch.pow(COVX, 2)


def inverseCOV(X: torch.tensor):
    """
    :param X: the (n x d) tensor of d historgrams to be inverted through the change of variables
    :return:
    """
    return torch.log(X)
