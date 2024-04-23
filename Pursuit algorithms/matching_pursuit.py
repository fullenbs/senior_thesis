import numpy as np 
import ot 
import random
import matplotlib.pyplot as plt 

#Matching Pursuit (MP) algorithm
#Goal: We have an overlearned dictionary that represents a multidimensional space.
# The goal is we have a signal in the space, and want to find coeffs that get
# best approximation of sample signal. 
# This is an analogue to solving sparsity problem, which solving exactly is NP-hard
# Algorithmic process: R is residual vector, find max inner product, and then 
# update coefficients and R doing that. 

def matching_pursuit(D, test_sample, tol=0.001):
    coeff = np.zeros(D.shape[0])
    maxiter = 0
    R = test_sample
    while np.linalg.norm(R) > tol:
        (max_inner, max_count) = (0, 0)
        for k in range(D.shape[0]):
            l = np.abs(np.inner(D[k,:], R))
            if l > max_inner: 
                (max_inner, max_count) = (l, k)
        coeff[max_count] = np.inner(R, D[max_count,:])
        R = R - coeff[max_count]*D[max_count,:]
        maxiter += 1 
    rec = np.zeros(D.shape[1])
    coeff = coeff/np.sum(coeff)
    print('LEARNED COEFFICIENTS', coeff)
    for i in range(D.shape[0]):
        rec += coeff[i]*D[i,:]
    plt.plot(rec, label='recreation')
    plt.plot(test_sample, label='original sample')
    plt.title('Learned vs original signal MP')
    plt.legend(loc='upper right')
    plt.savefig('Approximate signal MP')
    return (coeff, rec)

#Matching pursuit but uses orthogonal subspace to update coefficients
#Less computationally efficient than matching pursuit, but better results 
def ortho_matching_pursuit(D, test_sample, tol=0.001):
    test_sample = test_sample.T
    coeff = np.zeros(D.shape[0])
    maxiter = 0
    R = test_sample
    while np.linalg.norm(R) > tol:
        (max_inner, max_count) = (0, 0)
        for k in range(D.shape[0]):
            l = np.abs(np.inner(D[k,:], R))
            if l > max_inner: 
                (max_inner, max_count) = (l, k)
        coeff[max_count] = np.inner(R, D[max_count,:])
        R = R - coeff[max_count]*D[max_count,:]
        elems = np.nonzero(coeff)[0]
        A = D[elems,:].T 
        x = np.matmul(np.linalg.pinv(A), test_sample)
        R = test_sample - A @ x
        maxiter += 1 
    rec = np.zeros(D.shape[1])
    coeff = coeff/np.sum(coeff)
    for i in range(D.shape[0]):
        rec += coeff[i]*D[i,:]
    plt.plot(rec, label='recreation')
    plt.plot(test_sample, label='original sample')
    plt.legend(loc='upper right')
    plt.title('Learned vs original signal OMP')
    plt.savefig('Approximate_signal_OMP.pdf')
    return (coeff, rec)
    


if __name__ == '__main__':
    random.seed(10)
    D = np.zeros((10, 100))
    for i in range(D.shape[0]): 
        D[i,:] = ot.datasets.make_1D_gauss(D.shape[1], m=random.randint(20, 60), 
                                        s=np.random.randint(5, 10))
    plt.plot(D.T)
    plt.savefig('Dictionary_elements.pdf', bbox_inches='tight')
    plt.clf()
    test_sample = 0.5*D[2,:] + 0.3*D[5,:] + 0.2*D[1,:]
    (c, rec1) = matching_pursuit(D, test_sample)
    (c, rec2) = ortho_matching_pursuit(D, test_sample)
    plt.close()
    plt.plot(test_sample, label='Original signal')
    plt.plot(rec1, label='MP')
    plt.plot(rec2, label='Orthogonal MP')
    plt.legend(loc='upper right')
    plt.savefig('compare.pdf', bbox_inches='tight')