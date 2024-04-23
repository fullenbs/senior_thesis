import helper
import numpy as np 
#from sklearn.cluster import KMeans
import torch
from kmeans.kmeans import wassersteinKMeansInit_update
from wdl.bregman import OT
import argparse

def evil(size2):
    X = helper.data_loader('gt')

    idx = np.arange(size2)
    for i in range(len(idx)): 
        if X[i] == 0:
            idx[i] = -1

    idx = idx[idx != -1]
    print(idx, idx.shape)
    return idx


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64) 
    dev = torch.device('cpu')
    parser = argparse.ArgumentParser() #Reads in command line arguments
    parser.add_argument("--points", type=int)
    parser.add_argument('--count', type=int)


    args = parser.parse_args()
    points = args.points
    count = args.count
    dir_name = 'sampling_test_points=' + str(points) + '_count=' + str(count)
    helper.dir_check(dir_name)

    X = helper.data_loader('data')
    Y = helper.data_loader('gt')
    Y = Y.reshape(Y.shape[0]*Y.shape[1], -1)
    temp = np.nonzero(Y)[0]
    idx = evil(83*86)

    C = helper.Cost(4)
    X = X[temp]
    X = X/X.sum(axis=1)[:,None]

    train_data = torch.tensor(np.copy(X))    
    OTsolver = OT(C, method='bregman', reg=0.1, maxiter=100)
    train_data = torch.tensor(train_data.T)
    (D, idx_fin) = wassersteinKMeansInit_update(train_data, k=points, OTmethod=OTsolver, 
                                            idx_track=idx)

    torch.save(D, dir_name + '/train_data.pt')
    torch.save(idx_fin, dir_name + '/common_index.pt')
    
    helper.wdl_instance(k=32, train_size=points, dir_name=dir_name, reg=0.1, mu=0.001,
                 max_iters=400, n_restarts=2, n_clusters=6, training_data='', dummy=dir_name)
