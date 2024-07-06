import numpy as np 
import random
import helper
import torch
import math
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy.spatial import distance
from sklearn.manifold import TSNE
from statistics import mode

#Gets the initial points and handles some basic setup across all methods
def get_init_points(init_points, gt_data, labels, idx, X, rand_mark=True):
    if not rand_mark:
        random.seed(10)
    all_points = set()
    point_grouping = [set() for i in range(0, 6)]

    for i in range(1, len(labels) + 1):
            while len(list(point_grouping[i - 1])) < init_points: 
                val = random.randint(0, gt_data.shape[0] - 1)
                if gt_data[val][0] == labels[i-1] and (val in idx):
                    point_grouping[i - 1].add(val)
                    all_points.add(val)
    #Gets initial associated weights
    point_grouping = [list(point_grouping[i]) for i in range(0, 6)]
    weights = [np.zeros((32,init_points)) for i in range(0, 6)]
    for i in range(0, 6): 
        weights[i][:,0] = X[:,np.where(idx == point_grouping[i][0])[0][0]]
    return (all_points, point_grouping, weights)

#Creates color map to get things the right way
def cmap_init_marker(rep_c=(1,0,0,1)):
    cmap_temp = cm.get_cmap('viridis', 8)
    cmap_temp2 = cm.get_cmap('viridis', 7)
    new_cmap = mcolors.ListedColormap(cmap_temp.colors) 
    fix = mcolors.ListedColormap(cmap_temp2.colors) 

    for i in range(0, cmap_temp2.colors.shape[0]):
        new_cmap.colors[i] = fix.colors[i]
    new_cmap.colors[0] = (1, 1, 1, 1)
    new_cmap.colors[7] = rep_c
    return new_cmap

def kmeans_init(data, k, idx_track):
    d = data.shape[0]
    n = data.shape[1]
    C = np.zeros((d, k))

    idx_track = list(idx_track)
    idxes = list(range(n))
    idx_temp = np.zeros(k)
    a_1 = np.random.choice(n, 1)[0]
    idx_temp[0] = idx_track[int(a_1)]
    C[:,0] = data[:,a_1]

    for i in range(1, k):
        p = np.zeros(n - i)

        # compute distances to centroids
        for j in range(n - i):
            idx = idxes[j]
            p[j] = 100000
            for l in range(0, k): 
                d = np.linalg.norm(C[:,l] - data[:,idx])
                if d < p[j]:
                    p[j] = d
        p /= p.sum()

        # pick new centroid
        new_centroid_idx = np.random.choice(n - i, 1, p=p)[0]
        idx_temp[i] = idx_track[int(new_centroid_idx)]
        C[:,i] = data[:, new_centroid_idx]
        del idxes[new_centroid_idx]
        del idx_track[new_centroid_idx]
    return (C, idx_temp)

#SSL but given score assigns to all points in one go
def ssl_assign_all(rho=0.01, init_points=10, preselect=False, point_grouping=None):
    dir_name = 'random_sample_tests/random_samp_32/random_sample_k=32_mu=0.0001_reg=0.08'
    X = torch.load(dir_name + '/coeff.pt').numpy()
    idx = torch.load(dir_name + '/train_index.pt').numpy()
    num_points = idx.shape[0]

    #Needed initialization information
    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    labels = [1, 2, 3, 4, 5, 6]
    (gt_data, mask) = helper.gt_and_mask(remap)
    if preselect == False: 
        (all_points, point_grouping, weights) = get_init_points(init_points, gt_data, labels, idx, X, rand_mark=True)
    else: 
        all_points = set()
        for label in point_grouping:
            for i in label: 
                all_points.add(i)
        #Gets initial associated weights
        size_init = [len(point_grouping[i]) for i in range(0, 6)]
        print(size_init)
        weights = [np.zeros((32,size_init[i])) for i in range(0, 6)]
        for i in range(0, 6): 
            weights[i][:,0] = X[:,np.where(idx == point_grouping[i][0])[0][0]]
    dup_point = all_points.copy()

    spat_tracker = [np.zeros((2,size_init[i])) for i in range(0, 6)]
    for i in range(len(spat_tracker)): 
        for j in range(spat_tracker[i].shape[1]):
            k = point_grouping[i][j]
            k1 = math.floor(k/86) 
            k2 = k - k1*86
            spat_tracker[i][:,j] = [k1, k2]
    
    s_norm = 100
    acc = 0
    m_score = np.zeros((7138, 6))
    for i in range(0, num_points):
        z = idx[i]
        if not (z in all_points):
            k1 = math.floor(z/86) 
            k2 = z - k1*86
            X_temp = np.array([[k1, k2]])

            #This calculates the score vector
            score = np.zeros(6)
            for j in range(0, 6): 
                Y = distance.cdist(X[:,i].reshape(1, 32), weights[j].T, metric='euclidean')
                d1 = np.sum(Y)
                d2 = np.sum(distance.cdist(X_temp, spat_tracker[j].T, metric='euclidean'))/s_norm
                score[j] = (d1 + rho*d2)/len(point_grouping[j])
            score /= np.sum(score)
            m_score[z,:] = score

    #Finds the minimum point
    non_zero = np.any(m_score != 0, axis=1)
    m_score = m_score[non_zero]
    idx_2 = np.zeros(7138)
    for i in range(0, idx.shape[0]):
        idx_2[idx[i]] = int(idx[i])
    idx_2 = idx_2[non_zero]

    vals = np.argmin(m_score, axis=1)
    for i in range(0, vals.shape[0]):
        point_grouping[vals[i]].append(int(idx_2[i]))
            
    for i in range(0, len(point_grouping)): 
        for j in point_grouping[i]: 
            if (not (j in dup_point)) and (i + 1) == gt_data[j][0]: 
                acc += 1
    acc = acc/X.shape[1]
    print(rho, init_points,' TOTAL ACCUARCY: ',  acc)
    coloring = np.zeros(7138)
    for i in range(0, 7138): 
        for j in range(0, len(point_grouping)): 
            if i in point_grouping[j]: 
                coloring[i] = j + 1
        if i in dup_point:
            coloring[i] = 7
    coloring = np.reshape(coloring, (83, 86))
    new_cmap = cmap_init_marker()

    plt.imshow(coloring, cmap=new_cmap)
    plt.title('acc: ' +  str(acc) + ' rho=' + str(rho))
    plt.savefig('active_label_rho=' + str(rho) + '.png')
    plt.clf()

#SSL iteratively assign point after point
def ssl_solo_act(point_find, rho=0.05, init_points=25, init_mode='random'):
    random.seed(20)
    dir_name = 'random_sample_tests/random_samp_32/random_sample_k=32_mu=0.0001_reg=0.08'
    X = torch.load(dir_name + '/coeff.pt').numpy()
    idx = torch.load(dir_name + '/train_index.pt').numpy()
    num_points = idx.shape[0]

    #Needed initialization information
    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    labels = [1, 2, 3, 4, 5, 6]
    (gt_data, mask) = helper.gt_and_mask(remap)

    all_points = set()
    point_grouping = [set() for i in range(0, 6)]

    if init_mode == 'kmeans++': 
        (C, pts) = kmeans_init(X, init_points, idx)
        for i in pts: 
            k = int(i)
            all_points.add(k)
            point_grouping[gt_data[k][0] - 1].add(k)
        point_grouping = [list(point_grouping[i]) for i in range(0, 6)]
    elif init_mode == 'random': 
        (all_points, point_grouping, weights) = get_init_points(init_points, gt_data, labels, idx, X)
    dup_point = all_points.copy()

    weight_d_tracker = [np.zeros((num_points, init_points)) for i in range(0, 6)]

    #On scale, spatial-norm dominates, so meant to scale things down
    s_norm = 100
    
    #UPDATE 2, add only certain points
    acc = 0
    for marking in range(0, point_find): 
        m_score = np.zeros((7138, 6))
        for i in range(0, num_points):
            z = idx[i]
            if not (z in all_points):
                #Grid location of image
                k1 = math.floor(z/86) 
                k2 = z - k1*86

                #This calculates the score vector
                score = np.zeros(6)
                for j in range(0, 6): 
                    (d1, d2) = (0, 0)
                    size = weights[j].shape[0]
                    for k in range(0, weights[j].shape[1]): 
                        if weight_d_tracker[j][i, k] == 0: 
                            dist = np.linalg.norm(X[:,i] - weights[j][:,k])
                            weight_d_tracker[j][i, k] = dist
                            d1 += dist
                        else:
                            d1 += weight_d_tracker[j][i, k]
                        test_loc = point_grouping[j][k]
                        x_d = math.floor(test_loc/86)
                        y_d = test_loc - x_d*86
                        d2 += (np.abs(k1-x_d) + np.abs(k2-y_d))
                        print(d1, d2, point_grouping[j][k])
                        exit()
                    
                    d2 /= s_norm
                    score[j] = (d1 + rho*d2)/size
                score /= np.sum(score)
                m_score[z,:] = score

        #Finds the minimum point
        non_zero = np.any(m_score != 0, axis=1)
        m_score = m_score[non_zero]
        idx_2 = np.zeros(7138)
        for i in range(0, idx.shape[0]):
            idx_2[idx[i]] = int(idx[i])
        idx_2 = idx_2[non_zero]

        #Pixel by pixel variant
        row_index, col_index = np.unravel_index(np.argmin(m_score), m_score.shape)
        m_row = m_score[row_index,:]
        m_lab_idx = np.argmin(m_row)
        label = m_lab_idx + 1
        idx_2 = int(idx_2[row_index])

        # #NOW THIS COMPLETES ONE ITERATION: 
        point_grouping[m_lab_idx].append(idx_2)
        weights[m_lab_idx] = np.append(weights[m_lab_idx], X[:,row_index].reshape(-1, 1), axis=1)
        weight_d_tracker[m_lab_idx] = np.append(weight_d_tracker[m_lab_idx], np.zeros((num_points, 1)), axis=1)

        all_points.add(idx_2)
        if label == gt_data[idx_2][0]:
            acc +=1
        print('Prediction:', label, 'True label:', gt_data[idx_2][0], marking)
            
    acc = acc/point_find
    print(rho, init_points,' TOTAL ACCUARCY: ',  acc)
    coloring = np.zeros(7138)
    for i in range(0, 7138): 
        for j in range(0, len(point_grouping)): 
            if i in point_grouping[j]: 
                coloring[i] = j + 1
        if i in dup_point:
            coloring[i] = 7

    coloring = np.reshape(coloring, (83, 86))
    new_cmap = cmap_init_marker()
    plt.imshow(coloring, cmap=new_cmap)
    plt.savefig('comparison_old=' + str(rho) + '_init_points=' + str(init_points) + '_acc=' + 
                str(round(acc, 2)) +  '.pdf',
                bbox_inches='tight')
    plt.clf()

#Updates in fancy batch method
def ssl_batch_error_update(rho=0.01, init_points=1): 
    random.seed(10)
    dir_name = 'random_sample_tests/random_samp_32/random_sample_k=32_mu=0.0001_reg=0.08'
    X = torch.load(dir_name + '/coeff.pt').numpy()
    idx = torch.load(dir_name + '/train_index.pt').numpy()
    num_points = idx.shape[0]

    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    labels = [1, 2, 3, 4, 5, 6]
    (gt_data, mask) = helper.gt_and_mask(remap)
    (all_points, point_grouping, weights) = get_init_points(init_points, gt_data, labels, idx, X, rand_mark=True)
    dup_point = all_points.copy()
    weight_d_tracker = [np.zeros((num_points, init_points)) for i in range(0, 6)]
    
    s_norm = 100
    
    #SCORE UPDATE LOOP
    #UPDATE 2, add only certain points
    acc = 0
    error = 0.17
    tot_lab = 0
    while (error < 0.21) and (tot_lab != (X.shape[1] - len(dup_point))):
        m_score = np.zeros((7138, 6))
        for i in range(0, num_points):
            z = idx[i]
            if not (z in all_points):
                #Grid location of image
                k1 = math.floor(z/86) 
                k2 = z - k1*86

                #This calculates the score vector
                score = np.zeros(6)
                for j in range(0, 6): 
                    (d1, d2) = (0, 0)
                    size = weights[j].shape[0]
                    for k in range(0, weights[j].shape[1]): 
                        if weight_d_tracker[j][i, k] == 0: 
                            dist = np.linalg.norm(X[:,i] - weights[j][:,k])
                            weight_d_tracker[j][i, k] = dist
                            d1 += dist
                        else:
                            d1 += weight_d_tracker[j][i, k]
                        test_loc = point_grouping[j][k]
                        x_d = math.floor(test_loc/86)
                        y_d = test_loc - x_d*86
                        d2 += (np.abs(k1-x_d) + np.abs(k2-y_d))
                    
                    d2 /= s_norm
                    score[j] = (d1 + rho*d2)/size
                score /= np.sum(score)
                m_score[z,:] = score

        #Finds the minimum point
        non_zero = np.any(m_score != 0, axis=1)
        m_score = m_score[non_zero]
        idx_2 = np.zeros(7138)
        for i in range(0, idx.shape[0]):
            idx_2[idx[i]] = int(idx[i])
        idx_2 = idx_2[non_zero]

        vals = np.argmin(m_score, axis=1)
        p_lab = 0
        for i in range(0, vals.shape[0]):
            score = vals[i] + 1
            if m_score[i, vals[i]] < error:
                point_grouping[score - 1].append(int(idx_2[i]))
                all_points.add(int(idx_2[i]))
                weights[vals[i]] = np.append(weights[vals[i]], X[:,i].reshape(-1, 1), axis=1)
                weight_d_tracker[vals[i]] = np.append(weight_d_tracker[vals[i]], np.zeros((num_points, 1)), axis=1)
                p_lab += 1

        print('Points labeed: ', p_lab, error)
        error += 0.01
        tot_lab += p_lab
            
    for i in range(0, len(point_grouping)): 
        for j in point_grouping[i]: 
            if (not (j in dup_point)) and (i + 1) == gt_data[j][0]: 
                acc += 1
    acc = acc/X.shape[1]
    print(rho, init_points,' TOTAL ACCUARCY: ',  acc)
    coloring = np.zeros(7138)
    for i in range(0, 7138): 
        for j in range(0, len(point_grouping)): 
            if i in point_grouping[j]: 
                coloring[i] = j + 1
        if i in dup_point:
            coloring[i] = 7

    coloring = np.reshape(coloring, (83, 86))
    new_cmap = cmap_init_marker()
    plt.imshow(coloring, cmap=new_cmap)
    plt.savefig('ssl_batch_test/rho=' + str(rho) + '_init_points=' + str(init_points) + '.pdf',
                bbox_inches='tight')
    plt.clf()

#SSL iteratively assign point after point
def ssl_solo_fix_act(point_find, rho=0.05, init_points=25, init_mode='random'):
    random.seed(20)
    dir_name = 'random_sample_tests/random_samp_32/random_sample_k=32_mu=0.0001_reg=0.08'
    X = torch.load(dir_name + '/coeff.pt').numpy()
    idx = torch.load(dir_name + '/train_index.pt').numpy()
    num_points = idx.shape[0]

    #Needed initialization information
    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    labels = [1, 2, 3, 4, 5, 6]
    (gt_data, mask) = helper.gt_and_mask(remap)

    all_points = set()
    point_grouping = [set() for i in range(0, 6)]

    if init_mode == 'kmeans++': 
        (C, pts) = kmeans_init(X, init_points, idx)
        for i in pts: 
            k = int(i)
            all_points.add(k)
            point_grouping[gt_data[k][0] - 1].add(k)
        point_grouping = [list(point_grouping[i]) for i in range(0, 6)]
    elif init_mode == 'random': 
        (all_points, point_grouping, weights) = get_init_points(init_points, gt_data, labels, idx, X)
    dup_point = all_points.copy()

    spat_tracker = [np.zeros((2,init_points)) for i in range(0, 6)]
    for i in range(len(spat_tracker)): 
        for j in range(spat_tracker[i].shape[1]):
            k = point_grouping[i][j]
            k1 = math.floor(k/86) 
            k2 = k - k1*86
            spat_tracker[i][:,j] = [k1, k2]
    #On scale, spatial-norm dominates, so meant to scale things down
    s_norm = 100
    
    #UPDATE 2, add only certain points
    start = time.time()
    acc = 0
    for marking in range(0, point_find): 
        m_score = np.zeros((7138, 6))
        for i in range(0, num_points):
            z = idx[i]
            if not (z in all_points):
                #Grid location of image
                k1 = math.floor(z/86) 
                k2 = z - k1*86
                X_temp = np.array([[k1, k2]])

                #This calculates the score vector
                score = np.zeros(6)
                for j in range(0, 6): 
                    Y = distance.cdist(X[:,i].reshape(1, 32), weights[j].T, metric='euclidean')
                    d1 = np.sum(Y)
                    d2 = np.sum(distance.cdist(X_temp, spat_tracker[j].T, metric='euclidean'))/s_norm
                    score[j] = (d1 + rho*d2)/len(point_grouping[j])

                score /= np.sum(score)
                m_score[z,:] = score

        #Finds the minimum point
        non_zero = np.any(m_score != 0, axis=1)
        m_score = m_score[non_zero]
        idx_2 = np.zeros(7138)
        for i in range(0, idx.shape[0]):
            idx_2[idx[i]] = idx[i]
        idx_2 = idx_2[non_zero]

        #Pixel by pixel variant
        row_index, col_index = np.unravel_index(np.argmin(m_score), m_score.shape)
        m_row = m_score[row_index,:]
        finx = math.floor(idx_2[row_index]/86)
        finy = idx_2[row_index] - finx*86
        temp_arr = np.array([finx, finy])
        m_lab_idx = np.argmin(m_row)
        label = m_lab_idx + 1
        idx_2 = int(idx_2[row_index])

        #NOW THIS COMPLETES ONE ITERATION: 
        point_grouping[m_lab_idx].append(idx_2)
        weights[m_lab_idx] = np.append(weights[m_lab_idx], X[:,row_index].reshape(-1, 1), axis=1)
        spat_tracker[m_lab_idx] = np.append(spat_tracker[m_lab_idx], temp_arr.reshape(-1, 1), axis=1)

        all_points.add(idx_2)
        if label == gt_data[idx_2][0]:
            acc +=1
        print('Prediction:', label, 'True label:', gt_data[idx_2][0], marking)
            
    acc = acc/point_find
    print(rho, init_points,' TOTAL ACCUARCY: ',  acc)
    coloring = np.zeros(7138)
    for i in range(0, 7138): 
        for j in range(0, len(point_grouping)): 
            if i in point_grouping[j]: 
                coloring[i] = j + 1
        if i in dup_point:
            coloring[i] = 7

    coloring = np.reshape(coloring, (83, 86))
    new_cmap = cmap_init_marker()
    plt.imshow(coloring, cmap=new_cmap)
    plt.savefig('fixed_rho=' + str(rho) + '_init_points=' + str(init_points) + '_acc=' + 
                str(round(acc, 2)) +  '.pdf',
                bbox_inches='tight')
    plt.clf()
    print('FINISH: ', time.time() - start)

def active_labeling(rho=0.05, init_points=25, label_pts=100): 
    random.seed(20)
    dir_name = 'random_sample_tests/random_samp_32/random_sample_k=32_mu=0.0001_reg=0.08'
    X = torch.load(dir_name + '/coeff.pt').numpy()
    idx = torch.load(dir_name + '/train_index.pt').numpy()
    num_points = idx.shape[0]

    #Needed initialization information
    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    labels = [1, 2, 3, 4, 5, 6]
    (gt_data, mask) = helper.gt_and_mask(remap)

    all_points = set()
    point_grouping = [set() for i in range(0, 6)]
    (all_points, point_grouping, weights) = get_init_points(init_points, gt_data, labels, idx, X)

    spat_tracker = [np.zeros((2,init_points)) for i in range(0, 6)]
    for i in range(len(spat_tracker)): 
        for j in range(spat_tracker[i].shape[1]):
            k = point_grouping[i][j]
            k1 = math.floor(k/86) 
            k2 = k - k1*86
            spat_tracker[i][:,j] = [k1, k2]
    #On scale, spatial-norm dominates, so meant to scale things down
    s_norm = 100
    m_score = np.zeros((7138, 6))

    for i in range(0, num_points):
        z = idx[i]
        if not (z in all_points):
            #Grid location of image
            k1 = math.floor(z/86) 
            k2 = z - k1*86
            X_temp = np.array([[k1, k2]])

            #This calculates the score vector
            score = np.zeros(6)
            for j in range(0, 6): 
                Y = distance.cdist(X[:,i].reshape(1, 32), weights[j].T, metric='euclidean')
                d1 = np.sum(Y)
                d2 = np.sum(distance.cdist(X_temp, spat_tracker[j].T, metric='euclidean'))/s_norm
                score[j] = (d1 + rho*d2)/len(point_grouping[j])

            score /= np.sum(score)
            m_score[z,:] = score

    non_zero = np.any(m_score != 0, axis=1)
    m_score = m_score[non_zero]
    idx_2 = np.zeros(7138)
    for i in range(0, idx.shape[0]):
        idx_2[idx[i]] = idx[i]
    idx_2 = idx_2[non_zero]

    m_vals = np.zeros(idx_2.shape[0])
    for i in range(0, m_vals.shape[0]):
        m_vals[i] = np.min(m_score[i,:])
        
    m_idx = np.argsort(m_vals)[::-1]
    m_vals = np.sort(m_vals)[::-1]
    idx_2 = idx_2[m_idx]
    labels = np.zeros(label_pts)

    for i in range(0, label_pts): 
        labels[i] = gt_data[int(idx_2[i])] - 1
        point_grouping[int(labels[i])].append(int(idx_2[i]))
    return point_grouping


#Mask that is 1 if data is in set, 0 otherwise
def labeled_mask(idx): 
    data = np.zeros(83*86)
    for i in range(0, idx.shape[0]): 
        data[int(idx[i])] = 1
    data = np.reshape(data, (83, 86))
    return data

#TODO: Build up spectral clustering on embedding
def over_cluster(embed, n_clusters, n_points, idx, gt_data, S=0): 
    if S == 0: 
        km = KMeans(init='k-means++', n_init='auto', n_clusters=n_clusters)
    #Core assignment algorithm
    else: 
        km = SpectralClustering(n_clusters=n_clusters)
    km.fit(embed)
    labels = km.labels_
    centroid = km.cluster_centers_
    vals = distance.cdist(centroid, embed)
    label_remap = np.zeros(n_clusters)
    for row in range(vals.shape[0]):
        labeled = 0
        dist_track = vals[row,:]
        args = np.argsort(dist_track)
        j = 0         #NOTE: Want first n labeled points? 
        while labeled < n_points: 
            if gt_data[idx[args[j]]] != 0: 
                labeled += 1
            j += 1
        args_label = gt_data[idx[args[0:j]]] 
        args_label = args_label[args_label != 0] 
        gt_label = np.zeros(n_points)
        for k in range(0, args_label.shape[0]): 
            gt_label[k] = args_label[k]
        label_remap[row] = mode(gt_label) 
    for i in range(len(labels)): 
        labels[i] = label_remap[labels[i]]
    return labels


def modified_mass_cluster(num_clusters=20, n_points=5, tsne=True, SC=False):
    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    dir_name = 'true_random_1'
    #dir_name = 'random_sample_tests/random_samp_32/random_sample_k=32_mu=0.0001_reg=0.08'

    (gt_data, mask) = helper.gt_and_mask(remap)

    X = torch.load(dir_name + '/coeff.pt').numpy()
    idx = torch.load(dir_name + '/train_index.pt').numpy()
    save_name = 'testing'
    #training_mask = labeled_mask(idx)

    data_size = idx.shape[0]
    data = helper.data_loader('data')
    mass = np.zeros(data_size)

    for i in range(data_size): 
        mass[i] = np.sum(data[idx[i],:])
    mass = np.reshape(mass, (1, data_size))
    X = np.append(X.T, mass.reshape(-1, 1), axis=1)

    cmap = cm.get_cmap('viridis', 7)
    new_cmap = mcolors.ListedColormap(cmap.colors)
    new_cmap.colors[0] = (1, 0, 0, 1)
    for i in range(X.shape[0]):
        X[i,:] /= np.sum(X[i,:])
    if tsne: 
        embed = TSNE(n_components=2, learning_rate='auto', init='random', 
                    perplexity=25).fit_transform(X)
        plt.scatter(embed[:,0], embed[:,1], c=gt_data[idx], cmap=new_cmap)
        #plt.title('Embedding of data (Red is unlabeled)')
        plt.savefig(save_name + '/2_embedding_n_clusters=' + str(num_clusters) + '_n_points=' + 
                    str(n_points) + '.pdf', bbox_inches='tight')
        plt.clf()    
    else: 
        embed = X

    if not SC: 
        labels = over_cluster(embed, num_clusters, n_points, idx, gt_data)
    else: 
        labels = over_cluster(embed, num_clusters, n_points, idx, gt_data, S=embed)
    #Accuracy of labeling, must exclude unlabeled data from these results
    acc = 0
    zero_count = 0
    train_plot = np.zeros(83*86)
    for i in range(data_size):
        t = idx[i]
        j = labels[i]
        train_plot[t] = j
        if gt_data[t] == 0: 
            zero_count += 1
        elif gt_data[t] == j:
            acc += 1
    acc = acc/(data_size - zero_count)

    print('Clusters: ', num_clusters, 'Points: ', n_points, 'Accuracy: ', acc)
    if tsne: 
        plt.scatter(embed[:,0], embed[:,1], c=labels, cmap='viridis')
        #plt.show()
        #plt.title('Post algorithm labels accuracy=' + str(acc))
        plt.savefig(save_name + '/2_post_algo_n_clusters=' + str(num_clusters) + '_n_points='
                 + str(n_points) + '_acc=' + str(round(acc, 2)) + '.pdf', bbox_inches='tight')
        plt.clf()

    train_plot = np.reshape(train_plot, (83, 86))
    relabel = helper.spatial_NN(train_plot, 10, mask, run_mode='relabel')
    #mask = np.ceil((mask + training_mask)/2)
    relabel2 = helper.spatial_NN(relabel, 10, mask, run_mode='NN')

    X = relabel2.reshape(-1)
    paint_acc = 0
    count = 0
    for i in range(83*86):
        if X[i] != 0 and gt_data[i] != 0:
            count += 1
            if X[i] == gt_data[i]: 
                paint_acc +=1 
    paint_acc = paint_acc/count
    paint_display = str(round(paint_acc, 3))
    print('Post inpainting and relabeling accuracy: ', paint_display)
    plt.imshow(relabel2, cmap=helper.virid_modify())
    #plt.show()
    #plt.title('Post inpainting/relabeling=' + paint_display)
    plt.savefig(save_name + '/2_post_inpaint_n_clusters=' + str(num_clusters) + '_n_points='
                + str(n_points) + '_acc=' + paint_display + '.pdf', bbox_inches='tight')
    plt.clf()
    return paint_acc

def mass_cluster(): 
    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    dir_name = 'true_random_1'
    #dir_name = 'random_sample_tests/random_samp_32/random_sample_k=32_mu=0.0001_reg=0.08'

    (gt_data, mask) = helper.gt_and_mask(remap)

    X = torch.load(dir_name + '/coeff.pt').numpy()
    idx = torch.load(dir_name + '/train_index.pt').numpy()
    #save_name = 'whisper_pics'
    #training_mask = labeled_mask(idx)

    data_size = idx.shape[0]
    data = helper.data_loader('data')
    mass = np.zeros(data_size)

    for i in range(data_size): 
        mass[i] = np.sum(data[idx[i],:])
    mass = np.reshape(mass, (1, data_size))
    embed = TSNE(n_components=2, learning_rate='auto', init='random', 
        perplexity=25).fit_transform(mass.T)
    labels = over_cluster(embed, 8, 1, idx, gt_data)
    acc = 0
    zero_count = 0
    train_plot = np.zeros(83*86)
    for i in range(data_size):
        t = idx[i]
        j = labels[i]
        train_plot[t] = j
        if gt_data[t] == 0: 
            zero_count += 1
        elif gt_data[t] == j:
            acc += 1
    acc = acc/(data_size - zero_count)

    print('Clusters: ', 8, 'Points: ', 1, 'Accuracy: ', acc)
    plt.scatter(embed[:,0], embed[:,1], c=labels, cmap='viridis')
    plt.show()
    #plt.title('Post algorithm labels accuracy=' + str(acc))
    # plt.savefig(save_name + '/post_algo_n_clusters=' + str(num_clusters) + '_n_points='
    #          + str(n_points) + '_acc=' + str(round(acc, 2)) + '.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':  
    #n_clusters = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    n_clusters = [8]
    n_points = [1]
    i = 0
    n = 1
    vals = np.zeros((len(n_clusters), n))
    mean_store = np.zeros(len(n_clusters))

    for c in n_clusters: 
        for p in n_points: 
            k = 0
            for r in range(0, n): 
                k = modified_mass_cluster(num_clusters=c, n_points=p, tsne=True)
                vals[i, r] = k
                mean_store[i] += k
            mean_store[i] /= n
        i += 1
    # np.save('vals.npy', vals)
    # X = np.load('vals.npy')
    # Y = np.std(X, axis=1)
    # plt.scatter(n_clusters, Y)
    # plt.show()
    print(np.mean(vals, axis=1), np.std(vals, axis=1))

