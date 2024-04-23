import numpy as np 
import random
import helper
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

#Gets the initial points and handles some basic setup across all methods
def get_init_points(init_points, gt_data, labels, idx, X, rand_mark=False):
    if rand_mark:
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

#SSL but given score assigns to all points in one go
def ssl_assign_all(rho=0.01, init_points=10):
    dir_name = 'random_sample_tests/random_samp_32/random_sample_k=32_mu=0.0001_reg=0.08'
    X = torch.load(dir_name + '/coeff.pt').numpy()
    idx = torch.load(dir_name + '/train_index.pt').numpy()
    num_points = idx.shape[0]

    #Needed initialization information
    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    labels = [1, 2, 3, 4, 5, 6]
    (gt_data, mask) = helper.gt_and_mask(remap)
    (all_points, point_grouping, weights) = get_init_points(init_points, gt_data, labels, idx, X, rand_mark=True)
    dup_point = all_points.copy()
    weight_d_tracker = [np.zeros((num_points, init_points)) for i in range(0, 6)]
    
    s_norm = 100
    acc = 0
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
                        dist = np.linalg.norm(X[:,i] - weights[j][:,k])**2
                        weight_d_tracker[j][i, k] = dist
                        d1 += dist
                    else:
                        d1 += weight_d_tracker[j][i, k]
                    test_loc = point_grouping[j][k]
                    x_d = math.floor(test_loc/86)
                    y_d = test_loc - x_d*86
                    d2 += (np.abs(k1-x_d)**2 + np.abs(k2-y_d)**2)
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
    for i in range(0, vals.shape[0]):
        score = vals[i] + 1
        point_grouping[score - 1].append(int(idx_2[i]))
            
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
    plt.savefig('ssl_l2/rho=' + str(rho) + '_pts=' + str(init_points) + '_acc=' + str(round(acc,2)) + '.pdf', 
                bbox_inches='tight')
    plt.clf()

#SSL iteratively assign point after point
def ssl_solo_act(point_find, rho=0.05, init_points=25):
    random.seed(10)
    dir_name = 'random_sample_tests/random_samp_32/random_sample_k=32_mu=0.0001_reg=0.08'
    X = torch.load(dir_name + '/coeff.pt').numpy()
    idx = torch.load(dir_name + '/train_index.pt').numpy()
    num_points = idx.shape[0]

    #Needed initialization information
    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    labels = [1, 2, 3, 4, 5, 6]

    (gt_data, mask) = helper.gt_and_mask(remap)
    (all_points, point_grouping, weights) = get_init_points(10, gt_data, labels, idx, X, random=True)
    dup_point = all_points.copy()
    weight_d_tracker = [np.zeros((num_points, init_points)) for i in range(0, 6)]
    #On scale, spatial-norm dominates, so meant to scale things down
    
    s_norm = 100
    
    #SCORE UPDATE LOOP
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

    new_cmap = cmap_init_marker()
    plt.imshow(coloring, cmap=new_cmap)
    plt.title('Accuracy = ' + str(round(acc, 2)))
    plt.show()
    #plt.savefig('ssl_test2/rho=' + str(rho) + '_init_points=' + str(init_points) + '.png')
    plt.clf()

#Updates in fancy batch method
def ssl_batch_error_update(rho=0.01, init_points=1): 
    random.seed(10)
    dir_name = 'random_sample_tests/random_samp_32/random_sample_k=32_mu=0.0001_reg=0.08'
    X = torch.load(dir_name + '/coeff.pt').numpy()
    idx = torch.load(dir_name + '/train_index.pt').numpy()
    num_points = idx.shape[0]

    #Needed initialization information
    remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    labels = [1, 2, 3, 4, 5, 6]
    (gt_data, mask) = helper.gt_and_mask(remap)
    (all_points, point_grouping, weights) = get_init_points(10, gt_data, labels, idx, X, random=True)
    dup_point = all_points.copy()    
    weight_d_tracker = [np.zeros((num_points, init_points)) for i in range(0, 6)]
    #On scale, spatial-norm dominates, so meant to scale things down
    
    s_norm = 100
    
    #SCORE UPDATE LOOP
    #UPDATE 2, add only certain points
    acc = 0
    error = 0.2
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

    new_cmap = cmap_init_marker()
    plt.imshow(coloring, cmap=new_cmap)
    plt.title('Accuracy = ' + str(round(acc, 2)))
    plt.show()
    plt.clf()


if __name__ == '__main__':

    rhos = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
            1]
    pts = [1, 5, 10, 25, 50, 100]
    for r in rhos:
        for p in pts:
            ssl_assign_all(rho=r, init_points=p)