import helper
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_atoms", type=int)
    parser.add_argument("--geom", type=float)
    parser.add_argument('--reg', type=float)
    parser.add_argument('--count', type=int)
    args = parser.parse_args()
    k = args.n_atoms
    mu = args.geom
    reg = args.reg
    iter = args.count

    #for i in range(0, iter):
    name = 'robust_test_k=' + str(iter)
    helper.wdl_instance(k=k, train_size=1000, dir_name=name, reg=reg, mu=mu,
                max_iters=400, n_restarts=2, cost_power=1, 
                mode = 'train_classes', n_clusters=6, 
                label_hard=[1, 10, 11, 12, 13, 14], training_data='true_random_data.pt',
                dummy='nono')
    
