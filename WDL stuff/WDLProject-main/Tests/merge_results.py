import pathlib
import numpy as np
import argparse

#When stuff is downloaded from the cluster, it is very compartmentalized, with NN
#files split up across many directories. This merges them into one array. 
#Variables:
#dir_name: directory looking through (str), mode:NN-mode (str), in_here (bool)= 
#reshapes to be right size if true, false otherwise.
def merge_process(dir_name, mode, in_here=True):
    k = np.array([[]])
    for path in pathlib.Path(dir_name).iterdir():
        try: 
            path_temp = str(path)
            temp = np.load(path_temp + '/nn_results_' + mode + '.npy')
            temp = temp[~np.all(temp == 0, axis=1)] #Removes all 0 rows
            k = np.append(k, temp)
        except: 
            print(str(path))
    if in_here == False:
        return k
    else:
        k = np.reshape(k, (int(k.shape[0]/5), 5))
        return k
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    args = parser.parse_args()
    dir_name = args.root

    X = merge_process(dir_name, 'or', in_here=True)
    np.save('NN_results.npy', X)