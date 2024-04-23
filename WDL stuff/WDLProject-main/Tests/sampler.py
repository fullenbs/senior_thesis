import helper
import torch
import argparse

#Generates quick random sample, samples same amount from each class
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int)
    args = parser.parse_args()
    size = args.size
    data = helper.data_loader('data')
    (train_data, lst, train_classes) = helper.sample(data, size, mode='train_classes', 
                                    n_labels=6, label_hard=[1, 10, 11, 12, 13, 14])
    
    torch.save(train_data, 'testing_data.pt')
    torch.save(lst, 'testing_index.pt')
