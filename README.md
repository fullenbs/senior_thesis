# Senior thesis

All code specifically for the WDL algorithm and any code in the folders wdl, kmeans, old, and utilities are attributed here: 
https://github.com/MarshMue/GeoWDL-TAG-ML-ICML-2023

Information on the Salinas A sensor used for constructing the cost matrix is here:
https://purr.purdue.edu/publications/1947/1

All code here is for the completion of a Senior Honors Thesis at Tufts University. For any questions please contact me at sfulle03@tufts.edu 

If you are interested in running these experiments these are some of the core dependencies: 

I recommend a Conda environment to maintain. Scripts to run experiments on Salinas A were run on the Tufts HPC Cluster. For specific packages, this isn't an exhaustive list, but here are versions used for the main ones:
1. ```numpy 1.23.5```
2. ```pytorch 1.13.1```
3. ```POT 0.8.2 ```
4. ```scikit-learn 1.2.1```

The code was also run using ```Python 3.9.15```

The above packages are not a fully comprehensive list, but these are the more technical packages that play a vital role. 

These experiments overlap with the experiments performed in the WDL_HSI repo linked here: https://github.com/fullenbs/WDL_HSI . The information to recreate some of the core experiments is also listed there. 

Now, to begin outlining the many folders and the information inside of them, all results are stored in the subdirectory "WDL Stuff"/WDLProject-main/Tests. Here is a comprehensive list of the experiments stored in each folder: 
1. ```Salinas_A_experiments```: WDL unmixing and clustering on the parameters listed on each directory name. Data is fixed data stored in common_data.pt and the indexes are stored in common_index.pt For naming, mu = geometric regularizer and reg = entropic regularize.
2. ```NN matrices```: The npy files store all NN results for Salinas A experiments. For reference, or-NN is the same as mutual-NN. 
3. ```Robustness Test```: WDL unmixing and clustering on a random sample over uniform sampling.
4. ```WDL_150```: Early WDL results with 150 iterations and 2 restarts.
5. ```WDL_250```: Early WDL results with 250 iterations and 2 restarts.
6. ```random_sample_tests```: WDL results with a random sample of half of the labeled data.
7. ```wasser_kmeans_sampling```: Results where sample was done with Wasserstein-kmeans++
8. ```Label_removed```: Clustering experiments with the annoying lower right label removed from sampling.
9. ```Synthetic Experiments```: All synthetic results with all details
10. ```slices```: All 201 slices of Salinas A HSI
11. Anything with ```ssl``` in the name is related to ssl
12. If a directory wasn't mentioned here, it just contains figures modified for presentation purposes.

There are two main code files: 
1. ```helper.py```: This can recreate all results, and most experiments.
2. ```semisup.py```: This does all the SSL stuff

There are other files with code also outlined in the aforementioned WDL_HSI repo above. Those recreate specific experiments as one-off scripts for ease of use. The main two code files have comments outlining their use cases. 
