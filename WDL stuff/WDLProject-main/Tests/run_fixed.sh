#!/bin/bash 
for i in 2 4 8 12 16 20 24 28 32 34 36
do 
    for j in 10 100 1000 10000 
    do 
        for k in 0 1
        do
            sbatch run.sh $i $j $k true 
        done
    done
done
