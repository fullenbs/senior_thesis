#!/bin/bash 
for i in  2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36
do 
    for j in 3 5 8 10 
    do 
        sbatch run.sh $i $j $false 
    done

done
