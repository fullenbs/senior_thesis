#!/bin/bash 
for i in  60 100 200 400 600 800 1000
do 
    for j in 1 2 3 4 5
    do 
        sbatch run_point.sh $i $j 
    done

done
