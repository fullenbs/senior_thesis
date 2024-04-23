#!/bin/bash
#SBATCH -J kmeans_init 
#SBATCH --time=00-48:00:00 
#SBATCH -p batch #running on mpi partition
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=48g
#SBATCH --output=MyJob.%j.%N.out #output file
#SBATCH --error=MyJob.%j.%N.out #Error file
#SBATCH --mail-type=END
#SBATCH --mail-user=sfulle03@tufts.edu


module load anaconda/2021.05
source activate /cluster/home/sfulle03/condaenv/HSI #Path to conda environment goes here
python3 test.py "--points=$1" "--count=$2"

