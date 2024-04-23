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
python3 sample_robustness.py "--atoms=32" "--geom=0.001" "--reg=0.1" "--count=$1"