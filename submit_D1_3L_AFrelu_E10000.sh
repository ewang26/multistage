#!/bin/bash
#SBATCH --partition=serc
#SBATCH --job-name=test1a
#SBATCH --output=test_%A.out  
#SBATCH --error=test_%A.out 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=10G

module load python/3.9.0

source ../venvs/sciml3/bin/activate
python3 D1_3L_AFrelu_E10000.py
