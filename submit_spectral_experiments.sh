#!/bin/bash
#SBATCH --partition=serc
#SBATCH --job-name=test1a
#SBATCH --output=test_%A.out  
#SBATCH --error=test_%A.out 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=16:00:00
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:A100_SXM4
#SBATCH --mem-per-cpu=10G

module load python/3.9.0

source ../venvs/sciml3/bin/activate
python3 /home/users/erikwang/multistage/scripts/spectral_bias/spectral_bias_different_kernels.py
