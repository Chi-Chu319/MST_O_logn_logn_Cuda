#!/bin/bash
#SBATCH --job-name=mpiTest
#SBATCH --account=project_2009665
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=strong_scale_sparse_32768_10/result/%j_32768_256.txt

module load gcc/11.3.0 cuda/11.7.0

time srun nvprof ./main 1 128 32768 256
