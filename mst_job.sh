#!/bin/bash
#SBATCH --job-name=mpiTest
#SBATCH --account=project_2009665
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=sparse_composition_300/composition_%j.txt

module load gcc/11.3.0 cuda/11.7.0

# time srun nvprof ./main 2 1024 2048 1
# time srun nvprof ./main 4 1024 4096 1
# time srun nvprof ./main 8 1024 8192 1
# time srun nvprof ./main 16 1024 16384 1
# time srun nvprof ./main 32 1024 32768 1
time srun nvprof ./main 64 1024 65536 1
