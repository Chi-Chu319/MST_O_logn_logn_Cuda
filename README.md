# MST_O_logn_logn_Cuda


## Overview

This repository contains an implementation of the distributed Minimum Spanning Tree (MST) algorithm with O(log log n) complexity. The algorithm is based on the research paper "MST construction in O(log log n) communication rounds" presented in June 2003 (DOI:10.1145/777412.777428).

The implementation is written in C++ (Cuda)

## Prerequisites

Before building and running the project, ensure that you have the following installed:
- An available Cuda-enabled GPU
- A C++ compiler supporting C++17 or later

## How to Build
Before building the project, check for the compute compatibility of the GPU in use. And change the `XX` of the `NVCC_FLAGS := -arch=sm_XX -lineinfo` in the `makefile` to the corresponding number.

To build the project, you need to load the necessary module for the Cuda library and then use `make` to compile the code. Follow these steps:

1. Load the Boost module with MPI support:
```
module load gcc/11.3.0 cuda/11.7.0
```

2. Compile the project:
```
make
```

## How to Run

After building the project, you can run it using the provided batch script. This script submits a job to an Cuda environment. To submit the job, execute the following command:
```
sbatch mst_job.sh
```

Ensure that `mst_job.sh` is properly configured for your computing environment and job scheduler (e.g., Slurm, PBS).

## How to Debug

1. Use the compute-sanitizer:
```
srun compute-sanitizer main
```
