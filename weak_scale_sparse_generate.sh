#!/bin/bash

module load gcc/11.3.0 cuda/11.7.0
make

n=8192
thread_max=1024
file_dir="weak_scale_sparse_${n}_20"

mkdir $file_dir

# Array to store the values: 1, 2, 4, 8, ..., 8192
values=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
# values=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536)
# values=(32768 34816 36864 38912 40960 43008 45056 47104 49152 51200 53248 55296 57344 59392 61440 63488 65536)

# Loop through the array values
for val in "${values[@]}"; do
    n_thead=$((thread_max))
    n_block=$((val/thread_max))
    if [ $val -gt $thread_max ]; then
        n_thread=$((thread_max))
        n_block=$((val/thread_max))
    else
        n_block=1
        n_thread=$((val))
    fi

    # Create a new job script for each value
    cat << EOF > ${file_dir}/job_${val}.sh
#!/bin/bash
#SBATCH --job-name=mpiTest
#SBATCH --account=project_2009665
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=${file_dir}/result/%j_${val}.txt

module load gcc/11.3.0 cuda/11.7.0

time srun nvprof ./main $n_block $n_thread $val 1
EOF

    # Submit the job script
    sbatch ${file_dir}/job_${val}.sh
done
