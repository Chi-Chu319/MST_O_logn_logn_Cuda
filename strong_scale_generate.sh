#!/bin/bash

make

n=8192
thread_max=1024
file_dir="strong_scale_sparse_${n}_40"

mkdir $file_dir

# Array to store the values: 1, 2, 4, 8, ..., 8192
# values=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536)
values=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)

# Loop through the array values
for val in "${values[@]}"; do
    n_thead=$((thread_max))
    n_block=$((n/thread_max))
    if [ $val -gt $n_block ]; then
        n_thread=$((n/val))
        n_block=1
    else
        n_block=$((n/(val*thread_max)))
        n_thread=$((thread_max))
    fi

    # Create a new job script for each value
    cat << EOF > ${file_dir}/job_${n}_${val}.sh
#!/bin/bash
#SBATCH --job-name=mpiTest
#SBATCH --account=project_2009665
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=${file_dir}/result/%j_${n}_${val}.txt

module load gcc/11.3.0 cuda/11.7.0

time srun nvprof ./main $n_block $n_thread $n $val
EOF

    # Submit the job script
    sbatch ${file_dir}/job_${n}_${val}.sh
done
