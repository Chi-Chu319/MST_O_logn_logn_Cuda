#!/bin/bash

module load gcc/11.3.0 cuda/11.7.0
make

num_vertex_local=32
# thread count
val=16
thread_max=1024
file_dir="weak_scale_complete"

mkdir $file_dir

while [ $num_vertex_local -gt  ]
do
    n_thead=$((thread_max))
    n_block=$((val/thread_max))

    if [ $val -gt $thread_max ]; then
        n_thread=$((thread_max))
        n_block=$((val/thread_max))
    else
        n_block=1
        n_thread=$((val))
    fi


    n=$((val*num_vertex_local)) 

    # Create a new job script for each value
    cat << EOF > ${file_dir}/job_${val}_${num_vertex_local}.sh
#!/bin/bash
#SBATCH --job-name=mpiTest
#SBATCH --account=project_2009665
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=${file_dir}/result/%j_${val}_${num_vertex_local}.txt

module load gcc/11.3.0 cuda/11.7.0

time srun nvprof ./main $n_block $n_thread $n $num_vertex_local
EOF
    # Submit the job script
    sbatch ${file_dir}/job_${val}_${num_vertex_local}.sh

    val=$((val*4))
    num_vertex_local=$((num_vertex_local/2))
done
