#!/bin/bash

# if [ ! -f "$DATA_PATH" ] || [ ! -f "$LABEL_PATH" ]; then
#     python data_preprocess.py
# fi


REPEAT_TIMES=1

while getopts "r:" opt; do
  case $opt in
    r) REPEAT_TIMES=$OPTARG ;;  # -r 参数用于指定重复次数
    *) echo "Usage: $0 [-r repeat_times]" ;;
  esac
done

# Set OpenMP to use a maximum of 18 threads for parallel computation
# Help the program better utilize the computational power of multi-core processors.

export OMP_NUM_THREADS=10

# --nproc_per_node=3
# Specify the number of processes to be launched on each node. 
# 3 means launching 3 processes on the local node, with each process typically bound to a single GPU.

# --nnodes=1
# Only one compute node will participate in the training.
# If training requires multiple nodes (i.e., distributed training across multiple physical machines)
# This number should be increased.

# --rdzv_backend=c10d
# The c10d backend is used for efficient communication between multiple processes and machines.
# It supports communication methods like NCCL, Gloo, and MPI.
# In most GPU training scenarios, the c10d backend is commonly used.

# --max_restarts=3
# Specify the maximum number of retries when a failure occurs during training.
# This option is typically used for fault tolerance during training.
# If a process terminates due to an error, PyTorch will attempt to restart the process, with a maximum of 3 retries.

for ((i=0; i<REPEAT_TIMES; i++))
do
    echo "Running iteration $i"

    # python data_split_shuffle.py
    torchrun --nproc_per_node=3 --nnodes=1 --rdzv_backend=c10d --max_restarts=3 train.py --train_iteration=$i

    echo "Iteration $i completed."
done