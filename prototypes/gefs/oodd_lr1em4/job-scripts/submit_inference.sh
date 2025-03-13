#!/bin/bash

#SBATCH -J inference-gefs-oodd-lr1em4
#SBATCH -o /pscratch/sd/t/timothys/graph-ufs/gefs/oodd_lr1em4/slurm/inference.%j.out
#SBATCH -e /pscratch/sd/t/timothys/graph-ufs/gefs/oodd_lr1em4/slurm/inference.%j.err
#SBATCH --nodes=8
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=debug
#SBATCH --account=m4718
#SBATCH --constraint=gpu&hbm80g
#SBATCH -t 00:30:00

conda activate /global/common/software/m4718/timothys/graphufs
export MPI4JAX_USE_CUDA_MPI=1

cd /global/common/software/m4718/timothys/graph-ufs/prototypes/gefs/oodd_lr1em4
srun $COMMON/select_gpu_device python inference.py
