#!/bin/bash

#SBATCH -J perturb-p2
#SBATCH -o /pscratch/sd/t/timothys/p2/slurm/inference.%j.out
#SBATCH -e /pscratch/sd/t/timothys/p2/slurm/inference.%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=gpu&hbm80g
#SBATCH -t 02:00:00

conda activate /global/common/software/m4718/timothys/graphufs

cd /global/common/software/m4718/timothys/graph-ufs/prototypes/p2/perturbation
srun $COMMON/select_gpu_device python run_perturbation.py
