#!/bin/bash

#SBATCH -J distributed
#SBATCH -o slurm/distributed.%j.out
#SBATCH -e slurm/distributed.%j.err
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=gpu2
#SBATCH -t 00:05:00

source /contrib2/Tim.Smith/miniconda3/etc/profile.d/conda.sh
conda activate graphufs

ifconfig
srun python test_jax_distributed.py
