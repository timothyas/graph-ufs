#!/bin/bash

#SBATCH -J postprocess-perturbation
#SBATCH -o /pscratch/sd/t/timothys/p2/slurm/postprocess.%j.out
#SBATCH -e /pscratch/sd/t/timothys/p2/slurm/postprocess.%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --qos=debug
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 00:30:00

conda activate graphufs-cpu

cd /global/common/software/m4718/timothys/graph-ufs/prototypes/p2/perturbation
python postprocess.py
