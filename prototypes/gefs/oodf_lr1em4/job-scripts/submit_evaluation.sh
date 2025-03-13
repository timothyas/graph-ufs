#!/bin/bash

#SBATCH -J eval-gefs-oodf_lr1em4
#SBATCH -o /pscratch/sd/t/timothys/graph-ufs/gefs/oodf_lr1em4/slurm/evaluation.%j.out
#SBATCH -e /pscratch/sd/t/timothys/graph-ufs/gefs/oodf_lr1em4/slurm/evaluation.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 03:00:00

project=gefs
subproject=oodf_lr1em4
ckpt_id=18

conda activate graphufs-cpu
cd /global/common/software/m4718/timothys/graph-ufs/prototypes/${project}/${subproject}

#python postprocess_inference.py
. ./evaluate_with_wb2.sh

# cleanup, copy to community
mywork=$WORK/graph-ufs/${project}/${subproject}
mycommunity=$COMMUNITY/graph-ufs/${project}/${subproject}
local_inference_dir=inference/c${ckpt_id}/validation
mkdir -p $mycommunity/${local_inference_dir}
mkdir -p $mycommunity/logs/training
mkdir -p $mycommunity/logs/inference

cp $mywork/loss.nc $mycommunity
cp -r $mywork/models $mycommunity
cp $mywork/logs/training/*.00.*.* $mycommunity/logs/training
cp $mywork/logs/inference/*.00.*.* $mycommunity/logs/inference
cp $mywork/${local_inference_dir}/*.nc $mycommunity/${local_inference_dir}
cp -r $mywork/${local_inference_dir}/graphufs*.zarr $mycommunity/${local_inference_dir}

cd $mycommunity/${local_inference_dir}
mkdir to-psl
cp *.nc to-psl/
cp $mywork/loss.nc to-psl/
cp -r graphufs.240h.spectra.zarr to-psl/
tar -zcvf to-psl.tar.gz to-psl/
