#!/bin/bash

#SBATCH -J eval-gefs-oodd_lr1em4
#SBATCH -o /pscratch/sd/t/timothys/graph-ufs/gefs/oodd_lr1em4/slurm/evaluation.%j.out
#SBATCH -e /pscratch/sd/t/timothys/graph-ufs/gefs/oodd_lr1em4/slurm/evaluation.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 06:00:00

project=gefs
subproject=oodd_lr1em4
ckpt_id=22

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
tar -zcvf to-psl.tar.gz to-psl/ && rm -rf to-psl
echo "Archived and removed to-psl/"

# Now, tar up all the inference directories
cd $mycommunity/${local_inference_dir}
for dir in *.zarr; do
    if [ -d "$dir" ]; then
        tar -zcf "${dir}.tar.gz" "$dir" && rm -rf "$dir"
        echo "Archived and removed $dir"
    else
        echo "Directory $dir does not exist, skipping..."
    fi
done
