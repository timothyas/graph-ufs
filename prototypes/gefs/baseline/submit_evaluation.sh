#!/bin/bash

#SBATCH -J eval-gefs-baseline
#SBATCH -o /pscratch/sd/t/timothys/graph-ufs/gefs/baseline/slurm/evaluation.%j.out
#SBATCH -e /pscratch/sd/t/timothys/graph-ufs/gefs/baseline/slurm/evaluation.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 08:00:00

project=gefs
subproject=baseline

conda activate graphufs-cpu
cd /global/common/software/m4718/timothys/graph-ufs/prototypes/${project}/${subproject}

python postprocess_inference.py
. ./evaluate_with_wb2.sh

# cleanup, copy to community
mywork=$WORK/graph-ufs/${project}/${subproject}
mycommunity=$COMMUNITY/graph-ufs/${project}/${subproject}
local_inference_dir=inference/validation
mkdir -p $mycommunity/${local_inference_dir}

cp $mywork/${local_inference_dir}/*.nc $mycommunity/${local_inference_dir}
cp -r $mywork/${local_inference_dir}/*.zarr $mycommunity/${local_inference_dir}

cd $mycommunity/${local_inference_dir}
mkdir to-psl
cp *.nc to-psl/
cp -r *.spectra.zarr to-psl/
tar -zcf to-psl.tar.gz to-psl/ && rm -rf to-psl
echo "Archived and removed to-psl/"

# Now, tar up all the inference directories
cd $mycommunity/${local_inference_dir}
for dir in *.zarr; do
    echo "Archiving $dir..."
    if [ -d "$dir" ]; then
        tar -zcf "${dir}.tar.gz" "$dir" && rm -rf "$dir"
        echo "Archived and removed $dir"
    else
        echo "Directory $dir does not exist, skipping..."
    fi
done
