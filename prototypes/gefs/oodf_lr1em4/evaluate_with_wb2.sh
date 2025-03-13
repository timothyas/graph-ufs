#!/bin/bash

# Note that we have to have pip install --no-deps weatherbench2 as in the README
#

project=gefs
subproject=oodf_lr1em4
ckpt_id=18

output_dir=/pscratch/sd/t/timothys/graph-ufs/${project}/${subproject}/inference/c${ckpt_id}/validation
wb2_dir=$COMMON/graph-ufs/weatherbench2
gefs_path=/pscratch/sd/t/timothys/gefs/one-degree/forecasts.validation.zarr
gefs_mean_path=/pscratch/sd/t/timothys/gefs/one-degree/ensemble-mean.validation.zarr

forecast_duration="240h"
time_start="2019-07-01T00"
time_stop="2019-12-31T18"
surface_variables="surface_pressure,10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature"
noaa_variables="2m_specific_humidity"
level_variables="temperature,specific_humidity,u_component_of_wind,v_component_of_wind,vertical_velocity,geopotential"

levels=250,500,850

truth_names=("era5")
truth_paths=( \
    "gs://gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr" \
)
rename_variables='{"sp":"surface_pressure","u10":"10m_u_component_of_wind","v10":"10m_v_component_of_wind","t2m":"2m_temperature","t":"temperature","u":"u_component_of_wind","v":"v_component_of_wind","w":"vertical_velocity","q":"specific_humidity","sh2":"2m_specific_humidity","lat":"latitude","lon":"longitude","pressure":"level","gh":"geopotential","t0":"time"}'

# Standard WB2 deterministic evaluation
for dataset in "graphufs" "graphufs.ensemble-mean"
do
    forecast_path=${output_dir}/${dataset}.${forecast_duration}.zarr

    for i in "${!truth_names[@]}"
    do
        truth_name=${truth_names[i]}
        truth_path=${truth_paths[i]}

        echo "Evaluating ${dataset} against ${truth_name} ..."
        python ${wb2_dir}/scripts/evaluate.py \
          --forecast_path=${forecast_path} \
          --obs_path=${truth_path} \
          --by_init=True \
          --output_dir=${output_dir} \
          --output_file_prefix=${dataset}_vs_${truth_name}_${forecast_duration}_ \
          --eval_configs=deterministic,deterministic_spatial \
          --ensemble_dim=member \
          --time_start=${time_start} \
          --time_stop=${time_stop} \
          --evaluate_climatology=False \
          --evaluate_persistence=False \
          --variables="${surface_variables},${level_variables}" \
          --levels=${levels} \
          --rename_variables=${rename_variables} \
          --skipna=True
    done

    echo "Computing spectra for ${dataset} ..."
    python ${wb2_dir}/scripts/compute_zonal_energy_spectrum.py \
      --input_path=${forecast_path} \
      --output_path=${output_dir}/${dataset}.${forecast_duration}.spectra.zarr \
      --base_variables="${surface_variables},${level_variables}" \
      --time_dim="time" \
      --time_start=${time_start} \
      --time_stop=${time_stop} \
      --levels=${levels} \
      --averaging_dims="time" \
      --rename_variables=${rename_variables}
done

echo "Comparing GraphUFS vs GEFS"
python ${wb2_dir}/scripts/evaluate.py \
  --forecast_path=${output_dir}/graphufs.${forecast_duration}.zarr \
  --obs_path=${gefs_path} \
  --by_init=True \
  --output_dir=${output_dir} \
  --output_file_prefix=graphufs_vs_gefs_${forecast_duration}_ \
  --eval_configs=deterministic,deterministic_spatial \
  --time_start=${time_start} \
  --time_stop=${time_stop} \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --variables="${surface_variables},${level_variables},${noaa_variables}" \
  --rename_variables=${rename_variables} \
  --levels=${levels}

echo "Comparing Ensemble Mean GraphUFS vs GEFS"
python ${wb2_dir}/scripts/evaluate.py \
  --forecast_path=${output_dir}/graphufs.ensemble-mean.${forecast_duration}.zarr \
  --obs_path=${gefs_mean_path} \
  --by_init=True \
  --output_dir=${output_dir} \
  --output_file_prefix=graphufs.ensemble-mean_vs_gefs.ensemble-mean_${forecast_duration}_ \
  --eval_configs=deterministic,deterministic_spatial \
  --time_start=${time_start} \
  --time_stop=${time_stop} \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --variables="${surface_variables},${level_variables},${noaa_variables}" \
  --rename_variables=${rename_variables} \
  --levels=${levels}
