import os
import xarray as xr
import subprocess

from graphufs import StatisticsComputer, add_derived_vars
from graphufs.log import setup_simple_log

from config import GEFSEmulator as Emulator, _scratch

_3dvars = (
    "u",
    "v",
    "w",
    "t",
    "q",
    "log_q",
)

def main(varname):

    setup_simple_log()
    normer = StatisticsComputer(
        path_in=Emulator.data_url,
        path_out=os.path.dirname(Emulator.norm_urls["mean"]),
        start_date=Emulator.training_dates[0],
        end_date=Emulator.training_dates[-1],
        load_full_dataset=varname not in _3dvars,
        transforms=Emulator.input_transforms,
        to_zarr_kwargs={"mode": "a"},
        dims=("time", "fhr", "member", "lat", "lon"),
        delta_t="6h",
        rename={
            "t0": "time",
            "pressure": "level",
            "latitude": "lat",
            "longitude": "lon",
        },
    )
    normer(varname, integration_period="6h")


def submit_slurm_job(name, varlist):

    logdir = f"{_scratch}/gefs/one-degree/statistics/slurm/statistics"
    scriptdir = f"./job-scripts"
    for d in [logdir, scriptdir]:
        if not os.path.isdir(d):
            os.makedirs(d)

    if name == "surface":
        time = "06:00:00"
        n_tasks = 6

    else:
        time = "12:00:00"
        n_tasks = 3

    jobscript = f"#!/bin/bash\n\n"+\
        f"#SBATCH -J {name}_stats\n"+\
        f"#SBATCH -o {logdir}/{name}.%j.out\n"+\
        f"#SBATCH -e {logdir}/{name}.%j.err\n"+\
        f"#SBATCH --nodes=1\n"+\
        f"#SBATCH --ntasks={n_tasks}\n"+\
        f"#SBATCH --cpus-per-task=30\n"+\
        f"#SBATCH --qos=regular\n"+\
        f"#SBATCH --account=m4718\n"+\
        f"#SBATCH --constraint=cpu\n"+\
        f"#SBATCH -t {time}\n\n"+\
        f"cd {os.getenv('PWD')}\n"+\
        f"conda activate $graphufs\n"

    for varname in varlist:
        jobscript += f"srun -n 1 python -c 'from calc_statistics import main ; main(\"{varname}\")' > {logdir}/{varname}.log 2>&1\n"

    scriptname = f"{scriptdir}/submit_{name}_statistics.sh"
    with open(scriptname, "w") as f:
        f.write(jobscript)

    subprocess.run(f"sbatch {scriptname}", shell=True)

if __name__ == "__main__":

    varsets = {
        "surface": [
            # Surface Variables
            "u10",
            "v10",
            "t2m",
            "sh2",
            "log_sh2",
            "sp",
            # Forcing Variables at Input Time
            "toa_incident_solar_radiation",
            "year_progress_sin",
            "year_progress_cos",
            "day_progress_sin",
            "day_progress_cos",
            # Static Variables
            "lsm",
            "orog",
        ],
        "dynamics": [
            # 3D Variables
            "u",
            "v",
            "w",
        ],
        "thermodynamics": [
            "t",
            "q",
            "log_q",
        ],
    }

    for name, varlist in varsets.items():
        submit_slurm_job(name, varlist)
