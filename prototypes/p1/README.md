# Prototype P1

This prototype aims to reproduce `Graphcast_small` outcomes. `GraphCast_small`, is
a smaller, low-resolution version of GraphCast that uses 1 degree spatial
resolution, 13 pressure levels, and a smaller mesh. Like p0, we would benchmark the
performance using weatherbench 2. Although we have plans to roll out the forecasts upto
10 days in advance, just like the original Graphcast, initial developments of this prototype
would focus on only 3 hr forecasts. Once this is achieved, the autoregressive steps will be
performed to produce longer lead time forecasts.

## Training, Validation, and Testing
The original `Graphcast_small` was trained on ERA5 data from 1979 to 2015, validated on
2016-2017, and tested on 2018-2021. We aim to split the available Replay dataset as
 - Training: 1994 - 2019
 - Validation: 2022-23
 - Test: 2020-2021

The reason for the discontinuity in Training and Validation datasets is the fact that
weatherbench 2 uses 2020-21 for testing purposes, and thus it is not justified
to include these years in the training/validation.

## Inputs, Targets, and Forcings:
The inputs, targets, and forcings used in this model are as follows:

* `land`: land-sea mask [static] 			- INPUT/FORCING
* `hgtsfc`: geopotential at surface [static] 		- INPUT/FORCING
* `delz`: height thickness (proxy for geopotential) [atmos]     - INPUT/TARGET
* `tmp`: temperature [atmos]					- INPUT/TARGET
* `ugrd`: u component of wind [atmos] 				- INPUT/TARGET
* `vgrd`: v component of wind [atmos]				- INPUT/TARGET
* `dzdt`: vertical velocity [atmos]				- INPUT/TARGET
* `spfh`: specific humidity [atmos]				- INPUT/TARGET
* `tmp2m`: 2m temperature [single]				- INPUT/TARGET
* `pressfc`: surface pressure [single]			    - INPUT/TARGET
* `ugrd10m`: 10m u component of wind [single]			- INPUT/TARGET
* `vgrd10m`: 10m v component of wind [single]			- INPUT/TARGET
* `prateb_ave`: total precipitation 3hr [single]	- INPUT/TARGET
* `dswrf_avetoa`: top of atmosphere (toa) incident shortwave solar radiation [single]	- INPUT/FORCING
* `day_progress_cos/sin`: local time of the day [clock]					- INPUT/FORCING
* `year_progress_cos/sin`: local time of the year [clock]				- INPUT/FORCING

Here, `static` means that the property is time independent, `single` means time-varying
single-level property (surface variables included), and `atmos` means time-varying
multi-level atmospheric property.

Note that we should probably compute TOA incident solar radiation, since right
now we're only including the shortwave component.
The full radiation would need to be computed with
`dswrf_avetoa - ulwrf_avetoa - uswrf_avetoa`
Or we could just use the package that GraphCast uses.

## Configuration
The model configuration and the details are provided in `p1.py`.

## Training
For training this configuration, use `run_training.py` as

```bash
python train.py
```
