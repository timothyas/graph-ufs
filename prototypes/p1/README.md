# graph-ufs: Prototype P1

This prototype aims to reproduce `Graphcast_small` outcomes. `GraphCast_small`, is
a smaller, low-resolution version of GraphCast that uses 1 degree spatial
resolution, 13 pressure levels, and a smaller mesh. Like p0, we would benchmark the
performance using weatherbench 2. Although we have plans to roll out the forecasts upto 
10 days in advance, just like the original Graphcast, initial developments of this prototype 
would focus on only 3 hr forecasts. Once this is achieved, the autoregressive steps will be 
performed to produce longer lead time forecasts.

# Training, testing, and validation
The original `Graphcast_small` was trained on ERA5 data from 1979 to 2015, validated on 
2016-2017, and tested on 2018-2021. We aim to split the available Replay dataset as 
 - Training: 1994 - 2019
 - Validation: 2022-23
 - Test: 2020-2021

The reason for the discontinuity in Training and Validation datasets is the fact that 
weatherbench 2 uses 2020-21 for testing purposes, and thus it is not justified
to include these years in the training/validation.  

# INPUTS, TARGETS, and FORCINGS:
The inputs, targets, and forcings used in this model are as follows: 

* land-sea mask [static] 			- INPUT/FORCING
* geopotential at surface [static] 		- INPUT/FORCING
* height thickness (proxy for geopotential) [atmos]     - INPUT/TARGET
* temperature [atmos]					- INPUT/TARGET
* u component of wind [atmos] 				- INPUT/TARGET
* v component of wind [atmos]				- INPUT/TARGET
* vertical velocity [atmos]				- INPUT/TARGET
* specific humidity [atmos]				- INPUT/TARGET
* 2m temperature [single]				- INPUT/TARGET
* mean sea level pressure [single]			- INPUT/TARGET
* 10m u component of wind [single]			- INPUT/TARGET
* 10m v component of wind [single]			- INPUT/TARGET
* total precipitation 3hr [single]	- INPUT/TARGET
* top of atmosphere (toa) incident solar radiation [single]	- INPUT/FORCING
* local time of the day [clock]					- INPUT/FORCING	
* local time of the year [clock]				- INPUT/FORCING

Here, `static` means that the property is time independent, `single` means time-varying 
single-level property (surface variables included), and `atmos` means time-varying
multi-level atmospheric property.
   
# Configuration
The model configuration and the details are provided in `p1.py`.

# Training
For training this configuration, use `run_training.py` as 

```bash
python run_training.py 
```

# Normalization
Unlike in p0, here we aim to compute our own per-pressure level mean and variance to 
normalize the inputs to zero mean and unit standard deviation. 

# Coarsening Replay Dataset
The 1 degree Replay Dataset is only available for 5.5 years, which is not long enough for
training. Therefore, we aim to use the 1/4 degree Replay data by coarsening it to 1 degree
using subsampling, i.e., by picking every 4th point on the gaussian grid. One concern is that
the dataset achieved in this fashing is not colocated with 1 degree gaussian grid in the
latitudinal direction. This is something to just keep in mind, but this should not alter the 
results qualitatively.

