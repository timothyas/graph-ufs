# graph-ufs: Prototype P1

This prototype aims to reproduce `Graphcast_small` outcomes. `GraphCast_small`, is
a smaller, low-resolution version of GraphCast that uses 1 degree spatial
resolution, 13 pressure levels, and a smaller mesh. While the original `Graphcast_small` 
is trained on ERA5 data from 1979 to 2015, we aim to train it on full length 1/4 degrees 
Replay dataset coarsened to 1 degree. 
The inputs, targets, and forcings used in this model are as follows:

# INPUTS, TARGETS, and FORCINGS:

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
* total precipitation 3hr (we need 6 hrs though) [single]	- INPUT/TARGET
* top of atmosphere (toa) incident solar radiation [single]	- INPUT/FORCING
* local time of the day [clock]					- INPUT/FORCING	
* local time of the year [clock]				- INPUT/FORCING
 
# Pressure Levels [13]
50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 (native pressure levels)  

# Graphcast Model Configuration
* `delta_t` = 6 hr
	the model time step

* `target_lead_time` = 6 hr
	the model forecast lead time

* resolution = 1 degrees 
	nominal spatial resolution

* `mesh_size` = 5 
   	how many refinements to do on the multi-mesh

* `latent_size` = 512
	how many latent features to include in the various MLPs

* `gnn_msg_steps` = 16
	 how many Graph Network message passing steps to do.

* `hidden_layers` = 1
	number of hidden layers for each MLP

* `radius_query_fraction_edge_length` = 0.6
	Scalar that will be multiplied by the
        length of the longest edge of the finest mesh to define the radius of
        connectivity to use in the Grid2Mesh graph. Reasonable values are
        between 0.6 and 1. 0.6 reduces the number of grid points feeding into
        multiple mesh nodes and therefore reduces edge count and memory use, but
        1 gives better predictions.

* `mesh2grid_edge_normalization_factor` = 0.6180 (approx.)
	Allows explicitly controlling edge normalization for mesh2grid edges. 
	If None, defaults to max edge length.This supports using pre-trained 
	model weights with a different graph structure to what it was trained on.

# Replay dataset details
* resolution = 1 degree
	obtained by coarsening the original 1/4 degree dataset by subsampling (TBD)

* `training_dates` = Dec 31, 1993 18:00:00 - Dec 31, 1994, 18:00:00
	both dates are inclusive (this will be changed as the new dataset becomes available)
