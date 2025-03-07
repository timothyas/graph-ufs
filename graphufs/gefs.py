import logging
import numpy as np
import xarray as xr
import pandas as pd
from jax import tree_util

from graphcast import data_utils, solar_radiation
from graphcast.graphcast import ModelConfig, TaskConfig, CheckPoint

from .emulator import ReplayEmulator


class GEFSForecastEmulator(ReplayEmulator):

    dim_names = {
        "time": "t0",
        "datetime": "valid_time",
        "level": "pressure",
        "lat": "latitude",
        "lon": "longitude",
    }

    missing_dates = (
        "2017-09-25 06:00:00",
        "2017-10-28 06:00:00",
        "2018-03-29 18:00:00",
        "2018-03-30 18:00:00",
        "2018-04-20 00:00:00",
        "2018-07-02 06:00:00",
        "2018-07-03 06:00:00",
        "2018-07-12 06:00:00",
        "2019-02-19 06:00:00",
        "2020-01-25 12:00:00",
        "2020-01-30 00:00:00",
        "2020-09-23 12:00:00",
        "2020-09-23 18:00:00",
    )

    possible_stacked_dims = ("batch", "lat", "lon", "channels")

    @property
    def input_dims(self):
        return {
            "time": self.n_input,
            "member": 1,
            "lat": len(self.latitude),
            "lon": len(self.longitude),
            "level": len(self.pressure_levels),
        }

    @property
    def input_overlap(self):
        return dict()


    def __init__(self, mpi_rank=None, mpi_size=None):

        if self.local_store_path is None:
            logging.error(f"{self.name}.__init__: no local_store_path set, data will always be accessed remotely. Proceed with patience.")
            raise ValueError

        if any(x not in self.input_variables for x in self.target_variables):
            raise NotImplementedError(f"GraphUFS cannot predict target variables that are not also inputs")

        if self.delta_t != "6h":
            raise NotImplementedError(f"{self.name}.__init__: no timestep other than 6h implemented")

        if self.input_duration != "6h":
            raise NotImplementedError(f"{self.name}.__init__: it's unclear how to get two consistent timesteps with GEFS, also consider the missing dates!")

        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size

        pressure, latitude, longitude = self._get_gefs_grid(self.resolution)
        self.latitude = tuple(float(x) for x in latitude)
        self.longitude = tuple(float(x) for x in longitude)
        for p in self.pressure_levels:
            assert p in pressure, f"{self.name}.__init__: pressure level {p} not in dataset, must be one of {pressure}"
        self.levels = list(p for p in self.pressure_levels)
        self.model_config = ModelConfig(
            resolution=self.resolution,
            mesh_size=self.mesh_size,
            latent_size=self.latent_size,
            gnn_msg_steps=self.gnn_msg_steps,
            hidden_layers=self.hidden_layers,
            radius_query_fraction_edge_length=self.radius_query_fraction_edge_length,
            mesh2grid_edge_normalization_factor=self.mesh2grid_edge_normalization_factor,
        )
        self.task_config = TaskConfig(
            input_variables=self.input_variables,
            target_variables=self.target_variables,
            forcing_variables=self.forcing_variables,
            pressure_levels=tuple(self.levels),
            input_duration=self.input_duration,
            longitude=self.longitude,
            latitude=self.latitude,
        )

        self.all_variables = tuple(set(
            self.input_variables + self.target_variables + self.forcing_variables
        ))

        # convert some types
        self.delta_t = pd.Timedelta(self.delta_t)
        self.input_duration = pd.Timedelta(self.input_duration)
        lead_times, duration = data_utils._process_target_lead_times_and_get_duration(self.target_lead_time)
        self.forecast_duration = duration

        logging.debug(f"target_lead_time: {self.target_lead_time}")
        logging.debug(f"lead_times: {lead_times}")
        logging.debug(f"self.forecast_duration: {self.forecast_duration}")
        logging.debug(f"self.time_per_forecast: {self.time_per_forecast}")
        logging.debug(f"self.n_input: {self.n_input}")
        logging.debug(f"self.n_forecast: {self.n_forecast}")
        logging.debug(f"self.n_target: {self.n_target}")

        # limited case for now
        if self.delta_t != pd.Timedelta(hours=6):
            raise NotImplementedError(f"{self.name}.__init__: delta_t=6h only so far")

        # set normalization here so that we can jit compile with this class
        # a bit annoying, have to copy datatypes here to avoid the Ghost Bus problem
        self.norm_urls = self.norm_urls.copy()
        self.norm = dict()
        self.stacked_norm = dict()
        self.set_normalization()
        self.set_stacked_normalization()

        # TOA Incident Solar Radiation integration period
        if self.tisr_integration_period is None:
            self.tisr_integration_period = self.delta_t

    def subsample_dataset(self, xds, new_time=None):
        """Get the subset of the data that we want in terms of time, vertical levels, and variables

        Note:
            This is the EXACT same as for the replay emulator, except that it only subsamples the time
            via the bounds, not via the frequency of the "new_time" argument.
            The reason is that we are pulling a forecast dataset, which could have initial conditions at
            any frequency (e.g. just once a day, once a month, whatever)
            but have a separate "fhr" dimension that is the frequency of the model.

        Args:
            xds (xarray.Dataset): with replay data
            new_time (pandas.Daterange or similar, optional): time vector to select from the dataset

        Returns:
            newds (xarray.Dataset): subsampled/subset that we care about
        """

        # only grab variables we care about
        myvars = list(x for x in self.all_variables if x in xds)
        xds = xds[myvars]

        if new_time is not None:
            xds = xds.sel({self.dim_names["time"]: slice(new_time[0], new_time[-1])})

        # select our vertical levels
        xds = xds.sel({self.dim_names["level"]: self.levels})

        # if we have any transforms to apply, do it here
        xds = self.transform_variables(xds)
        return xds

    def extract_inputs_targets_forcings(self, sample, drop_datetime=True, drop_original_member=False, **tisr_kwargs):
        """This should mirror graphcast.data_utils.extract_inputs_targets_forcings,
        except that this sample is very easy to get the inputs and targets from... I think
        """

        # First, make "time" and "member" consistent quantities among samples
        # rename time back to t0, time used here is actually lead_time
        sample = sample.rename({"time": "t0", "lead_time": "time"})
        sample = sample.swap_dims({"fhr": "time"})

        # Rename member to original_member
        # In forecast case, it's always 0
        # In deviation case, it's always [0, 1]
        sample = sample.rename({"member": "original_member"})
        sample["member"] = xr.DataArray(range(len(sample["original_member"])), coords=sample.original_member.coords)
        sample = sample.set_coords("member")
        sample = sample.swap_dims({"original_member": "member"})

        # get valid_time for computing stuff
        sample["datetime"].load()

        # forcings are actually tricky... since we want to compute them as a function of valid_time
        if set(self.forcing_variables) & data_utils._DERIVED_VARS:
            data_utils.add_derived_vars(sample)
        if "toa_incident_solar_radiation" in self.all_variables and "toa_incident_solar_radiation" not in sample:
            tisr = xr.concat(
                [
                    solar_radiation.get_toa_incident_solar_radiation_for_xarray(
                        data_array_like=sample.sel(time=lead_time),
                        **tisr_kwargs,
                    ).expand_dims({"time": [lead_time]})
                    for lead_time in sample.time.values
                ],
                dim="time",
            )
            sample["toa_incident_solar_radiation"] = tisr

        if drop_datetime:
            sample = sample.drop_vars("datetime")

        # always squeeze out the t0 dim
        sample = sample.squeeze("t0", drop=True)
        sample = sample.drop_vars("fhr")

        # inputs will only come from data at fhr=0
        inputs = sample.sel(time=[pd.Timedelta(0)])
        inputs = inputs[[v for v in self.input_variables]]

        # Get target lead times
        leads = self.target_lead_time
        leads = [leads] if isinstance(leads, str) else leads
        leads = [pd.Timedelta(lead) for lead in leads]

        targets = sample.sel(time=leads)
        forcings = targets[[v for v in self.forcing_variables]]
        targets = targets[[v for v in self.target_variables]]
        if drop_original_member:
            inputs = inputs.drop_vars("original_member")
            targets = targets.drop_vars("original_member")
            forcings = forcings.drop_vars("original_member") if "original_member" in forcings else forcings
        return inputs, targets, forcings


    def _get_gefs_grid(self, resolution: int | float):
        if int(resolution) != 1:
            raise NotImplementedError
        latitude = np.arange(89, -90, -1).astype(float)
        longitude = np.arange(360).astype(float)
        pressure = np.array(
            [
                10, 20, 30, 50, 70,
                100, 150, 200, 250, 300, 350, 400, 450,
                500, 550, 600, 650, 700, 750, 800, 850,
                900, 925, 950, 975, 1000,
            ],
            dtype=float,
        )
        return pressure, latitude, longitude

    def open_dataset(self, **kwargs):
        xds = xr.open_zarr(self.data_url, **kwargs)
        xds = xds.sel(latitude=slice(89.5, -89.5))
        return xds


    # Want to make it clear I don't know what will happen with these routines
    def preprocess(self, xds, batch_index=None):
        raise NotImplementedError

    def get_the_data(self, all_new_time, mode):
        raise NotImplementedError

    @staticmethod
    def divide_into_slices(N, K):
        raise NotImplementedError

    @staticmethod
    def rechunk(xds):
        raise NotImplementedError

    def get_batches(self, n_optim_steps, drop_cftime, mode):
        raise NotImplementedError

    @staticmethod
    def get_replay_levels():
        raise NotImplementedError

    def _get_replay_grid(self, resolution):
        raise NotImplementedError

    @classmethod
    def from_parser(cls):
        raise NotImplementedError

class GEFSDeviationEmulator(GEFSForecastEmulator):


    norm_urls = {
        "mean": "",
        "std": "",
        "stddiff": "",
        "deviation_stddev": "",
    }
    forecast_loss_weight = 0.5 # weight of the total forecast component, so each forecast is weighted half of this.
    deviation_loss_weight = 0.5
    possible_stacked_dims = ("batch", "member", "lat", "lon", "channels")

    @property
    def input_dims(self):
        return {
            "time": self.n_input,
            "member": 2,
            "lat": len(self.latitude),
            "lon": len(self.longitude),
            "level": len(self.pressure_levels),
        }

    @property
    def input_overlap(self):
        return dict()

    def calc_loss_weights(self, gds):
        loss_weights = super().calc_loss_weights(gds)

        array = loss_weights["forecast_mse"][:,0,...]

        assert self.forecast_loss_weight + self.deviation_loss_weight == 1, \
            f"{self.name}.calc_loss_weights: forecast_loss_weight={self.forecast_loss_weight} & deviation_loss_weight={self.deviation_loss_weight}, but they should sum to 1"

        individual_fcst_weight = .5*self.forecast_loss_weight
        msg = f"{self.name}.calc_loss_weights: Weighting deviation loss function as follows:\n"
        msg += "\tLoss = Forecast_MSE_1 + Forecast_MSE_2 + Deviation_MSE\n"
        msg += f"\tForecast_MSE_1 = {individual_fcst_weight}\n"
        msg += f"\tForecast_MSE_2 = {individual_fcst_weight}\n"
        msg += f"\tDeviation_MSE = {self.deviation_loss_weight}\n"
        logging.info(msg)

        return {
            "forecast_mse": individual_fcst_weight * array,
            "deviation_mse": self.deviation_loss_weight * array,
        }


tree_util.register_pytree_node(
    GEFSForecastEmulator,
    GEFSForecastEmulator._tree_flatten,
    GEFSForecastEmulator._tree_unflatten,
)

tree_util.register_pytree_node(
    GEFSDeviationEmulator,
    GEFSDeviationEmulator._tree_flatten,
    GEFSDeviationEmulator._tree_unflatten,
)
