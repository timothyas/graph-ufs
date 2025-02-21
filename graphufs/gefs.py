import logging
import numpy as np
import xarray as xr
import pandas as pd
from jax import tree_util

from graphcast import data_utils, solar_radiation
from graphcast.graphcast import ModelConfig, TaskConfig, CheckPoint

from .emulator import ReplayEmulator

class GEFSEmulator(ReplayEmulator):

    n_members = 21

    dim_names = {
        "time": "t0",
        "datetime": "valid_time",
        "level": "pressure",
        "lat": "latitude",
        "lon": "longitude",
    }

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
        return {
            "member": 1,
        }


    def __init__(self, mpi_rank=None, mpi_size=None):

        if self.local_store_path is None:
            warnings.warn(f"{self.name}.__init__: no local_store_path set, data will always be accessed remotely. Proceed with patience.")

        if any(x not in self.input_variables for x in self.target_variables):
            raise NotImplementedError(f"GraphUFS cannot predict target variables that are not also inputs")

        if self.delta_t != "6h":
            raise NotImplementedError(f"{self.name}.__init__: no timestep other than 6h implemented")

        if self.input_duration != "6h":
            raise NotImplementedError(f"{self.name}.__init__: it's unclear how to get two consistent timesteps with GEFS")

        if self.target_lead_time != "6h":
            logging.warning(f"{self.name}.__init__: it's unclear how target_lead_time != 6h will work")

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

        if self.forecast_duration != pd.Timedelta(hours=6):
            raise NotImplementedError(f"{self.name}.__init__: forecast_duration=6h only so far")

        # set normalization here so that we can jit compile with this class
        # a bit annoying, have to copy datatypes here to avoid the Ghost Bus problem
        self.norm_urls = self.norm_urls.copy()
        self.norm = dict()
        self.stacked_norm = dict()
        print("Ignoring normalization for now")
        #self.set_normalization()
        #self.set_stacked_normalization()

        # TOA Incident Solar Radiation integration period
        if self.tisr_integration_period is None:
            self.tisr_integration_period = self.delta_t

    def extract_inputs_targets_forcings(self, sample, drop_datetime=True, keep_member_id=False, **tisr_kwargs):
        """This should mirror graphcast.data_utils.extract_inputs_targets_forcings,
        except that this sample is very easy to get the inputs and targets from... I think
        """

        # preprocess
        sample["datetime"].load()

        # forcings are actually tricky... since we want to compute them as a function of valid_time
        if set(self.forcing_variables) & data_utils._DERIVED_VARS:
            data_utils.add_derived_vars(sample)
        if "toa_incident_solar_radiation" in self.all_variables:
            tisr = xr.concat(
                [
                    solar_radiation.get_toa_incident_solar_radiation_for_xarray(
                        data_array_like=sample.sel(fhr=fhr),
                        **tisr_kwargs,
                    ).expand_dims({"fhr": [fhr]})
                    for fhr in sample.fhr.values
                ],
                dim="fhr",
            )
            sample["toa_incident_solar_radiation"] = tisr

        if drop_datetime:
            sample = sample.drop_vars("datetime")

        # Note: It's unclear how to handle this with multiple ICs
        sample = sample.squeeze("time", drop=True)

        # Rename member to original_member
        sample = sample.rename({"member": "original_member"})
        sample["member"] = xr.DataArray(range(len(sample["original_member"])), coords=sample.original_member.coords)
        sample = sample.set_coords("member")
        sample = sample.swap_dims({"original_member": "member"})

        # inputs will only come from data at fhr=0
        leads = self.target_lead_time
        leads = [leads] if isinstance(leads, str) else leads
        fhrs = [int(pd.Timedelta(lead).value / 1e9 / 3600) for lead in leads]

        inputs = sample.sel(fhr=0, drop=True)
        inputs = inputs[[v for v in self.input_variables]]

        # Note: it's not clear if this squeeze is necessary
        # if it is, it's not clear what will happen with longer fhr's
        targets = sample.sel(fhr=fhrs).squeeze(dim="fhr", drop=True)
        forcings = targets[[v for v in self.forcing_variables]]
        targets = targets[[v for v in self.target_variables]]
        if not keep_member_id:
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

tree_util.register_pytree_node(
    GEFSEmulator,
    GEFSEmulator._tree_flatten,
    GEFSEmulator._tree_unflatten,
)
