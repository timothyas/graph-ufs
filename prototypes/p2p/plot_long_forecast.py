import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import pandas as pd

from matplotlib.colorbar import ColorbarBase
import cartopy.crs as ccrs
import cmocean

from ufs2arco import Layers2Pressure
from graphufs.log import setup_simple_log

def plot_t2m(pda, tda):

    fig, axs = plt.subplots(
        3, 1,
        figsize=(10,11),
        constrained_layout=True,
        sharex=True,
    )

    for xda, title, ax in zip(
        [pda, tda],
        ["GraphUFS 2m Temperature", "Replay 2m Temperature"],
        axs,
    ):
        plotme = xda.mean("lon").squeeze()
        plotme = plotme - 273.15
        plotme.plot.contourf(
            ax=ax,
            x="time",
            cmap="cmo.thermal",
            vmin=-10, vmax=30,
            levels=11,
            cbar_kwargs={"label": r"$^\circ$C"},
        )
        ax.set(xlabel="", ylabel="Latitude", title=title)

    plotme = pda.mean("lon").squeeze() - tda.mean("lon").squeeze()
    plotme.plot.contourf(
        ax=axs[-1],
        x="time",
        cmap="cmo.balance",
        levels=11,
        vmin=-5,
        vmax=5,
        cbar_kwargs={"label": r"$^\circ$C"},
    )
    axs[-1].set(xlabel="", ylabel="Latitude", title="Top - Middle")
    return fig, axs

def plot_z500(pda, tda):

    fig, axs = plt.subplots(
        2, 1,
        figsize=(10,7),
        constrained_layout=True,
        sharex=True,
    )

    for xda, title, ax in zip(
        [pda, tda],
        ["GraphUFS Z500", "ERA5 Z500"],
        axs,
    ):
        plotme = xda.mean("lon").squeeze()
        plotme = plotme / 9.80665 / 1000
        plotme.plot.contourf(
            ax=ax,
            x="time",
            cmap="Spectral_r",
            vmin=4.9, vmax=5.9,
            levels=11,
            cbar_kwargs={"label": r"km"},
        )
        ax.set(xlabel="", ylabel="Latitude", title=title)
    return fig, axs

def calc_z500(Emulator, xds):
    mds = xr.open_zarr(Emulator.norm_urls["mean"], storage_options={"token": "anon"})
    for key in ["ak", "bk"]:
        xds[key] = mds[key]
        xds = xds.set_coords(key)

    lp = Layers2Pressure(ak=mds["ak"].values, bk=mds["bk"].values, level_name="level")
    xds["delz"] = lp.calc_delz(xds["pressfc"], xds["tmp"], xds["spfh"])
    xds["geopotential"] = lp.calc_geopotential(hgtsfc=xds["hgtsfc_static"], delz=xds["delz"])
    prsl = lp.calc_layer_mean_pressure(xds["pressfc"], xds["tmp"], xds["spfh"], xds["delz"])

    p = 500
    cds = lp.get_interp_coefficients(p*100, prsl)
    cds.load();
    z500 = lp.interp2pressure(xds["geopotential"], p*100, prsl, cds)
    return z500


def main(
    Emulator,
    ckpt_path=None,
    t0="2019-01-01T03",
    tf="2019-12-31T21",
):

    setup_simple_log()
    forecast_dir = f"{Emulator.local_store_path}/long-forecasts"
    fig_dir = f"{forecast_dir}/figures"

    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    pds = xr.open_zarr(f"{forecast_dir}/graphufs.{t0}.{tf}.zarr")
    pds = pds.sel(t0=t0)

    tds = xr.open_zarr(Emulator.data_url, storage_options={"token": "anon"})
    tds = tds.rename({
        "grid_yt": "lat",
        "grid_xt": "lon",
    })
    tds = tds.sel(time=pds.time.values)

    # 2m temperature
    logging.info(f"Loading predicted t2m\n{pds.tmp2m}\n")
    pds["tmp2m"] = pds.tmp2m.load()
    logging.info(" ... done")
    logging.info(f"Loading true t2m\n{tds.tmp2m}\n")
    tds["tmp2m"] = tds.tmp2m.load()
    logging.info(" ... done")

    fig, axs = plot_t2m(pds["tmp2m"], tds["tmp2m"])
    fig.savefig(f"{fig_dir}/zonal_mean_tmp2m.jpeg", bbox_inches="tight", dpi=300)

    # geopotential
    # subsample to 6h
    pds = pds.isel(time=slice(None, None, 2))
    pds["hgtsfc_static"] = tds["hgtsfc_static"].load()

    era = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
        storage_options={"token": "anon"},
    )
    era = era.sel(
        time=pds.time.values,
    )
    era = era.rename({"latitude": "lat", "longitude": "lon"})
    tz500 = era["geopotential"].sel(level=500)

    logging.info(f"Loading true z500\n{tz500}\n")
    tz500 = tz500.load();
    logging.info(" ... done")

    logging.info(f"Computing predicted z500")
    pz500 = calc_z500(Emulator, pds)
    logging.info(" ... done")

    logging.info(f"Loading predicted z500\n{pz500}\n")
    pz500 = pz500.load();
    logging.info(" ... done")
    fig, axs = plot_z500(pz500, tz500)
    fig.savefig(f"{fig_dir}/zonal_mean_z500.jpeg", bbox_inches="tight", dpi=300)
