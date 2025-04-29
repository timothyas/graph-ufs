import numpy as np
import xarray as xr
import pandas as pd

import cmocean
import xmovie

import cartopy.crs as ccrs

from graphufs.spatialmap import get_extend

def movie_func(xds, fig, time, *args, **kwargs):

    axs = []
    dalist = []

    truthname = [y for y in list(xds.data_vars) if y not in ("GraphUFS",)][0]
    vtime = xds["time"].isel(time=time).values
    stime = str(vtime)[:13]

    # Plot the two forecast datasets
    for i, name in enumerate([truthname, "GraphUFS"]):
        ax = fig.add_subplot(
            1, 2, i+1,
            projection=ccrs.Orthographic(
                central_longitude=-80,
                central_latitude=20,
            ),
        )
        axs.append(ax)

        plotme = xds[name].isel(time=time)
        dalist.append(plotme)
        p = plotme.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)
        ax.set(
            xlabel="",
            ylabel="",
            title=name,
        )
    # now the colorbar
    extend, kwargs["vmin"], kwargs["vmax"] = get_extend(
        dalist,
        kwargs.get("vmin", None),
        kwargs.get("vmax,", None),
    )
    label = xds.attrs.get("label", "")
    label += f"\n{stime}"
    fig.colorbar(
        p,
        ax=axs,
        orientation="horizontal",
        shrink=.8,
        pad=0.05,
        aspect=35,
        label=label,
        extend=extend,
    )
    fig.set_constrained_layout(True)

    return None, None

def calc_wind_speed(xds):
    if "ugrd10m" in xds:
        ws = np.sqrt(xds["ugrd10m"]**2 + xds["vgrd10m"]**2)
    else:
        ws = np.sqrt(xds["10m_u_component_of_wind"]**2 + xds["10m_v_component_of_wind"]**2)
    ws.attrs["units"] = "m/sec"
    ws.attrs["long_name"] = "10m Wind Speed"
    return ws

def get_truth(name):
    if name.lower() == "era5":
        url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
        rename = {}
    elif name.lower() == "replay":
        url = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr"
        rename = {"pfull": "level", "grid_yt": "lat", "grid_xt": "lon"}

    truth = xr.open_zarr(
        url,
        storage_options={"token": "anon"},
    )
    truth = truth.rename(rename)
    truth.attrs["name"] = name
    return truth


def main(Emulator, t0="2019-01-01T03", tf="2019-12-31T21", ifreq=2):

    psl = xr.open_zarr(f"{Emulator.local_store_path}/long-forecasts/graphufs.{t0}.{tf}.zarr")
    psl = psl.isel(time=slice(None, None, ifreq))

    for tname in ["ERA5"]:#, "Replay"]:

        truth = get_truth(tname)

        # Compute this
        psl["10m_wind_speed"] = calc_wind_speed(psl)
        truth["10m_wind_speed"] = calc_wind_speed(truth)

        # setup for each variable
        plot_options = {
            "tmp2m": {
                "cmap": "cmo.thermal",
                "vmin": -10,
                "vmax": 30,
            },
            #"spfh": {
            #    "cmap": "cmo.haline",
            #    "vmin": 2e-6,
            #    "vmax": 3e-6,
            #},
            "10m_wind_speed": {
                "cmap": "cmo.tempo",
                "vmin": 0,
                "vmax": 40,
            },
        }

        evars = {"tmp": "temperature", "spfh": "specific_humidity", "wind_speed": "wind_speed", "10m_wind_speed": "10m_wind_speed", "tmp2m": "2m_temperature"}

        for varname, options in plot_options.items():

            ds = xr.Dataset({
                "GraphUFS": psl[varname].sel(t0=t0).load(),
            })

            tvname = evars[varname] if truth.name == "ERA5" else varname
            ds[truth.name] = truth[tvname].sel(
                time=ds["GraphUFS"].time.values,
            ).load()

            # Convert to degC
            if varname[:3] == "tmp":
                for key in ds.data_vars:
                    ds[key] -= 273.15
                    ds[key].attrs["units"] = "degC"


            label = " ".join([x.capitalize() for x in evars[varname].split("_")])
            ds.attrs["label"] = f"{label} ({ds[truth.name].units})"

            dpi = 300
            pixelwidth = 10*dpi
            pixelheight = 6*dpi
            mov = xmovie.Movie(
                ds,
                movie_func,
                framedim="time",
                input_check=False,
                add_colorbar=False,
                pixelwidth=pixelwidth,
                pixelheight=pixelheight,
                dpi=dpi,
                **options
            )
            mov.save(
                f"{Emulator.local_store_path}/long-forecasts/figures/{truth.name.lower()}_vs_graphufs_{varname}_{t0}.mp4",
                progress=True,
                overwrite_existing=True,
            )
