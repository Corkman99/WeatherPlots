import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from common_utils import *
from panels import *

save_path = "/WeatherPlots/HurricaneIan_14step/amse_vs_original.png"

time_axis = [
    np.datetime64("2022-09-26T00:00:00") + np.timedelta64(6 * x, "h") for x in range(16)
]

# Dataset definitions:
# Load datasets from ~/GenCast/DATA/GraphCast_OP/Hurricane_Ian_GC-OP
pattern = "regional_ep-*.nc"
original = merge_netcdf_files(
    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/optim_noprecip_2e-4_verylocal",
    pattern=pattern,
).sel(epoch=0, drop=True)
amse = merge_netcdf_files(
    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/AMSE/optim_noprecip_5e-4",
    pattern=pattern,
)

# Load HRES data
hres = xr.open_dataset(
    "/home/users/f/froelicm/scratch/Data/GraphCast_OP/custom-hres_2022-09-26_res-0.25_levels-13_steps-16.nc"
)

# Dataset preparations:
region = (20, -90 + 360, 30, -74 + 360)  # 14, 250, 35, 286
# region = (10, -125 + 360, 42, -50 + 360)
variables = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "mslp",
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
}
plevels = [500]
times = [-5]


def to_degreeC(x):
    return x - 273.25


def to_hPa(x):
    return x / 100.0


def to_geopotentialheight(x):
    return x / 9.80665


def add_wind(
    data: xr.Dataset, var1: str = "u10", var2: str = "v10", name: str = "uv10"
) -> xr.Dataset:
    data[name] = np.sqrt(data[var1] ** 2 + data[var2] ** 2)
    return data


def wrapped_prep_data(
    x: xr.Dataset, precip_name="total_precipitation_6hr"
) -> xr.Dataset:
    vars = variables.copy()
    vars[precip_name] = "tp"
    prepped = prep_data(
        x,
        vars,
        plevels,
        region,
        times,
        transform={
            "mean_sea_level_pressure": to_hPa,
            "geopotential": to_geopotentialheight,
            "temperature": to_degreeC,
            "2m_temperature": to_degreeC,
        },
    )
    prepped_with_wind = add_wind(prepped)
    prepped_with_wind = add_wind(prepped_with_wind, "u500", "v500", "uv500")
    return prepped_with_wind


original = wrapped_prep_data(original).squeeze()
amse_og = wrapped_prep_data(amse).sel(epoch=0, drop=True).squeeze()
amse = wrapped_prep_data(amse).sel(epoch=9, drop=True).squeeze()

hres = wrapped_prep_data(hres).squeeze()
hres = hres.assign_coords(time=original.time.values)

# Plotting items
vars = ["t2m", "uv10", "mslp", "z500", "t500", "uv500"]
mse_orig = ((original - hres) ** 2 / np.abs(hres))[vars]
mse_amse_orig = ((amse_og - hres) ** 2 / np.abs(hres))[vars]
mse_amse = ((amse - hres) ** 2 / np.abs(hres))[vars]

rel_1 = mse_orig - mse_amse
rel_2 = mse_amse_orig - mse_amse

# Define plotting specifications
title = "Relative improvement of AMSE vs MSE ckpt at 2022-09-28 18z (10step)"

# Plotting specs:
cmap = "coolwarm"
extend = "both"
levels = [-10, -5, -2, -1, -0.5, 0.5, 1, 2, 5, 10]
maps = []
fcontours = [
    {
        "variable": "t2m",
        "specs": {
            "cmap": cmap,
            "levels": levels,
            "extend": extend,
        },
    },
    {
        "variable": "uv10",
        "specs": {"cmap": cmap, "levels": levels, "extend": extend},
    },
    {
        "variable": "mslp",
        "specs": {
            "cmap": cmap,
            "levels": levels,
            "extend": extend,
        },
    },
    {
        "variable": "z500",
        "specs": {
            "cmap": cmap,
            "levels": levels,
            "extend": extend,
        },
    },
    {
        "variable": "t500",
        "specs": {
            "cmap": cmap,
            "levels": levels,
            "extend": extend,
        },
    },
    {
        "variable": "uv500",
        "specs": {
            "cmap": cmap,
            "levels": levels,
            "extend": extend,
        },
    },
]

for fc in fcontours:
    map = lambda ax, fc=fc, title=fc["variable"]: plot_map_panel(
        ax,
        rel_1,
        fcontour=fc,
        region=region,
        title=title,
    )
    maps.append(map)

fig = create_multi_panel_figure(
    maps,
    nrows=3,
    ncols=2,
    figsize=(12, 10),
    subplot_kw={"projection": ccrs.PlateCarree()},
    # the colormap takes the same inputs as fcontour specs
    # and positioned at the bottom of the graph, centered
    colormap=None,
)

plt.tight_layout()
HOME = os.environ["HOME"]
save_path = (
    HOME
    + "/WeatherPlots/HurricaneIan_GC-OP/AMSE/optim-amse_vs_original_relative-gain-hres.png"
)
plt.savefig(save_path)
