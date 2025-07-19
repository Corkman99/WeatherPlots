from panels import *
from common_utils import *
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np

save_path = "/WeatherPlots/HurricaneIan_14step/.png"

time_axis = [
    np.datetime64("2022-09-26T00:00:00") + np.timedelta64(6 * x, "h") for x in range(16)
]

# Dataset definitions:
# Load datasets from ~/GenCast/DATA/GraphCast_OP/Hurricane_Ian_GC-OP
pattern = "regional_ep-*.nc"
optimal_track_16 = merge_netcdf_files(
    "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/date-20250704_hurricane-optimal-1e-4",
    pattern=pattern,
)
optimal_track_14 = merge_netcdf_files(
    "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/date-20250710_hurricane-ian_optimal_1e-4_14step",
    pattern=pattern,
)

# Load HRES data
hres = xr.open_dataset(
    "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/custom-hres_2022-09-26_res-0.25_levels-13_steps-16.nc"
)

# Dataset preparations:
region = (20, -90 + 360, 30, -74 + 360)  # 14, 250, 35, 286
# region = (10, -125 + 360, 42, -50 + 360)
variables = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "mslp",
    "geopotential": "z500",
    # "temperature": "t500",
    # "u_component_of_wind": "u1000",
    # "v_component_of_wind": "v1000",
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
            # "temperature": to_degreeC,
            "2m_temperature": to_degreeC,
        },
    )
    prepped_with_wind = add_wind(prepped)
    return prepped_with_wind


optimal_track_16 = wrapped_prep_data(optimal_track_16).squeeze()
original_16 = optimal_track_16.sel(epoch=0, drop=True)
optimal_16 = optimal_track_16.sel(epoch=39, drop=True)

optimal_track_14 = wrapped_prep_data(optimal_track_14).squeeze()
original_14 = optimal_track_14.sel(epoch=0, drop=True)
optimal_14 = optimal_track_14.sel(epoch=49, drop=True)

hres = wrapped_prep_data(hres).squeeze()
hres = hres.assign_coords(time=optimal_track_16.time.values)

# Plotting items
rel = (optimal_track_16 - hres) / hres

# Define plotting specifications
title = "Hurricane Ian - HRES-fc0 Landfall at 2022-09-28 18z"

# Plotting specs:
cmap = "coolwarm"
extend = "max"
fcontour = {
    "variable": "t2m",
    "specs": {"cmap": cmap, "levels": np.arange(16, 36, 2), "extend": extend},
}
fcontour = {
    "variable": "uv10",
    "specs": {"cmap": cmap, "levels": np.arange(0, 50, 4), "extend": extend},
}

maps = []

maps = []
fcontours = [
    {
        "variable": "t2m",
        "specs": {
            "cmap": cmap,
            "levels": np.arange(-0.2, 0.2, 0.002),
            "extend": extend,
        },
    },
    {
        "variable": "uv10",
        "specs": {"cmap": cmap, "levels": np.arange(-1, 1, 0.05), "extend": extend},
    },
    {
        "variable": "mslp",
        "specs": {
            "cmap": cmap,
            "levels": np.arange(-0.02, 0.02, 0.0002),
            "extend": extend,
        },
    },
    {
        "variable": "z500",
        "specs": {
            "cmap": cmap,
            "levels": np.arange(-0.02, 0.02, 0.0002),
            "extend": extend,
        },
    },
]

for fc in fcontours:
    map = lambda ax, fc=fc: plot_map_panel(
        ax,
        rel.sel(epoch=39),
        fcontour=fc,
        region=region,
        title=None,
    )
    maps.append(map)

fig = create_multi_panel_figure(
    maps,
    nrows=2,
    ncols=2,
    figsize=(22, 12),
    subplot_kw={"projection": ccrs.PlateCarree()},
    # the colormap takes the same inputs as fcontour specs
    # and positioned at the bottom of the graph, centered
    colormap=None,
)

plt.tight_layout()
# plt.subplots_adjust(bottom=0.1)
plt.savefig(save_path)
