import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panels import *
from common_utils import *
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np

import os

HOME = os.environ["HOME"]

save_path = HOME + "/WeatherPlots/HurricaneIan_GC-OP/optimal_noprecip_leadtime.png"

time_axis = [
    np.datetime64("2022-09-26T00:00:00") + np.timedelta64(6 * x, "h") for x in range(16)
]

# Dataset definitions:
# Load datasets from ~/GenCast/DATA/GraphCast_OP/Hurricane_Ian_GC-OP
pattern = "regional_ep-*.nc"
optimal_track_14 = merge_netcdf_files(
    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/optim_noprecip_1e4_ep80",
    pattern=pattern,
)
optimal_track_8 = merge_netcdf_files(
    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/optim_noprecip_1e-4_8step",
    pattern=pattern,
)
optimal_track_4 = merge_netcdf_files(
    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/optim_noprecip_1e-4_4step",
    pattern=pattern,
)

# miami_track = merge_netcdf_files(
#    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/from_raw/miami_1e-5",
#    pattern=pattern,
# )
# tallahassee_track = merge_netcdf_files(
#    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/from_raw/tallahassee_1e-5",
#    pattern=pattern,
# )

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
    # "geopotential": "z500",
    # "temperature": "t500",
    # "u_component_of_wind": "u1000",
    # "v_component_of_wind": "v1000",
    "total_precipitation_6hr": "tp",
}
plevels = None  # [500]
times = [-4, -3, -2, -1]


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
            # "geopotential": to_geopotentialheight,
            # "temperature": to_degreeC,
            "2m_temperature": to_degreeC,
        },
    )
    prepped_with_wind = add_wind(prepped)
    return prepped_with_wind


optimal_track_14 = wrapped_prep_data(optimal_track_14).squeeze()
original_14 = optimal_track_14.sel(epoch=0, drop=True)
optimal_14 = optimal_track_14.sel(epoch=79, drop=True)
optimal_track_8 = wrapped_prep_data(optimal_track_8).squeeze()
original_8 = optimal_track_8.sel(epoch=0, drop=True)
optimal_8 = optimal_track_8.sel(epoch=9, drop=True)
optimal_track_4 = wrapped_prep_data(optimal_track_4).squeeze()
original_4 = optimal_track_4.sel(epoch=0, drop=True)
optimal_4 = optimal_track_4.sel(epoch=9, drop=True)

# miami_track = wrapped_prep_data(miami_track).squeeze().sel(epoch=19, drop=True)
# tallahassee_track = (
#    wrapped_prep_data(tallahassee_track).squeeze().sel(epoch=19, drop=True)
# )

hres = wrapped_prep_data(hres).squeeze()
hres = hres.assign_coords(time=original_4.time.values)

# Define plotting specifications
title = "Hurricane Ian - HRES-fc0 Landfall at 2022-09-28 18z"

# Plotting items
dats = [
    hres,
    original_14,
    optimal_14,
    original_8,
    optimal_8,
    original_4,
    optimal_4,
    # miami_track,
    # tallahassee_track,
]

# Plotting specs:
cmap = "coolwarm"
extend = "max"
fcontour = {
    "variable": "uv1000",
    "specs": {"cmap": cmap, "levels": np.arange(10, 30, 2), "extend": extend},
}
fcontour = {
    "variable": "uv10",
    "specs": {"cmap": cmap, "levels": np.arange(10, 50, 4), "extend": extend},
}
fcontour = {
    "variable": "t500",
    "specs": {"cmap": cmap, "levels": np.arange(-10, 5, 0.5), "extend": extend},
}
fcontour = {
    "variable": "t2m",
    "specs": {"cmap": cmap, "levels": np.arange(16, 36, 2), "extend": extend},
}
fcontour = {
    "variable": "uv10",
    "specs": {"cmap": cmap, "levels": np.arange(0, 30, 4), "extend": extend},
}
contour = {
    "variable": "z1000",
    "specs": {"colors": "grey", "levels": np.arange(80, 160, 20), "label": True},
}
contour = {
    "variable": "z500",
    "specs": {"colors": "grey", "levels": np.arange(5500, 6200, 100), "label": True},
}
contour = {
    "variable": "tp",
    "specs": {"colors": "black", "levels": np.arange(1e-6, 1e-2, 2e-3), "label": True},
}
contour = {
    "variable": "mslp",
    "specs": {"colors": "black", "levels": np.arange(930, 1050, 10), "label": True},
}
arrows = {
    "variable": ["u500", "v500"],
    "specs": {
        "color": "green",
        "scale": 300,
        "regrid_slice": (slice(None, None, 4), slice(None, None, 4)),
    },
}

maps = []
for dat in dats:
    for t in range(len(times)):
        map = lambda ax, dat=dat, t=t: plot_map_panel(
            ax,
            dat.isel(time=t),
            fcontour=fcontour,
            contour=contour,
            # arrows=arrows,
            region=region,
            title=None,
        )
        maps.append(map)

column_titles = [str(6 * (x + 5)) for x in times]
row_titles = [
    "HRES-fc0",
    "Original 14step",
    "Optimized 14step (ep80)",
    "Original 8step",
    "Optimized 8step (ep10)",
    "Original 4step",
    "Optimized 4step (ep10)",
    # "Miami Target (ep20)",
    # "Tallahassee (ep20)",
    # "Origianl 10step",
    # "Optimized 10step (ep50)",
]

"""
norm = BoundaryNorm(boundaries=levels, ncolors=len(levels) + 1, extend=extend)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # Required for colorbar
colorbar = (
    {
        "uv10": (
            sm,
            (0.3, 0.02, 0.4, 0.02),  # (x, y, width, height)
        )
    },
)"""

fig = create_multi_panel_figure(
    maps,
    nrows=len(dats),
    ncols=len(times),
    figsize=(12, 16),
    subplot_kw={"projection": ccrs.PlateCarree()},
    panel_labels={
        "row": row_titles,
        "col": column_titles,
    },
    # the colormap takes the same inputs as fcontour specs
    # and positioned at the bottom of the graph, centered
    colormap=None,
)

plt.tight_layout()
# plt.subplots_adjust(bottom=0.1)
plt.savefig(save_path)
