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

time_axis = [
    np.datetime64("2022-09-26T00:00:00") + np.timedelta64(6 * x, "h") for x in range(16)
]

# Dataset definitions:
# Load datasets from ~/GenCast/DATA/GraphCast_OP/Hurricane_Ian_GC-OP
pattern = "regional_ep-*.nc"
amse = merge_netcdf_files(
    "/home/users/f/froelicm/scratch/AMSE/amse-with-amse-loss_steps-16",
    pattern=pattern,
)

"""gencast = (
    xr.open_dataset(
        os.path.join(
            HOME,
            "scratch/GenCast/HurricaneIan/init-22092025T12-23092025T00_end-29092025T00_10mem_A.nc",
        )
    )
    .squeeze()
    .sel(sample=8, drop=True)
    .isel(time=-7, drop=False)
)[["10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]]
gencast = gencast.expand_dims("time")
gencast = gencast.rename(
    {
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "mean_sea_level_pressure": "mslp",
    }
)"""

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
    "/home/users/f/froelicm/scratch/Data/GraphCast_OP/custom-hres_2022-09-23_res-0.25_levels-13_steps-24.nc"
).isel(time=slice(-16,None))

# Dataset preparations:
region = (20, -90 + 360, 30, -74 + 360)  # 14, 250, 35, 286
# region = (10, -125 + 360, 42, -50 + 360)
variables = {
    "2m_temperature": "t2m",
    #"10m_u_component_of_wind": "u10",
    #"10m_v_component_of_wind": "v10",
    #"mean_sea_level_pressure": "mslp",
    "geopotential": "z",
    # "temperature": "t",
    # "u_component_of_wind": "u1000",
    # "v_component_of_wind": "v1000",
    # "total_precipitation_6hr": "tp",
}
plevels =  [500]
times = [3,7,11,12,13,14,15]


def to_degreeC(x):
    return x - 273.25


def to_hPa(x):
    return x / 100.0


def to_geopotentialheight(x):
    return x / 9.80665


def add_wind(
    data: xr.Dataset, var1: str = "u10", var2: str = "v10", name: str = "uv10"
) -> xr.Dataset:
    data[name] = (data[var1] ** 2 + data[var2] ** 2) ** 0.5
    return data


def wrapped_prep_data(
    x: xr.Dataset, precip_name="total_precipitation_6hr"
) -> xr.Dataset:
    vars = variables.copy()
    # vars[precip_name] = "tp"
    prepped = prep_data(
        x,
        vars,
        plevels,
        region,
        times,
        transform={
            # "mean_sea_level_pressure": to_hPa,
            "geopotential": to_geopotentialheight,
            # "temperature": to_degreeC,
            "2m_temperature": to_degreeC,
        },
    )
    #prepped = add_wind(prepped)
    return prepped


amse = wrapped_prep_data(amse)

# original_amse = wrapped_prep_data(original_amse).sel(epoch=0, drop=True)

# optimized_amse = wrapped_prep_data(optimized_amse).squeeze()
# optimized_amse = optimized_amse.sel(epoch=0, drop=True)

hres = wrapped_prep_data(hres)
hres = hres.assign_coords(time=amse.time.values)

# gencast = add_wind(gencast, var1="u10", var2="v10", name="uv10")

# Define plotting specifications
title = "Hurricane Ian - HRES-fc0 Landfall at 2022-09-28 18z"

# Plotting items
dats = [
    hres,
    amse.sel(epoch=0),
    amse.sel(epoch=4),
    amse.sel(epoch=9),
    amse.sel(epoch=19),
    amse.sel(epoch=39),
]  # , original_amse]

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
    "variable": "uv10",
    "specs": {"cmap": cmap, "levels": np.arange(10, 30, 2), "extend": extend},
}
fcontour = {
    "variable": "t2m",
    "specs": {"cmap": cmap, "levels": np.arange(16, 36, 2), "extend": extend},
}
contour = {
    "variable": "z1000",
    "specs": {"colors": "grey", "levels": np.arange(80, 160, 20), "label": True},
}
contour = {
    "variable": "tp",
    "specs": {"colors": "black", "levels": np.arange(1e-6, 1e-2, 2e-3), "label": True},
}
contour = {
    "variable": "mslp",
    "specs": {"colors": "black", "levels": np.arange(930, 1050, 10), "label": True},
}
contour = {
    "variable": "z500",
    "specs": {"colors": "grey", "levels": np.arange(5500, 6200, 100), "label": True},
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
            land_color="#E3DFBF",
        )
        maps.append(map)

# column_titles = [str(6 * (x + 5)) for x in times]
row_titles = ["HRES-fc0", "GraphCast-AMSE", "5ep", "10ep", "20ep", "40ep"]
column_titles = ["-3days", "-2days", "-1day", "-18h", "-12h", "-6h", "landfall"]

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
    nrows=6,
    ncols=7,
    figsize=(16, 12),
    subplot_kw={
        "projection": ccrs.PlateCarree(),
    },
    panel_labels={
        "row": row_titles,
        "col": column_titles,
    },
    # the colormap takes the same inputs as fcontour specs
    # and positioned at the bottom of the graph, centered
    colormap=None,
)

plt.tight_layout()
save_path = HOME + "/WeatherPlots/HurricaneIan_GC-OP_new/amse_z500-t2m.png"
plt.savefig(save_path, bbox_inches="tight")
