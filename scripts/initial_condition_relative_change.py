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

save_path = HOME + "/WeatherPlots/HurricaneIan_GC-OP/IC_relative_per_variable.png"

time_axis = [
    np.datetime64("2022-09-26T00:00:00") + np.timedelta64(6 * x, "h") for x in range(16)
]

# Dataset definitions:
# Load datasets from ~/GenCast/DATA/GraphCast_OP/Hurricane_Ian_GC-OP
optimal_IC = merge_netcdf_files(
    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/optim_noprecip_1e4_ep80",
    pattern="optimized-inputs_ep-*.nc",
)
optimal_F = merge_netcdf_files(
    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/optim_noprecip_1e4_ep80",
    pattern="regional_ep-*.nc",
)

# Load HRES input data
hres = xr.open_dataset(
    "/home/users/f/froelicm/scratch/Data/GraphCast_OP/custom-hres_2022-09-26_res-0.25_levels-13_steps-16.nc"
)

# Dataset preparations:
region_global = (None, None, None, None)  # Global region
region_local = (20, -90 + 360, 30, -74 + 360)
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
plevels = [1000, 850, 500]
times_IC = (0, 2)
times_F = (-6, -3)


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
    x: xr.Dataset,
    region: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]],
    times: Tuple[Optional[int], Optional[int]],
) -> xr.Dataset:
    vars = variables.copy()
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
    for l in [10] + plevels:
        prepped = add_wind(prepped, f"u{l}", f"v{l}", f"uv{l}")
    return prepped


optimal_IC = wrapped_prep_data(
    optimal_IC, region=region_global, times=times_IC
).squeeze()
optimal_F = wrapped_prep_data(optimal_F, region=region_global, times=times_F).squeeze()
original_F = optimal_F.sel(epoch=0, drop=True)

hres_IC = wrapped_prep_data(hres, region=region_global, times=times_IC).squeeze()
hres_IC = hres_IC.assign_coords(time=optimal_IC["time"])
hres_F = wrapped_prep_data(hres, region=region_global, times=times_F).squeeze()
hres_F = hres_F.assign_coords(time=optimal_F.time.values)

# Prep dataset for initial condition difference
rel_diff_per_epoch = (optimal_IC - hres_IC) ** 2 / np.abs(hres_IC)


# Skill score
def calculate_skill_score(original, optimized, reference, score_func):
    """
    Calculate the Mean Squared Error (MSE) Skill Score between original and optimized forecasts.

    Parameters:
    original (xarray.DataArray): Original forecast data.
    optimized (xarray.DataArray): Optimized forecast data.

    Returns:
    float: xarray of MSE Skill Score per variable.
    """
    mse_orig = score_func(original, reference)
    mse_opt = score_func(optimized, reference)

    skill_score = 1 - (mse_opt / mse_orig)
    return skill_score


def mse(dat1, dat2):
    return ((dat1 - dat2) ** 2).mean()


from functools import partial

in_core_dims = ["lat", "lon"]
out_core_dims = []
func = partial(calculate_skill_score, score_func=mse)
skill_scores_local = xr.apply_ufunc(
    func,
    original_F.sel(
        lat=slice(region_local[0], region_local[2]),
        lon=slice(region_local[1], region_local[3]),
    ),
    optimal_F.sel(
        lat=slice(region_local[0], region_local[2]),
        lon=slice(region_local[1], region_local[3]),
    ),
    hres_F.sel(
        lat=slice(region_local[0], region_local[2]),
        lon=slice(region_local[1], region_local[3]),
    ),
    input_core_dims=[in_core_dims, in_core_dims, in_core_dims],
    output_core_dims=[out_core_dims],
    vectorize=True,
)

# one row per variable, one column per epoch
# plot map of rel_diff_per_epoch, title with skill_scores_global and skill_scores_local

variables = ["mslp"]  # ["t2m", "mslp", "uv10"]
epochs = [4]  # , 19, 49, 79]


region_wide = (10, -110 + 360, 40, -54 + 360)
maps = []
contours = [
    np.arange(0.01, 0.15, 0.01),
    np.arange(-0.1, 0.1, 0.01),
    np.arange(0, 5, 0.01),
]
for var in variables:
    for ep in epochs:
        rel_diff = rel_diff_per_epoch[[var]].sel(epoch=ep).isel(time=0).squeeze()
        skill_score = skill_scores_local[var].sel(epoch=ep).mean(dim="time").squeeze()
        t = f"{skill_score.data.item():.2f}"
        fcontour = {
            "variable": var,
            "specs": {
                "cmap": "Spectral_r",
                "levels": contours[variables.index(var)],
                "extend": "max",
            },
        }
        # Fix: Move all variables to default arguments
        map_func = lambda ax, ds=rel_diff, fc=fcontour, title=t, region=region_wide: plot_map_panel(
            ax, ds=ds, fcontour=fc, region=region, title=title
        )
        maps.append(map_func)


fig = create_multi_panel_figure(
    maps,
    nrows=len(variables),
    ncols=len(epochs),
    figsize=(16, 10),
    subplot_kw={"projection": ccrs.PlateCarree()},
    panel_labels={
        "row": [f"{var}" for var in variables],
        "col": [f"{ep} Epochs" for ep in epochs],
    },
)


save_path = HOME + "/WeatherPlots/HurricaneIan_GC-OP/IC_relative_per_variable_small.png"
plt.tight_layout()
plt.savefig(save_path)
