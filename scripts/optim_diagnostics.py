"""
Gives per-variable improvement or degradation of an optimized forecast against
original forecast.

MSE Skill Score: SS = 1 - MSE_opt / MSE_orig
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from panels import *
from common_utils import *

from functools import partial

pattern = "regional_ep-*.nc"
optimal_track_14 = merge_netcdf_files(
    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/optim_noprecip_1e4_ep80",
    pattern=pattern,
)

hres = xr.open_dataset(
    "/home/users/f/froelicm/scratch/Data/GraphCast_OP/custom-hres_2022-09-26_res-0.25_levels-13_steps-16.nc"
)

# Dataset preparations:
region = (20, -90 + 360, 30, -74 + 360)
variables = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "mslp",
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    # "total_precipitation_6hr": "tp",
    "vertical_velocity": "w",
    # "specifc_humidity": "q",
}
plevels = [500, 850, 1000]  # [500]
times = (None, None)


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
    prepped = add_wind(prepped)
    to_drop = ["u10", "v10"]
    for x in plevels:  # type: ignore
        prepped = add_wind(prepped, f"u{x}", f"v{x}", f"uv{x}")
        to_drop.append(f"u{x}")
        to_drop.append(f"v{x}")
    prepped = prepped.drop_vars(to_drop)
    return prepped


optimal_track_14 = wrapped_prep_data(optimal_track_14).squeeze()
original = optimal_track_14.sel(epoch=0, drop=True).squeeze()
optimal_20 = optimal_track_14.sel(epoch=19, drop=True).squeeze()
optimal_40 = optimal_track_14.sel(epoch=39, drop=True).squeeze()
optimal_80 = optimal_track_14.sel(epoch=79, drop=True).squeeze()

# miami_track = wrapped_prep_data(miami_track).squeeze().sel(epoch=19, drop=True)
# tallahassee_track = (
#    wrapped_prep_data(tallahassee_track).squeeze().sel(epoch=19, drop=True)
# )

hres = wrapped_prep_data(hres).squeeze()
start_time = hres.sizes["time"] - optimal_20.sizes["time"]
hres = hres.isel(time=slice(start_time, None))
hres = hres.assign_coords(time=original.time.values)


def mse(dat1, dat2):
    return ((dat1 - dat2) ** 2).mean()


def max_deviation(dat1, dat2):
    return np.abs(dat1 - dat2).max()


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


in_core_dims = ["lat", "lon"]
out_core_dims = []
func = partial(calculate_skill_score, score_func=mse)
skill_scores_20ep = xr.apply_ufunc(
    func,
    original,
    optimal_20,
    hres,
    input_core_dims=[in_core_dims, in_core_dims, in_core_dims],
    output_core_dims=[out_core_dims],
    vectorize=True,
)
skill_scores_40ep = xr.apply_ufunc(
    func,
    original,
    optimal_40,
    hres,
    input_core_dims=[in_core_dims, in_core_dims, in_core_dims],
    output_core_dims=[out_core_dims],
    vectorize=True,
)
skill_scores_80ep = xr.apply_ufunc(
    func,
    original,
    optimal_80,
    hres,
    input_core_dims=[in_core_dims, in_core_dims, in_core_dims],
    output_core_dims=[out_core_dims],
    vectorize=True,
)

var_groups = [
    ["t2m", "mslp", "uv10"],
    ["z500", "z850", "z1000"],
    ["t500", "t850", "t1000"],
    ["uv500", "uv850", "uv1000"],  # , "q500", "q850", "q1000"],
]
maps = []
for group in var_groups:
    maps.append(
        lambda ax, group=group: plot_variable_as_line(ax, skill_scores_20ep[group])
    )
    maps.append(
        lambda ax, group=group: plot_variable_as_line(ax, skill_scores_40ep[group])
    )
    maps.append(
        lambda ax, group=group: plot_variable_as_line(ax, skill_scores_80ep[group])
    )

fig = create_multi_panel_figure(
    maps,
    nrows=4,
    ncols=3,
    figsize=(14, 16),
    panel_labels={
        "col": ["20 Epochs", "40 Epochs", "80 Epochs"],
        "row": ["", "", "", ""],
    },
)

plt.tight_layout()
# plt.subplots_adjust(bottom=0.1)

save_path = "HurricaneIan_GC-OP/mse_skill_scores_optim-noprecip_20-80ep.png"
plt.savefig(save_path)
