from panels import plot_tropical_hurricane_track
from common_utils import *
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs

save_path = "plotting/Hurricane-Ian/tracks_14step_local.png"

time_axis = [
    np.datetime64("2022-09-26T00:00:00") + np.timedelta64(6 * x, "h") for x in range(16)
]

# Load datasets from ~/GenCast/DATA/GraphCast_OP/Hurricane_Ian_GC-OP
pattern = "regional_ep-*.nc"
optimal_track = merge_netcdf_files(
    "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/date-20250710_hurricane-ian_optimal_1e-4_14step",
    pattern=pattern,
)
optimal_finetune = merge_netcdf_files(
    "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/date-20250710_hurricane-ian_optimal_local_1e-4_14step",
    pattern=pattern,
)
"""
miami_track = merge_netcdf_files(
    "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/date-20250704_hurricane-ian_target-miami",
    pattern=pattern,
)
tallahassee_track = merge_netcdf_files(
    "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/date-20250704_hurricane-ian_target-tallahassee",
    pattern=pattern,
)
stronger_track = merge_netcdf_files(
    "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/date-20250704_hurricane-ian_wind-intensification",
    pattern=pattern,
)
"""

# Load HRES data
hres = xr.open_dataset(
    "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/custom-hres_2022-09-26_res-0.25_levels-13_steps-16.nc"
).isel(time=slice(4, None))

# Dataset preparations:
region = (14, -90 + 360, 35, -74 + 360)  # 14, 250, 35, 286
variables = {"mean_sea_level_pressure": "mslp"}


def wrapped_prep_data(x: xr.Dataset) -> xr.Dataset:
    return prep_data(
        x,
        variables,
        None,
        region,
        (None, None),
    )


optimal_track = wrapped_prep_data(optimal_track).squeeze()
optimal_track_ep50 = optimal_track.sel(epoch=49, drop=True)
optimal_finetune = wrapped_prep_data(optimal_finetune).squeeze()
optimal_finetune_ep50 = optimal_finetune.sel(epoch=49, drop=True)
# optimal_track_ep15 = optimal_track.sel(epoch=14, drop=True)
# miami_track = wrapped_prep_data(miami_track).squeeze()
# miami_track_ep10 = miami_track.sel(epoch=9, drop=True)
# tallahassee_track = wrapped_prep_data(tallahassee_track).squeeze()
# tallahassee_track_ep10 = tallahassee_track.sel(epoch=9, drop=True)
# stronger_track = wrapped_prep_data(stronger_track).squeeze()
# stronger_track_ep10 = stronger_track.sel(epoch=9, drop=True)
hres = wrapped_prep_data(hres).squeeze()

# Define plotting specifications
title = "Hurricane Ian - Landfall 2022-09-28 18z"

# Plotting items
maps = []
dats = [
    hres,
    # optimal_track_ep15,
    optimal_track_ep50,
    optimal_finetune_ep50,
    # miami_track_ep10,
    # tallahassee_track_ep10,
    # stronger_track_ep10,
]

label = [
    "HRES fc0",
    "Optimized (ep50)",
    "Local Optimized (ep50)",
    # "Target Miami",
    # "Target Tallahassee",
    # "Wind Intensified",
]

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8, 10))
ax = plot_tropical_hurricane_track(ax, dats, region=region, title=title, label=label)  # type: ignore
plt.savefig(save_path)
