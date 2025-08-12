import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panels import plot_tropical_hurricane_track
from common_utils import *
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs

save_path = "MSC/MSC_tracks_v2.png"
home = os.path.expanduser("~")

# TRUTH:
truth = xr.open_dataset(
    os.path.join(
        home,
        "scratch/Data/GraphCast_OP/custom-hres_2022-09-26_res-0.25_levels-13_steps-16.nc",
    )
).isel(time=slice(4, None))

# GENCAST:
gencast = xr.open_dataset(
    os.path.join(home, "scratch/GenCast/HurricaneIan/run0-3.nc")
).squeeze()
gencast = gencast.rename({"sample": "batch"})

# GRAPHCAST TRACKS
pattern = "regional_ep-*.nc"
miami_track = merge_netcdf_files(
    os.path.join(
        home, "scratch/GraphCast-OP_TC_5day/date-20250704_hurricane-ian_target-miami"
    ),
    pattern=pattern,
)
tallahassee_track = merge_netcdf_files(
    os.path.join(
        home,
        "scratch/GraphCast-OP_TC_5day/date-20250704_hurricane-ian_target-tallahassee",
    ),
    pattern=pattern,
)

# Dataset preparations:
search_region = (0, -100 + 360, 40, -70 + 360)  # 14, 250, 35, 286
variables = {
    # "2m_temperature": "t2m",
    # "10m_u_component_of_wind": "u10",
    # "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "mslp",
    # "geopotential": "z",
    # "temperature": "t",
    # "total_precipitation_6hr": "tp",
}

plevels = None
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
    # vars[precip_name] = "tp"
    prepped = prep_data(
        x,
        vars,
        plevels,
        search_region,
        times,
        transform={
            "mean_sea_level_pressure": to_hPa,
            # "geopotential": to_geopotentialheight,
            # "temperature": to_degreeC,
            # "2m_temperature": to_degreeC,
        },
    )
    # prepped_with_wind = add_wind(prepped)
    return prepped  # prepped_with_wind


truth = wrapped_prep_data(truth).squeeze()
miami_track = wrapped_prep_data(miami_track).squeeze()
# graphcast_original = miami_track.isel(epoch=0, drop=True)
miami = miami_track.isel(epoch=-1, drop=True)

tallahassee = wrapped_prep_data(tallahassee_track).squeeze().isel(epoch=-1, drop=True)

gencast = wrapped_prep_data(gencast, precip_name="total_precipitation_12hr").squeeze()
num_gencast = gencast.sizes["batch"]

# Define plotting specifications
title = "Hurricane Ian - Landfall 2022-09-28 18z"

# Plotting items
maps = []
dats = tuple(
    [gencast.sel(batch=x) for x in range(num_gencast)] + [miami, tallahassee, truth]
)

configs = gencast_like_configs_color_variation(
    1, 2, num_gencast, [x.sizes["time"] for x in dats]
)

plot_region = (18, -85 + 360, 30, -75 + 360)  # 14, 250, 35, 286
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 8))
ax = plot_tropical_hurricane_track(
    ax,
    dats,
    search_region=search_region,
    plot_region=plot_region,
    title=title,
    **configs,
)


# Add legend
import matplotlib.lines as mlines

legend_handles = [
    mlines.Line2D([], [], color="black", label="Truth", linewidth=3),
    mlines.Line2D([], [], color="blue", label="GenCast", linewidth=3),
    mlines.Line2D([], [], color="red", label="Customized track", linewidth=3),
]
ax.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=12,
    frameon=False,
)

# Add Miami, Tallahassee, Freeport markers
annotations = [
    {
        "label": "Miami",
        "coords": (-80.166, 25.773),
        "color": "black",
        "marker": 8,
    },
    {
        "label": "Freeport",
        "coords": (-78.77, 26.52),
        "color": "black",
        "marker": 7,
    },
    {
        "label": "Tampa",
        "coords": (-82.507860, 27.908182),
        "color": "black",
        "marker": 8,
    },
]
for spec in annotations:
    ax.plot(
        spec["coords"][0],
        spec["coords"][1],
        marker=spec["marker"],
        color=spec["color"],
        markersize=10,
    )
    ax.text(
        spec["coords"][0],
        spec["coords"][1] + 0.1,
        spec["label"],
        color=spec["color"],
        fontsize=12,
    )

ax.plot()
plt.savefig(save_path)
