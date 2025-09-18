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
from matplotlib.axes import Axes
import numpy as np

import os

HOME = os.environ["HOME"]

time_axis = [
    np.datetime64("2021-06-25T00:00:00") + np.timedelta64(6 * x, "h") for x in range(28)
]

# Dataset definitions:
optimized = merge_netcdf_files(
    os.path.join(
        HOME,
        "scratch/GraphCast-Mini_PNWH_12day/PNWH_28step_optim",
    ),
    pattern="out_ep-*.nc",
)

# Load ERA5
era5 = xr.open_dataset(
    os.path.join(
        HOME,
        "scratch/Data/GraphCast_small/graphcast_dataset_source-era5_date-2021-06-20_res-1.0_levels-13_steps-48.nc",
    )
)

# Dataset preparations:
region = (42, -130 + 360, 60, -110 + 360)
variables = {"geopotential": "z", "2m_temperature": "t2m"}
plevels = [1000]  # [500]
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
    x: xr.Dataset,
    time,
) -> xr.Dataset:
    return prep_data(
        x,
        variables,
        plevels,
        region,
        time,
        transform={
            "2m_temperature": to_degreeC,
            "geopotential": to_geopotentialheight,
        },
    )


optimized = wrapped_prep_data(optimized, (-8, None))
era5 = wrapped_prep_data(era5, (-8, None))
time_axis = time_axis[slice(-8, None)]


def add_timeseries(
    ax: Axes,
    arr: xr.DataArray,
    time: Optional[List[np.datetime64]] = None,
    **plot_kwargs,
) -> Axes:
    """Add a time series to a Matplotlib axis."""
    arr = arr.mean(dim=[d for d in arr.dims if d != "time"])
    arr = arr.squeeze()
    time_axis = arr.coords["time"].values if time is None else time
    ax.plot(time_axis, arr.data, **plot_kwargs)
    return ax


dats = [
    era5,
    optimized.sel(epoch=1),
    optimized.sel(epoch=3),
    optimized.sel(epoch=7),
]
colors = ["black", "#E6653E", "#E6A401", "#A2DE04"]  # "#DED504",
labels = ["ERA5", "GraphCast Small", "Optimized 2ep", "Optimized 8ep"]
linetypes = ["-", None, None, None, None]
alphas = [1, 0.8, 0.8, 0.8, 0.8]

plot_args = [
    {"color": color, "label": label, "linetype": linetype, "alpha": alpha}
    for color, label, linetype, alpha in zip(colors, labels, linetypes, alphas)
]

fig, axes = plt.subplots(
    figsize=(10, 2),
)
fig.suptitle("Mean over PNWH region")

axes.set_ylabel("2m Temperature (°C)")
axes.set_ylim(16, 30)
axes.grid()

for dat, args in zip(dats, plot_args):
    add_timeseries(
        axes,
        dat["t2m"],
        time_axis,
        color=args["color"],
        linestyle=args["linetype"],
        alpha=args["alpha"],
    )
    # add_timeseries(
    #    axes[1],
    #    dat["z"].sel(lat=50, lon=-121 + 360, drop=True),
    #    time_axis,
    #    color=args["color"],
    #    linestyle=args["linetype"],
    #    alpha=args["alpha"],
    # )


import matplotlib.lines as mlines

legend_handles = [
    mlines.Line2D(
        [], [], color=colors[0], label=labels[0], linewidth=3, linestyle=linetypes[0]
    ),
    mlines.Line2D(
        [], [], color=colors[1], label=labels[1], linewidth=3, linestyle=linetypes[1]
    ),
    mlines.Line2D(
        [], [], color=colors[2], label=labels[2], linewidth=3, linestyle=linetypes[2]
    ),
    mlines.Line2D(
        [], [], color=colors[3], label=labels[3], linewidth=3, linestyle=linetypes[3]
    ),
    # mlines.Line2D(
    #    [], [], color=colors[4], label=labels[4], linewidth=3, linestyle=linetypes[4]
    # ),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0),
    fontsize=10,
    frameon=False,
    ncols=5,
)

outpath = os.path.join(HOME, "WeatherPlots/PhD-day_plots/pnwh_optimized_ts.png")
fig.savefig(outpath, bbox_inches="tight")
