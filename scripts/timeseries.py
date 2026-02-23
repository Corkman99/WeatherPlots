import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from common_utils import *
from panels import *

HOME = os.environ["HOME"]

# Dataset definitions:
optimized = xr.open_dataset(
    os.path.join(
        HOME,
        "scratch/output/graphcast_optimization_B_for_tim/perturbed-inputs-and-forecast-20210626T12_epoch-4.nc",
    )
)

neural_GCM = xr.open_dataset(
    os.path.join(
        HOME,
        "scratch/output/graphcast_neuralGCM_input/gc_neuralGCM-perturbed-inputs-and-forecast-20210626T12_epoch-0.nc",
    )
)

# Load ERA5
era5 = xr.open_dataset(
    os.path.join(
        HOME,
        "scratch/ERA5/20210626T12-20210630T12_025_37lvl_formatted.nc",
    )
).isel(time=slice(-18, None))

# Dataset preparations:
region = (42, -130 + 360, 60, -110 + 360)
variables = {"geopotential": "z", "temperature": "t"}
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
            "temperature": to_degreeC,
            "geopotential": to_geopotentialheight,
        },
    )


optimized = wrapped_prep_data(optimized, None)
neural_GCM = wrapped_prep_data(neural_GCM, None)
era5 = wrapped_prep_data(era5, None)


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


dats = [era5, optimized, neural_GCM]
colors = ["black", "#E6653E", "#E6A401", "#A2DE04"]  # "#DED504",
labels = ["ERA5", "GC (GC-optim inputs)", "GC (NGCM-optim inputs)"]
linetypes = ["-", None, None, None, None]
alphas = [1, 0.8, 0.8, 0.8, 0.8]

plot_args = [
    {"color": color, "label": label, "linetype": linetype, "alpha": alpha}
    for color, label, linetype, alpha in zip(colors, labels, linetypes, alphas)
]

fig, axes = plt.subplots(
    figsize=(10, 2),
)
fig.suptitle("Timeseries of Mean 1000hPa Temperature over PNWH region")

axes.set_ylabel("°C")
axes.set_ylim(24, 38)
axes.grid()

import datetime

for dat, args in zip(dats, plot_args):
    print(dat["t1000"].mean())
    len = dat.sizes["time"]
    add_timeseries(
        axes,
        dat["t1000"],
        [x for x in dats[1].coords["time"].values][-len:],
        color=args["color"],
        linestyle=args["linetype"],
        alpha=args["alpha"],
    )


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
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0),
    fontsize=10,
    frameon=False,
    ncols=5,
)

outpath = os.path.join(HOME, "WeatherPlots/outputs/pnwh_optimized_ts_b.png")
fig.savefig(outpath, bbox_inches="tight")
fig.savefig(outpath, bbox_inches="tight")
