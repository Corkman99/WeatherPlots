import os
from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
from matplotlib.colors import BoundaryNorm


def variance(
    ds: List[xr.Dataset], dim: Tuple[str, ...] = ()
) -> Union[xr.DataArray, xr.Dataset]:
    ds_unified = xr.concat(ds, dim="dataset")
    return ds_unified.var(("dataset",) + dim)


def get_score_timeseries(
    ds: List[xr.Dataset],
) -> Union[xr.DataArray, xr.Dataset]:
    variances = variance(
        ds,
    )
    mean_variance_per_variable = variances.mean(
        dim=("latitude", "longitude", "pressure_level")
    )

    return mean_variance_per_variable


def prep_data(
    dirs: List[List[str]],
    variables: List[str],
    datetimes: Optional[List[List[np.datetime64]]] = None,
) -> List[xr.Dataset]:

    scores_per_start_time = []
    for i, paths in enumerate(dirs):
        dats = []
        for p in paths:
            dat = xr.open_dataset(
                p,
                engine="netcdf4",
                # chunks={
                #    "datetime": 12,
                #    "latitude": 721,
                #    "longitude": 721,
                #    "pressure_level": 13,
                # },
            ).squeeze()
            dat = dat[variables]
            if np.issubdtype(dat["datetime"].dtype, np.timedelta64):
                assert datetimes is not None
                dat = dat.assign_coords(datetime=("datetime", datetimes[i]))
            dats.append(dat)
        score = get_score_timeseries(dats)
        scores_per_start_time.append(score.compute())

    # List of length num_start_times
    # containing dataset of scores with dim forecast_time and
    # subsetted variables
    return scores_per_start_time


def plot_start_time(ds: xr.DataArray, ax, time_axis=None, **plot_kwargs):
    if time_axis is None:
        time_axis = ds.datetime
    if ds.ndim > 1:
        ds = ds.squeeze()
    if ds.ndim > 1:
        raise ValueError("DataArray is not 1-dimensional")

    if "label" not in plot_kwargs:
        plot_kwargs["label"] = ds.name
    ax.plot(time_axis, ds.values, **plot_kwargs)
    return ax


FORECAST_DIR = "/home/users/f/froelicm/scratch/model_comparisons/forecasts"
MODELS = ["pangu6", "aurora", "graphcast_small"]
START_TIMES = [f"2022{m:02d}22T12" for m in range(1, 2)]
PATHS = [
    [os.path.join(FORECAST_DIR, model, f"12step_{start}.nc") for model in MODELS]
    for start in START_TIMES
]
PATHS2 = [
    [
        os.path.join(FORECAST_DIR, model, f"12step_{start}_shuffled.nc")
        for model in MODELS
    ]
    for start in START_TIMES
]
VARIABLES = ["t2m", "msl", "u10", "v10", "z", "t", "u", "v"]
TIME_AXIS = [np.timedelta64(x * 6, "h") for x in range(1, 13)]
DATETIMES = [
    [np.datetime64(f"2022-{m:02d}-22T12:00:00") + x for x in TIME_AXIS]
    for m in range(1, 2)
]
COLORS = [matplotlib.colormaps.get_cmap("managua")(i / 11) for i in range(12)]


def make_plot():

    # with ProgressBar():
    data = prep_data(PATHS, VARIABLES, DATETIMES)
    data2 = prep_data(PATHS2, VARIABLES, DATETIMES)

    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(
        "Mean-variance per forecast-time across Pangu6, Aurora and GraphCast-Small forecasts"
    )
    axs = axs.flatten()
    for i, var in enumerate(VARIABLES):
        print(var)
        for t in range(len(START_TIMES)):
            plot_start_time(
                data[t][var], axs[i], time_axis=TIME_AXIS, label=None, color=COLORS[t]
            )
            plot_start_time(
                data2[t][var],
                axs[i],
                time_axis=TIME_AXIS,
                label=None,
                color=COLORS[t],
                linestyle="dashed",
            )
        axs[i].set_title(var)
        if i in [0, 4]:
            axs[i].set_ylabel("Unit^2")
        if i in [4, 5, 6, 7]:
            axs[i].set_xlabel("Forecast time")

    # colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
    cmap = matplotlib.colormaps.get_cmap("managua")
    bounds = BoundaryNorm(list(range(1, 12 + 2)), cmap.N)
    fig.colorbar(
        cm.ScalarMappable(
            cmap=cmap,
            norm=bounds,
        ),
        cax=cbar_ax,
        orientation="vertical",
        ticks=range(1, 12 + 1),
    )

    cbar_ax.set_title("Month")

    plt.savefig("AI_weather_models_2/variance_plot_comparison.png", bbox_inches="tight")


if __name__ == "__main__":
    make_plot()
