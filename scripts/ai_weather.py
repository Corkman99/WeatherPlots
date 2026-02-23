from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PATHS: List[Dict[str, str]] = [
    {
        "Pangu6": f"/home/users/f/froelicm/scratch/model_comparisons/forecasts/pangu6/12step_2022{m:02d}22T12_shuffled.nc/",
        "GraphCast_Small": f"/home/users/f/froelicm/scratch/model_comparisons/forecasts/graphcast_small/12step_2022{m:02d}22T12_shuffled.nc",
        "Aurora": f"/home/users/f/froelicm/scratch/model_comparisons/forecasts/aurora/12step_2022{m:02d}22T12_shuffled.nc/",
    }
    for m in range(1, 2)
]

TRUE_PATH: str = "/home/users/f/froelicm/ai_cache/gc-pangu-aurora_20220822T12.nc"

VARIABLE = ("z", 500)
# VARIABLE = "t2m"
VARIABLE_UNIT = "hPa"
# VARIABLE_UNIT = "degC"

LOCATION: Tuple[Tuple[float, float], str] = ((8, 47), "Close to Zurich")

TITLE = f"Timeseries of {VARIABLE} over {LOCATION[1]}"

TIME_AXIS = [
    [
        np.datetime64(f"2022-{m:02d}-22T18:00:00") + np.timedelta64(i * 6, "h")
        for i in range(12)
    ]
    for m in range(1, 2)
]

OUT_PATH = f"/home/users/f/froelicm/WeatherPlots/AI_weather_models_2/gc-pangu-aurora_20220122_{VARIABLE}_shuffled.png"

COLORS = ["black", "blue", "red", "green"]


def load_timeseries(
    path: str,
    variable: Union[str, Tuple[str, int]],
    location: Tuple[float, float],
    # time_axis: List[np.datetime64],
):
    ds = xr.open_dataset(path, engine="netcdf4")
    ds = (
        ds[variable[0]].sel({"pressure_level": variable[1]})
        if isinstance(variable, tuple)
        else ds[variable]
    )

    # print(time_axis)
    # print(ds.datetime)

    ds = ds.sel({"latitude": location[0], "longitude": location[1]}, method="nearest")
    # ds = ds.sel({"datetime": time_axis})
    ds = ds.squeeze()

    ds_np = ds.values
    return ds_np


def load_ERA5(time, variable, location):
    era5 = xr.open_dataset(TRUE_PATH)
    era5 = (
        era5[variable[0]].sel({"pressure_level": variable[1]})
        if isinstance(variable, tuple)
        else era5[variable]
    )
    era5 = era5.isel({"valid_time": slice(-len(time), None)})
    era5 = era5.sel(
        {"latitude": location[0], "longitude": location[1]}, method="nearest"
    )
    era5 = era5.assign_coords({"valid_time": time})
    era5 = era5.squeeze()
    return era5.values


def add_to_subplot(ax, time_axis, ts, plot_kwargs: Dict[str, Any]):
    ax.plot(time_axis, ts[: len(time_axis)], **plot_kwargs)


if __name__ == "__main__":

    fig, ax = plt.subplots(1, 2, figsize=(25, 10), dpi=300)
    # ax = ax.flatten()
    for j, t in enumerate(TIME_AXIS):
        print(j)
        era5 = load_ERA5(t, VARIABLE, LOCATION[0])

        ts = {}
        for model, path in PATHS[j].items():
            ts[model] = load_timeseries(path, VARIABLE, LOCATION[0])  # , TIME_AXIS[j])

        # ax.set_title(TITLE)
        ax[j].set_ylabel(f"{VARIABLE_UNIT}")

        add_to_subplot(
            ax[j],
            TIME_AXIS[j],
            era5 - 273.25,
            {"label": "ERA5", "color": COLORS[0]} if j == 0 else {"color": COLORS[0]},
        )
        for i, (model, tep) in enumerate(ts.items()):
            add_to_subplot(
                ax[j],
                TIME_AXIS[j],
                tep - 273.25,
                (
                    {"label": model, "color": COLORS[i + 1], "alpha": 0.5}
                    if j == 0
                    else {"color": COLORS[i + 1], "alpha": 0.5}
                ),
            )

    fig.suptitle(TITLE)

    # fig.subplots_adjust(bottom=0.8)
    # cbar_ax = fig.add_axes((0.85, 0.15, 0.7, 0.05))
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(PATHS) + 1,
        frameon=False,
    )
    plt.savefig(OUT_PATH, bbox_inches="tight")
