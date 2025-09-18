import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panels import plot_tropical_hurricane_track
from common_utils import *
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs

home = os.path.expanduser("~")

# TRUTH:
truth = (
    xr.open_dataset(
        os.path.join(
            home,
            "scratch/Data/GraphCast_OP/custom-hres_2022-09-23_res-0.25_levels-13_steps-24.nc",
        )
    )
    .squeeze()
    .isel(time=slice(3, None))
)

# GENCAST:
gencast = (
    xr.open_dataset(
        os.path.join(
            home,
            "scratch/GenCast/HurricaneIan/init-22092025T12-23092025T00_end-29092025T00_10mem_A.nc",
        )
    )
    .squeeze()
    .isel(time=slice(None, None))
)
gencast = gencast.rename({"sample": "batch"})

# GRAPHCAST TRACKS
pattern = "regional_ep-*.nc"
miami_track = merge_netcdf_files(
    os.path.join(
        home, "scratch/GraphCast-OP_TC_5day/date-20250704_hurricane-ian_target-miami"
    ),
    pattern=pattern,
).isel(time=slice(None, None))

miami_track2 = merge_netcdf_files(
    os.path.join(home, "scratch/GraphCast-OP_TC_6day/freeport-track_1e-3_12step"),
    pattern=pattern,
)

# tallahassee_track = merge_netcdf_files(
#    os.path.join(
#        home,
#        "scratch/GraphCast-OP_TC_6day/freeport-track_1e-3",
#    ),
#    pattern=pattern,
# ).isel(time=slice(1, None))
# tallahassee_track = xr.merge([truth.isel(time=[1, 2]), tallahassee_track])


# Dataset preparations:
search_region = (10, -92 + 360, 30.5, -65 + 360)  # 14, 250, 35, 286
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


gencast_input_timedeltas = [np.timedelta64(-12, "h"), np.timedelta64(0, "h")]
graphcast_input_timedeltas = [np.timedelta64(-6, "h"), np.timedelta64(0, "h")]

truth = wrapped_prep_data(truth).squeeze()
miami_track = wrapped_prep_data(miami_track).squeeze()
# graphcast_original = miami_track.isel(epoch=0, drop=True)
miami = miami_track.sel(epoch=9, drop=True)

miami_track2 = wrapped_prep_data(miami_track2).squeeze()
miami2 = miami_track2.sel(epoch=4, drop=True)

# tallahassee_track = wrapped_prep_data(tallahassee_track).squeeze()
# tallahassee = tallahassee_track.sel(epoch=9, drop=True)

miami = xr.merge(
    [
        truth.isel(time=[8, 9])
        .expand_dims(batch=1)
        .assign_coords(time=graphcast_input_timedeltas),
        miami,
    ]
)

miami2 = xr.merge(
    [
        truth.isel(time=[8, 9])
        .expand_dims(batch=1)
        .assign_coords(time=graphcast_input_timedeltas),
        miami2,
    ]
)

# tallahassee = wrapped_prep_data(tallahassee_track).squeeze().isel(epoch=-1, drop=True)

gencast = wrapped_prep_data(gencast, precip_name="total_precipitation_12hr").squeeze()
gencast = gencast.isel(batch=[i for i in gencast.batch.values if i not in [12, 15]])
num_gencast = gencast.sizes["batch"]

# mslp_mins = gencast["mslp"].isel(time=-1).min(dim=["lat", "lon"])

gencast = xr.merge(
    [
        truth.isel(time=[0, 2])
        .expand_dims(batch=num_gencast)
        .assign_coords(time=gencast_input_timedeltas),
        gencast,
    ]
)


# Define plotting specifications
title = "Hurricane Ian - Landfall 2022-09-28 18z"

# Plotting items
maps = []
dats = [gencast.isel(batch=x) for x in [13]] + [  # range(num_gencast)] + [
    miami,
    miami2,
    # tallahassee,
    truth,
]  # , truth]

configs = gencast_like_configs_color_variation(
    1, 2, num_gencast, [x.sizes["time"] for x in dats]
)

plot_region = (12, -90 + 360, 30, -72 + 360)  # 14, 250, 35, 286
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 8))
ax, centers = plot_tropical_hurricane_track(
    ax,
    dats,
    search_region=search_region,
    plot_region=plot_region,
    # title=title,
    **configs,
)


def identify_worst_track(centers):
    running_max = 0
    running_index = 0
    for i, data in enumerate(centers):
        last = data[-1]
        if running_max < last[1]:
            running_max = last[1]
            running_index = i

    return running_index


def identify_best_track(center1, center2):
    def _loss(center1, center2):
        assert len(center1) == len(
            center2
        ), f"Length mismatch: len(center1)={len(center1)}, len(center2)={len(center2)}"
        sum = 0
        for i in range(len(center1)):
            sum += np.sqrt(
                (center1[i][0] - center2[i][0]) ** 2
                + (center1[i][1] - center2[i][1]) ** 2
            )
        return sum

    return _loss(center1, center2)


"""center_truth = centers[-1][slice(3, None, 2)]
best_loss = np.inf
best_id = 0
for i in range(num_gencast):
    x = identify_best_track(center_truth, centers[i])
    if x < best_loss:
        best_loss = x
        best_id = i"""


# Add legend
import matplotlib.lines as mlines

legend_handles = [
    mlines.Line2D([], [], color="black", label="Truth", linewidth=3),
    mlines.Line2D([], [], color="blue", label="AI Ensemble Forecast", linewidth=3),
    mlines.Line2D([], [], color="red", label="Customized AI track", linewidth=3),
]
ax.legend(
    handles=legend_handles,
    loc="lower center",
    # bbox_to_anchor=(1, 0.5),
    fontsize=10,
    frameon=False,
    ncols=3,
)

# Add Miami, Tallahassee, Freeport markers
annotations = [
    {
        "label": "Miami",
        "coords": (-80.166, 25.773),
        "color": "black",
        "marker": 4,
        "offset": [+0.25, -0.175],
    },
    {
        "label": "Freeport",
        "coords": (-78.77, 26.52),
        "color": "black",
        "marker": 7,
        "offset": [-0.85, +0.3],
    },
    {
        "label": "Tampa",
        "coords": (-82.507860, 27.908182),
        "color": "black",
        "marker": 5,
        "offset": [-1.55, -0.17],
    },
]
for spec in annotations:
    ax.plot(
        spec["coords"][0],
        spec["coords"][1],
        marker=spec["marker"],
        color=spec["color"],
        markersize=6,
    )
    ax.text(
        spec["coords"][0] + spec["offset"][0],
        spec["coords"][1] + spec["offset"][1],
        spec["label"],
        color=spec["color"],
        fontsize=10,
    )


colors = ["blue"] * num_gencast + ["red", "black"]
sizes = [10] * num_gencast + [30, 30]  # gencast, optimized, truth
for i, data in enumerate(centers):
    ax.scatter(
        data[-1][0],
        data[-1][1],
        marker="o",
        s=sizes[i],  # Set marker size
        facecolors="none",  # Empty circle
        edgecolors=colors[i],
    )

save_path = "MSC/Ian_tracks_more.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
