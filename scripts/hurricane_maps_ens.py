import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, cast

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hurricane_maps import (
    add_derived_variables,
    plot_item_to_panel_spec,
    required_variables_from_config,
)
from hurricane_maps_config import HurricaneMapConfig

from common_utils import (
    RegionSpec,
    create_multi_panel_figure,
    extract_from_experiment_config,
    get_max_epoch_in_dir,
    prep_data,
)
from dataflows import load_dataset, select_datetime
from panels import plot_map_panel


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    return parser.parse_args()


CONFIG_FILE_NAME = "experiment.json"
EXPERIMENT_LIDT: tuple[Optional[type], tuple[str, ...]] = (
    datetime,
    ("last_input_datetime",),
)


START = "seed"
END = ".nc"
PERTURBATION_INFERENCE_SEED = 0
UNPERTURBED_INFERENCE_SEED = -1

PERTURBED_OUTPUT_PATH = f"{START}{PERTURBATION_INFERENCE_SEED}{END}"
UNPERTURBED_OUTPUT_PATH = f"{START}{UNPERTURBED_INFERENCE_SEED}{END}"

INPUT_START = "input_epoch-"
INPUT_END = ".nc"

CHUNKS = {
    "time": 1,
    "level": 13,
    "lat": 721,
    "lon": 1440,
}

GROUND_TRUTH_NAME = "HRES-fc0"
PERTURBED_NAME = "HRES-fc0 (perturbed)"

TRES = 6
INPUT_STEPS = [-1, 0]
STEPS = [1, 5, 10, 11, 12]

COLUMN_TITLES = {
    "t=-6h": -1,
    "t=0h": 0,
    "t=6h": 1,
    "t=1day6h": 5,
    "t=2day12h": 10,
    "t=2day18h": 11,
    "t=3day": 12,
}


def _to_hpa(x):
    return x / 100.0


def _region_for_dataset_lon_convention(
    ds: xarray.Dataset,
    region: RegionSpec,
) -> RegionSpec:
    minlat, minlon, maxlat, maxlon = region
    if minlon is None or maxlon is None or "lon" not in ds.coords:
        return region

    lon_min = float(ds["lon"].min().item())
    lon_max = float(ds["lon"].max().item())

    # Convert [-180, 180] region to [0, 360] if dataset longitudes are non-negative.
    if lon_min >= 0 and (minlon < 0 or maxlon < 0):
        minlon = (minlon + 360) % 360
        maxlon = (maxlon + 360) % 360
    # Convert [0, 360] region to [-180, 180] if dataset longitudes are centered on Greenwich.
    elif lon_max <= 180 and (minlon > 180 or maxlon > 180):
        minlon = ((minlon + 180) % 360) - 180
        maxlon = ((maxlon + 180) % 360) - 180

    return (minlat, minlon, maxlat, maxlon)


if __name__ == "__main__":

    row_names_to_ds: dict[str, xarray.Dataset] = {}

    config_path = parser().config
    config = HurricaneMapConfig.model_validate(json.load(open(config_path, "r")))
    root = config.folder

    root_experiment_name = "_".join(os.path.basename(root).split("_")[1:])
    print(f"Experiment name: {root_experiment_name}")

    # Extract revelant forecast timeline (experiment config)
    lidt: datetime = extract_from_experiment_config(root, EXPERIMENT_LIDT)
    ftdt = lidt + timedelta(hours=TRES)
    input_dts = [lidt + timedelta(hours=st * TRES) for st in INPUT_STEPS]
    forecast_dts = [lidt + timedelta(hours=st * TRES) for st in STEPS]

    # Extract variables
    assert config.region is not None
    variables, levels = required_variables_from_config(config)
    region = RegionSpec(config.region)
    transforms = {}
    if "mean_sea_level_pressure" in variables:
        transforms["mean_sea_level_pressure"] = _to_hpa

    # Load and prep ground truth
    assert config.ground_truth_path is not None
    ground_truth = load_dataset(
        config.ground_truth_path, first_target_datetime=ftdt, chunks=CHUNKS
    )
    ground_truth = select_datetime(ground_truth, input_dts + forecast_dts)
    ground_truth_region = _region_for_dataset_lon_convention(ground_truth, region)
    ground_truth = prep_data(
        ground_truth,
        variables=sorted(variables),
        levels=levels if len(levels) > 0 else None,
        region=ground_truth_region,
        transform=transforms if len(transforms) > 0 else None,
    )
    ground_truth = add_derived_variables(ground_truth, config)

    row_names_to_ds[GROUND_TRUTH_NAME] = ground_truth

    # Load optimized inputs
    optimized_dir = os.path.join(os.path.dirname(root), root_experiment_name)
    file_name, ep = get_max_epoch_in_dir(
        optimized_dir, startswith="input_epoch-", endswith=".nc"
    )
    optim_inputs = load_dataset(
        os.path.join(optimized_dir, file_name),
        first_target_datetime=ftdt,
        chunks=CHUNKS,
    )
    optim_inputs = select_datetime(optim_inputs, input_dts)
    optim_region = _region_for_dataset_lon_convention(optim_inputs, region)
    optim_inputs = prep_data(
        optim_inputs,
        variables=sorted(variables),
        levels=levels if len(levels) > 0 else None,
        region=optim_region,
        transform=transforms if len(transforms) > 0 else None,
    )
    optim_inputs = add_derived_variables(optim_inputs, config)

    # Load GraphCast and Optimized outputs:
    optimized_dir = os.path.join(os.path.dirname(root), root_experiment_name)
    graphcast = load_dataset(
        os.path.join(optimized_dir, "output_epoch-0.nc"),
        first_target_datetime=ftdt,
        chunks=CHUNKS,
    )
    graphcast = select_datetime(graphcast, forecast_dts)
    graphcast_region = _region_for_dataset_lon_convention(graphcast, region)
    graphcast = prep_data(
        graphcast,
        variables=sorted(variables),
        levels=levels if len(levels) > 0 else None,
        region=graphcast_region,
        transform=transforms if len(transforms) > 0 else None,
    )
    graphcast = add_derived_variables(graphcast, config)
    row_names_to_ds[f"GraphCast-AMSE"] = xarray.concat(
        [optim_inputs, graphcast],
        dim="time",
    )

    file_name, ep = get_max_epoch_in_dir(
        optimized_dir, startswith="output_epoch-", endswith=".nc"
    )
    graphcast_outputs = load_dataset(
        os.path.join(optimized_dir, file_name),
        first_target_datetime=ftdt,
        chunks=CHUNKS,
    )
    graphcast_outputs = select_datetime(graphcast_outputs, forecast_dts)
    graphcast_region = _region_for_dataset_lon_convention(graphcast_outputs, region)
    graphcast_outputs = prep_data(
        graphcast_outputs,
        variables=sorted(variables),
        levels=levels if len(levels) > 0 else None,
        region=graphcast_region,
        transform=transforms if len(transforms) > 0 else None,
    )
    graphcast_outputs = add_derived_variables(graphcast_outputs, config)
    row_names_to_ds[f"GraphCast-AMSE (epoch {ep})"] = xarray.concat(
        [optim_inputs, graphcast_outputs],
        dim="time",
    )

    # Load and prep model outputs
    for model_dir in os.listdir(root):
        if not os.path.isdir(os.path.join(root, model_dir)):
            continue

        # dir: outputs_model_name
        model_name = "_".join(model_dir.split("_")[1:])

        # Unperturbed
        model_name_unperturbed = f"{model_name}"
        model_output_path = os.path.join(root, model_dir, UNPERTURBED_OUTPUT_PATH)
        ds_unpert = load_dataset(
            model_output_path,
            first_target_datetime=ftdt,
            chunks=CHUNKS,
        )
        ds_unpert = select_datetime(ds_unpert, forecast_dts)
        unpert_region = _region_for_dataset_lon_convention(ds_unpert, region)
        ds_unpert = prep_data(
            ds_unpert,
            variables=sorted(variables),
            levels=levels if len(levels) > 0 else None,
            region=unpert_region,
            transform=transforms if len(transforms) > 0 else None,
        )
        ds_unpert = add_derived_variables(ds_unpert, config)
        row_names_to_ds[model_name_unperturbed] = xarray.concat(
            [ground_truth.isel(time=[0, 1], drop=False), ds_unpert],
            dim="time",
        )

        # Perturbed
        model_name_perturbed = f"{model_name} (perturbed)"
        model_output_path = os.path.join(root, model_dir, PERTURBED_OUTPUT_PATH)
        ds_pert = load_dataset(
            model_output_path, first_target_datetime=ftdt, chunks=CHUNKS
        )
        ds_pert = select_datetime(ds_pert, forecast_dts)
        pert_region = _region_for_dataset_lon_convention(ds_pert, region)
        ds_pert = prep_data(
            ds_pert,
            variables=sorted(variables),
            levels=levels if len(levels) > 0 else None,
            region=pert_region,
            transform=transforms if len(transforms) > 0 else None,
        )
        ds_pert = add_derived_variables(ds_pert, config)
        row_names_to_ds[model_name_perturbed] = xarray.concat(
            [optim_inputs, ds_pert],
            dim="time",
        )

    # Data all loaded and prepped:

    row_titles = list(row_names_to_ds.keys())
    column_titles = list(COLUMN_TITLES.keys())

    selected_times = INPUT_STEPS + STEPS
    time_idx_to_local_pos = {
        time_idx: pos for pos, time_idx in enumerate(selected_times)
    }

    fcontour = plot_item_to_panel_spec(config.fcontour)
    contour = plot_item_to_panel_spec(config.contour) if config.contour else None

    maps = []
    for _, dat in row_names_to_ds.items():
        for time_title, time_idx in COLUMN_TITLES.items():
            if time_idx not in time_idx_to_local_pos:
                raise ValueError(
                    f"Configured time index {time_idx} is not available in loaded rows. "
                    f"Available: {selected_times}."
                )
            local_t = time_idx_to_local_pos[time_idx]

            map_func = lambda ax, dat=dat, t=local_t: plot_map_panel(
                ax,
                dat.isel(time=t),
                fcontour=fcontour,
                contour=contour,
                arrows=None,
                region=None,
                title=None,
                land_color=config.land_color,
            )
            maps.append(map_func)

    scalar_mappable = None
    if "levels" in fcontour["specs"] and "cmap" in fcontour["specs"]:
        norm = BoundaryNorm(fcontour["specs"]["levels"], ncolors=256, clip=True)
        scalar_mappable = ScalarMappable(norm=norm, cmap=fcontour["specs"]["cmap"])
        scalar_mappable.set_array([])

    colormap = None
    if scalar_mappable is not None:
        colormap_position = cast(
            tuple[float, float, float, float],
            tuple(config.colormap_position),
        )
        colormap = {
            config.colormap_label: (scalar_mappable, colormap_position),
        }

    figsize = cast(tuple[int, int], tuple(config.figsize))

    create_multi_panel_figure(
        maps,
        nrows=len(row_titles),
        ncols=len(column_titles),
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        panel_labels={"row": row_titles, "col": column_titles},
        colormap=colormap,
    )

    if config.output_path is not None:
        save_path = os.path.join(config.output_path, config.output_file)
    else:
        save_path = os.path.join(config.folder, config.output_file)
    plt.savefig(save_path, bbox_inches="tight")
