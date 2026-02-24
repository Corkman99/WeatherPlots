import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_utils import (
    create_multi_panel_figure,
    from_aiwm2_to_graphcast,
    merge_netcdf_files,
    prep_data,
)
from panels import plot_map_panel
from scripts.hurricane_maps_config import HurricaneMapConfig

OUTPUT_PATTERN = "output_epoch-*.nc"
INPUT_PATTERN = "input_epoch-*.nc"
DEFAULT_CONFIG_PATH = "configs/hurricane_maps/default_uv10_mslp.json"
DERIVED_VARIABLES = {"10m_kinetic_energy", "kinetic_energy"}


@dataclass
class ExperimentTimeline:
    input_steps: int
    output_steps: int
    full_datetimes: np.ndarray
    start_stamp: Optional[np.datetime64]


def to_hpa(x):
    return x / 100.0


def resolve_plotted_name(variable: str, level: Optional[int]) -> str:
    if level is None:
        return variable
    return f"{variable}{level}"


def plot_item_to_panel_spec(item):
    return {
        "variable": resolve_plotted_name(item.variable, item.level),
        "specs": item.specs,
    }


def dependencies_for_variable(variable: str) -> set[str]:
    if variable in ["10m_wind_speed", "10m_kinetic_energy"]:
        return {"10m_u_component_of_wind", "10m_v_component_of_wind"}
    if variable in ["wind_speed", "kinetic_energy"]:
        return {"u_component_of_wind", "v_component_of_wind"}
    return {variable}


def required_variables_from_config(
    config: HurricaneMapConfig,
) -> tuple[set[str], list[int]]:
    variables = set()
    levels = []

    variables.update(dependencies_for_variable(config.fcontour.variable))

    if config.fcontour.level is not None:
        levels.append(config.fcontour.level)

    if config.contour is not None:
        variables.update(dependencies_for_variable(config.contour.variable))
        if config.contour.level is not None:
            levels.append(config.contour.level)

    if config.arrows is not None:
        variables.update(config.arrows.variable)

    return variables, sorted(set(levels))


def required_derived_plot_items(config: HurricaneMapConfig):
    items = []
    if config.fcontour.variable in DERIVED_VARIABLES:
        items.append(config.fcontour)
    if config.contour is not None and config.contour.variable in DERIVED_VARIABLES:
        items.append(config.contour)
    return items


def add_derived_variables(ds: xr.Dataset, config: HurricaneMapConfig) -> xr.Dataset:
    for item in required_derived_plot_items(config):
        if item.variable == "10m_wind_speed":
            ds["10m_wind_speed"] = (
                ds["10m_u_component_of_wind"] ** 2 + ds["10m_v_component_of_wind"] ** 2
            ) ** 0.5
        elif item.variable == "10m_kinetic_energy":
            ds["10m_kinetic_energy"] = (
                ds["10m_u_component_of_wind"] ** 2 + ds["10m_v_component_of_wind"] ** 2
            ) / 2.0
        elif item.variable == "wind_speed":
            speed_name = resolve_plotted_name("wind_speed", item.level)
            u_name = resolve_plotted_name("u_component_of_wind", item.level)
            v_name = resolve_plotted_name("v_component_of_wind", item.level)
            ds[speed_name] = (ds[u_name] ** 2 + ds[v_name] ** 2) ** 0.5
        elif item.variable == "kinetic_energy":
            ke_name = resolve_plotted_name("kinetic_energy", item.level)
            u_name = resolve_plotted_name("u_component_of_wind", item.level)
            v_name = resolve_plotted_name("v_component_of_wind", item.level)
            ds[ke_name] = (ds[u_name] ** 2 + ds[v_name] ** 2) / 2.0
    return ds


def selected_times_from_columns(config: HurricaneMapConfig) -> list[int]:
    return sorted(set(config.columns.values()))


def load_experiment_config(folder: str, filename: str) -> dict:
    path = Path(folder) / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment config file: {path}")
    return json.loads(path.read_text())


def infer_input_steps(folder: str) -> int:
    first_input = Path(folder) / "input_epoch-1.nc"
    if first_input.exists():
        ds = xr.open_dataset(first_input, chunks="auto")
        if "time" in ds.dims:
            return int(ds.sizes["time"])
    # Default to 2
    return 2


def build_experiment_timeline(
    folder: str,
    experiment_config_file: str,
    outputs_raw: xr.Dataset,
    input_steps: int,
) -> ExperimentTimeline:
    exp_config = load_experiment_config(folder, experiment_config_file)
    data_cfg = exp_config.get("data", {})

    start_stamp = None
    if (
        "first_target_datetime" in data_cfg
        and data_cfg["first_target_datetime"] is not None
    ):
        start_stamp = np.datetime64(data_cfg["first_target_datetime"])

    output_steps = int(outputs_raw.sizes.get("time", 0))

    output_datetimes = None
    if "datetime" in outputs_raw.coords:
        output_datetimes = np.asarray(outputs_raw.datetime.values)
    elif "time" in outputs_raw.coords and start_stamp is not None:
        output_time = np.asarray(outputs_raw.time.values)
        if np.issubdtype(output_time.dtype, np.timedelta64):
            output_datetimes = start_stamp + output_time

    if output_datetimes is None:
        output_datetimes = np.array([], dtype="datetime64[ns]")

    input_datetimes = np.array([], dtype="datetime64[ns]")
    first_input = Path(folder) / "input_epoch-1.nc"
    if first_input.exists():
        input_ds = xr.open_dataset(first_input, chunks="auto")
        if "datetime" in input_ds.coords:
            input_datetimes = np.asarray(input_ds.datetime.values)

    full_datetimes = np.concatenate([input_datetimes, output_datetimes])
    return ExperimentTimeline(
        input_steps=input_steps,
        output_steps=output_steps,
        full_datetimes=full_datetimes,
        start_stamp=start_stamp,
    )


def infer_time_axis(experiment_ds: xr.Dataset):
    if "datetime" in experiment_ds.coords:
        return experiment_ds.datetime.values
    if "time" in experiment_ds.coords:
        return experiment_ds.time.values
    raise ValueError(
        "Could not infer time axis: no 'time' or 'datetime' coordinate found."
    )


def to_iso_time_string(value) -> str:
    arr = np.asarray([value])
    if np.issubdtype(arr.dtype, np.datetime64):
        return str(np.datetime_as_string(np.datetime64(value), unit="s"))
    if np.issubdtype(arr.dtype, np.timedelta64):
        total_seconds = int(np.timedelta64(value, "s").astype(int))
        sign = "-" if total_seconds < 0 else ""
        total_seconds = abs(total_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{sign}PT{hours:02d}H{minutes:02d}M{seconds:02d}S"
    return str(value)


def align_dataset_time(ds: xr.Dataset, reference_time):
    if "time" in ds.coords and len(ds.time) == len(reference_time):
        ds = ds.assign_coords(time=reference_time)
    return ds


def select_by_time_axis(ds: xr.Dataset, time_axis):
    def _safe_select(dataset: xr.Dataset, dim_name: str) -> xr.Dataset:
        try:
            return dataset.sel({dim_name: time_axis})
        except Exception:
            pass

        try:
            return dataset.sel({dim_name: time_axis}, method="nearest")
        except Exception:
            pass

        if dim_name in dataset.dims:
            n = min(len(time_axis), dataset.sizes[dim_name])
            return dataset.isel({dim_name: slice(0, n)})
        return dataset

    if "datetime" in ds.coords and "time" in ds.dims:
        selected = _safe_select(ds.swap_dims({"time": "datetime"}), "datetime")
        return selected.swap_dims({"datetime": "time"})
    if "datetime" in ds.dims:
        selected = _safe_select(ds, "datetime")
        return selected.rename({"datetime": "time"})
    if "time" in ds.dims:
        return _safe_select(ds, "time")
    return ds


def ground_truth_datetime_values(ds: xr.Dataset) -> np.ndarray:
    if "datetime" in ds.coords:
        return np.asarray(ds["datetime"].values)
    if "datetime" in ds.dims:
        return np.asarray(ds["datetime"].values)
    if "time" in ds.coords and np.issubdtype(ds.time.dtype, np.datetime64):
        return np.asarray(ds.time.values)
    raise ValueError(
        "Ground truth dataset must contain a datetime-like 'datetime' or 'time' coordinate."
    )


def find_start_stamp_index_in_ground_truth(
    ground_truth_raw: xr.Dataset,
    start_stamp: np.datetime64,
) -> int:
    datetime_values = ground_truth_datetime_values(ground_truth_raw).astype(
        "datetime64[ns]"
    )
    stamp_ns = start_stamp.astype("datetime64[ns]")
    matches = np.where(datetime_values == stamp_ns)[0]
    if len(matches) == 0:
        raise ValueError(
            f"start_stamp {to_iso_time_string(stamp_ns)} not found in ground truth datetime coordinate."
        )
    return int(matches[0])


def requested_ground_truth_absolute_indices(
    requested_relative_times: list[int],
    start_stamp_index: int,
    input_steps: int,
    n_ground_truth_times: int,
) -> list[int]:
    absolute_indices = [
        start_stamp_index - input_steps + rel_idx
        for rel_idx in requested_relative_times
    ]
    out_of_bounds = [
        idx for idx in absolute_indices if idx < 0 or idx >= n_ground_truth_times
    ]
    if len(out_of_bounds) > 0:
        raise ValueError(
            "Requested relative times map to out-of-range ground truth indices: "
            f"{out_of_bounds}. start_stamp_index={start_stamp_index}, input_steps={input_steps}."
        )
    return absolute_indices


def prepare_dataset(
    ds: xr.Dataset,
    variables: set[str],
    levels: list[int],
    config: HurricaneMapConfig,
    time_indices: Optional[list[int]] = None,
) -> xr.Dataset:
    transforms = {}
    if "mean_sea_level_pressure" in variables:
        transforms["mean_sea_level_pressure"] = to_hpa

    region = None
    if config.region is not None:
        region = cast(tuple[float, float, float, float], tuple(config.region))

    prepared = prep_data(
        ds,
        variables=sorted(variables),
        levels=levels if len(levels) > 0 else None,
        region=region,
        time_range=time_indices if time_indices is not None else (None, None),
        transform=transforms if len(transforms) > 0 else None,
    )
    prepared = add_derived_variables(prepared, config)
    return prepared


def useful_hurricane_stats(da: xr.Dataset):
    if "mean_sea_level_pressure" in da:
        mslp = da["mean_sea_level_pressure"]
        mslp_stacked = mslp.stack(z=("lat", "lon"))
        min_idx = mslp_stacked.argmin().values
        center_lat = float(mslp_stacked.lat[min_idx].values)
        center_lon = float(mslp_stacked.lon[min_idx].values)
        print(f"Hurricane center: ({center_lat:.2f}, {center_lon:.2f})")
        print(f"Min MSLP: {mslp.min().values:.2f} hPa")


def needs_input_files(config: HurricaneMapConfig, input_steps: int) -> bool:
    """Check if any requested time indices are input timesteps."""
    requested_times = set(config.columns.values())
    return any(t < input_steps for t in requested_times)


def split_requested_times(
    config: HurricaneMapConfig,
    num_input_steps: int,
) -> tuple[list[int], list[int], list[int]]:
    requested_global = selected_times_from_columns(config)
    requested_input = [t for t in requested_global if t < num_input_steps]
    requested_output_global = [t for t in requested_global if t >= num_input_steps]
    requested_output_local = [t - num_input_steps for t in requested_output_global]
    return requested_global, requested_input, requested_output_local


def select_ground_truth_time_step(
    ground_truth: xr.Dataset,
    time_idx: int,
) -> xr.Dataset:
    try:
        return ground_truth.sel(time=time_idx)
    except Exception:
        if time_idx >= len(ground_truth.time):
            raise ValueError(
                f"Ground truth does not contain requested input time {time_idx}.",
                f"Available datetimes: {ground_truth.time.values}",
            )
        return ground_truth.isel(time=time_idx)


def build_epoch_dataset(
    epoch: int,
    outputs: xr.Dataset,
    inputs: Optional[xr.Dataset],
    ground_truth: Optional[xr.Dataset],
    requested_times: list[int],
    input_steps: int,
) -> xr.Dataset:
    """
    Build a dataset for a specific epoch by merging input/ground-truth and output data.

    Global time convention:
    - [0, input_steps-1] are input timesteps
    - input_steps is first output timestep (local output index 0)
    """
    epoch_output = outputs.sel(epoch=epoch)
    available_output_times = [int(t / 3600000000000) for t in epoch_output.time.values]
    print(f"INFO: Epoch {epoch}: output time steps available: {available_output_times}")
    pieces = []
    selected_global_times = []

    epoch_input = None
    if inputs is not None and "epoch" in inputs.coords and epoch in inputs.epoch.values:
        epoch_input = inputs.sel(epoch=epoch)

    for t_idx in requested_times:
        if t_idx < input_steps:
            if epoch == 0:
                if ground_truth is not None:
                    pieces.append(select_ground_truth_time_step(ground_truth, t_idx))
                    selected_global_times.append(t_idx)
                else:
                    raise ValueError(
                        "Input time indices requested for epoch 0 but no ground truth is loaded."
                    )
            else:
                if epoch_input is not None:
                    if t_idx >= len(epoch_input.time):
                        raise ValueError(
                            f"Inputs for epoch {epoch} do not contain requested input time {t_idx}."
                        )
                    pieces.append(epoch_input.isel(time=t_idx))
                    selected_global_times.append(t_idx)
                else:
                    raise ValueError(
                        f"Inputs required for epoch {epoch} at time {t_idx}"
                    )
        else:
            output_local_idx = t_idx - input_steps
            if output_local_idx >= len(epoch_output.time):
                raise ValueError(
                    f"Outputs for epoch {epoch} do not contain requested output time {t_idx} (local index {output_local_idx})."
                )
            pieces.append(epoch_output.isel(time=output_local_idx))
            selected_global_times.append(t_idx)

    if len(pieces) == 0:
        raise ValueError(
            f"No data assembled for epoch {epoch}. Check requested columns/time indices."
        )

    result = xr.concat(pieces, dim="time")
    result = result.assign_coords(time=selected_global_times)
    return result


def build_rows(
    outputs: xr.Dataset,
    inputs: Optional[xr.Dataset],
    ground_truth: Optional[xr.Dataset],
    config: HurricaneMapConfig,
    input_steps: int,
) -> dict[str, xr.Dataset]:
    """
    Build row datasets for each epoch, merging ground_truth/inputs/outputs as needed.
    """
    requested_times = sorted(set(config.columns.values()))

    rows = {}

    # Add ground truth as a separate row if requested and available
    if config.plot_ground_truth and ground_truth is not None:
        gt_pieces = [
            select_ground_truth_time_step(ground_truth, t) for t in requested_times
        ]
        if len(gt_pieces) > 0:
            gt_subset = xr.concat(gt_pieces, dim="time")
            gt_subset = gt_subset.assign_coords(time=requested_times)
            rows["Ground Truth"] = gt_subset
    elif config.plot_ground_truth and ground_truth is None:
        print("Warning: plot_ground_truth=True but no ground truth dataset is loaded.")

    # Build datasets for configured epochs
    available_epochs = (
        set(outputs.epoch.values.tolist()) if "epoch" in outputs.dims else set()
    )
    for epoch in config.epochs:
        if epoch not in available_epochs:
            print(f"Warning: Requested epoch {epoch} not present in outputs; skipping.")
            continue
        try:
            epoch_data = build_epoch_dataset(
                epoch,
                outputs,
                inputs,
                ground_truth,
                requested_times,
                input_steps,
            )
            rows[f"Epoch {epoch}"] = epoch_data
        except Exception as e:
            print(f"Warning: Could not build dataset for epoch {epoch}: {e}")

    if len(rows) == 0:
        raise ValueError(
            "No rows could be built. Check config.epochs, columns, and data availability."
        )

    return rows


def load_config(config_path: str) -> HurricaneMapConfig:
    return HurricaneMapConfig.model_validate_json(Path(config_path).read_text())


def main(config_path: str):
    config = load_config(config_path)

    exp_config = load_experiment_config(config.folder, config.experiment_config_file)

    variables, levels = required_variables_from_config(config)
    input_steps = infer_input_steps(config.folder)
    requested_global_times, requested_input_times, requested_output_local_times = (
        split_requested_times(config, input_steps)
    )
    requested_ground_truth_global_times = requested_global_times

    outputs_raw = merge_netcdf_files(config.folder, pattern=OUTPUT_PATTERN)
    timeline = build_experiment_timeline(
        config.folder,
        config.experiment_config_file,
        outputs_raw,
        input_steps,
    )
    experiment_time_axis = timeline.full_datetimes
    outputs = prepare_dataset(
        outputs_raw,
        variables,
        levels,
        config,
        time_indices=requested_output_local_times,
    )

    # Auto-detect if inputs are needed based on requested time indices
    inputs = None
    if needs_input_files(config, input_steps):
        inputs_raw = merge_netcdf_files(config.folder, pattern=INPUT_PATTERN)
        inputs = prepare_dataset(
            inputs_raw,
            variables,
            levels,
            config,
            time_indices=requested_input_times,
        )

    ground_truth = None
    if config.load_ground_truth:
        if config.ground_truth_path is None:
            raise ValueError(
                "load_ground_truth=True requires ground_truth_path in config."
            )
        ground_truth_raw = xr.open_dataset(config.ground_truth_path, chunks="auto")
        # Convert to GraphCast-format if coming from AIWM2
        if "valid_time" in ground_truth_raw.dims:
            data_cfg = exp_config.get("data", {})
            start_stamp = datetime.fromisoformat(data_cfg["first_target_datetime"])
            ground_truth_raw = from_aiwm2_to_graphcast(
                ground_truth_raw,
                start_stamp,
            )

        if timeline.start_stamp is not None:
            start_stamp_idx = find_start_stamp_index_in_ground_truth(
                ground_truth_raw,
                timeline.start_stamp,
            )
            requested_gt_absolute_indices = requested_ground_truth_absolute_indices(
                requested_global_times,
                start_stamp_idx,
                input_steps,
                len(ground_truth_datetime_values(ground_truth_raw)),
            )
            gt_datetime_values = ground_truth_datetime_values(ground_truth_raw)
            requested_axis = np.asarray(
                [gt_datetime_values[idx] for idx in requested_gt_absolute_indices]
            )
            ground_truth_raw = select_by_time_axis(ground_truth_raw, requested_axis)
            requested_ground_truth_global_times = requested_global_times
        elif len(experiment_time_axis) > 0:
            requested_ground_truth_global_times = [
                t for t in requested_global_times if t < len(experiment_time_axis)
            ]
            requested_axis = [
                experiment_time_axis[t] for t in requested_ground_truth_global_times
            ]
            if len(requested_axis) > 0:
                ground_truth_raw = select_by_time_axis(
                    ground_truth_raw,
                    np.asarray(requested_axis),
                )
        ground_truth = prepare_dataset(
            ground_truth_raw,
            variables,
            levels,
            config,
            time_indices=None,
        )
        if "time" in ground_truth.dims and len(ground_truth.time) == len(
            requested_ground_truth_global_times
        ):
            ground_truth = ground_truth.assign_coords(
                time=requested_ground_truth_global_times
            )

    # Print available time axis information
    num_times = len(experiment_time_axis)
    print(f"\n{'='*60}")
    print(f"Available time axis from experiment:")
    print(f"  Total time steps: {num_times} (indices 2 to {num_times+1})")
    preview_times = [
        to_iso_time_string(t)
        for t in experiment_time_axis[: min(5, len(experiment_time_axis))]
    ]
    print(
        f"  Time values: {preview_times}"
        + (
            f" ... (showing first 5 of {len(experiment_time_axis)})"
            if len(experiment_time_axis) > 5
            else ""
        )
    )
    print(
        f"  Time indices 0-{input_steps-1}: {'Input timesteps (ground truth for epoch 0)' if inputs is not None else 'Not loaded (outputs only)'}"
    )
    print(f"  Time indices {input_steps}+: Output timesteps (available for all epochs)")
    print(f"  Requested columns: {dict(config.columns)}")
    print(f"{'='*60}\n")

    rows = build_rows(outputs, inputs, ground_truth, config, input_steps)
    row_titles = list(rows.keys())
    column_titles = list(config.columns.keys())

    selected_times = requested_global_times
    time_idx_to_local_pos = {
        time_idx: pos for pos, time_idx in enumerate(selected_times)
    }

    fcontour = plot_item_to_panel_spec(config.fcontour)
    contour = (
        plot_item_to_panel_spec(config.contour) if config.contour is not None else None
    )

    maps = []
    # Rows are epochs, columns are time steps
    for row_label, dat in rows.items():
        for col_label, time_idx in config.columns.items():
            if time_idx not in time_idx_to_local_pos:
                raise ValueError(
                    f"Configured time index {time_idx} is not available after selection."
                )
            local_t = time_idx_to_local_pos[time_idx]
            region = None
            if config.region is not None:
                region = cast(tuple[float, float, float, float], tuple(config.region))

            map_func = lambda ax, dat=dat, t=local_t: plot_map_panel(
                ax,
                dat.isel(time=t),
                fcontour=fcontour,
                contour=contour,
                arrows=None,
                region=None,  # should be region, but handle lon / lat convention
                title=None,
                land_color=config.land_color,
            )
            useful_hurricane_stats(dat.isel(time=local_t))
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

    fig = create_multi_panel_figure(
        maps,
        nrows=len(row_titles),
        ncols=len(column_titles),
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        panel_labels={"row": row_titles, "col": column_titles},
        colormap=colormap,
    )

    save_path = os.path.join(config.folder, config.output_file)
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot hurricane maps from config.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON config file.",
    )
    args = parser.parse_args()
    main(args.config)
