import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Generator, Optional, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import hdf5plugin
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import xarray
from shapely.geometry import Point
from shapely.ops import unary_union

try:
    from dask.distributed import Client, LocalCluster
    from dask.distributed import as_completed as dask_as_completed
    from dask.distributed import progress
except ImportError:
    Client = None
    LocalCluster = None
    dask_as_completed = None
    progress = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_utils import get_latitude_weights, get_level_weights

MODEL_DIM = "model"
CHUNKS = {
    "valid_time": 7,  # divide num_days * 4
}
YEAR = 2022
MONTH = 2

# Conservative defaults to keep peak RAM bounded on shared/limited nodes.
# These can be overridden via environment variables.
DASK_N_WORKERS = 2
DASK_THREADS_PER_WORKER = 1
DASK_MEMORY_LIMIT_PER_WORKER = "8GB"
PROGRESS_EVERY = 10
EAGER_PER_INIT = False

PL_13 = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_seconds(duration_seconds: float) -> str:
    return f"{duration_seconds:.1f}s"


def is_dask_backed(ds: xarray.Dataset) -> bool:
    return any(hasattr(arr.data, "__dask_graph__") for arr in ds.data_vars.values())


def reduction_requested(mean_dim: tuple[str, ...], aliases: tuple[str, ...]) -> bool:
    return any(name in mean_dim for name in aliases)


def apply_mean_reductions(
    var: xarray.Dataset,
    mean_dim: tuple[str, ...],
    latitude_weighting: bool = True,
    level_weighting: bool = True,
) -> xarray.Dataset:
    if reduction_requested(mean_dim, ("latitude", "lat")) and "latitude" in var.dims:
        if latitude_weighting:
            lat_weights = get_latitude_weights(
                0.25, lat_dim="latitude", lon_dim="longitude"
            )
            if "latitude" in var.coords and "latitude" in lat_weights.dims:
                lat_weights = lat_weights.reindex({"latitude": var["latitude"]})
            lat_weight_sum = lat_weights.sum(dim="latitude")
            var = (var * lat_weights).sum(dim="latitude") / lat_weight_sum
        else:
            var = var.mean(dim="latitude")

    if reduction_requested(mean_dim, ("longitude", "lon")) and "longitude" in var.dims:
        var = var.mean(dim="longitude")

    if (
        reduction_requested(mean_dim, ("pressure_level", "level"))
        and "pressure_level" in var.dims
    ):
        if level_weighting:
            lvl_weights = get_level_weights(PL_13, dim="pressure_level")
            if "pressure_level" in var.coords and "pressure_level" in lvl_weights.dims:
                lvl_weights = lvl_weights.reindex(
                    {"pressure_level": var["pressure_level"]}
                )
            lvl_weight_sum = lvl_weights.sum(dim="pressure_level")
            var = (var * lvl_weights).sum(dim="pressure_level") / lvl_weight_sum
        else:
            var = var.mean(dim="pressure_level")

    if "valid_time" in mean_dim and "valid_time" in var.dims:
        var = var.mean(dim="valid_time")

    return var


def compute_mean_var(
    ds: xarray.Dataset,
    mean_dim: tuple[str, ...],
    var_dim: tuple[str, ...],
    latitude_weighting: bool = True,
    level_weighting: bool = True,
) -> xarray.Dataset:
    var = ds.var(dim=var_dim)
    return apply_mean_reductions(
        var,
        mean_dim,
        latitude_weighting=latitude_weighting,
        level_weighting=level_weighting,
    )


# Expects a generator of datasets which will be averaged over
# E.g. if the generator yields datasets with dimensions:
# (model, latitude, longitude, level, valid_time), we could call
# func(ds_gen, mean_dim("latitude", "longitude", "level"), var_dim=("model",))
# to obtain a dataset with dimensions (valid_time,)
ID = Any


def compute_mean_var_from_generator(
    ds: Generator[tuple[ID, xarray.Dataset], None, None],
    mean_dim: tuple[str, ...],
    var_dim: tuple[str, ...],
    latitude_weighting: bool = True,
    level_weighting: bool = True,
    total_items: Optional[int] = None,
    client: Optional[Any] = None,
    post_reduce_fn: Optional[Callable[[xarray.Dataset], xarray.Dataset]] = None,
) -> tuple[list[ID], xarray.Dataset]:
    """Process inits in parallel using a sliding-window of concurrent Dask futures.

    Up to MAX_CONCURRENT_INITS (default: DASK_N_WORKERS) inits are submitted at
    once. As each future completes, its per-init reduced result is stored and a
    new init is submitted, keeping workers busy while memory stays bounded.

    post_reduce_fn: optional transform applied to each per-init result after
        mean/variance reductions.  Use this to fold in masking or further
        spatial reductions so that only small arrays are accumulated in memory.
    """
    if client is None:
        raise RuntimeError("Distributed client is required in distributed-only mode")
    if dask_as_completed is None:
        raise RuntimeError("dask.distributed is not available in this environment")

    n_workers = int(os.environ.get("DASK_N_WORKERS", DASK_N_WORKERS))
    max_concurrent = int(os.environ.get("MAX_CONCURRENT_INITS", n_workers))
    progress_every = max(1, int(os.environ.get("PROGRESS_EVERY", PROGRESS_EVERY)))

    def build_reduced_var(ds_i: xarray.Dataset) -> xarray.Dataset:
        var = ds_i.var(dim=var_dim)
        result = apply_mean_reductions(
            var,
            mean_dim,
            latitude_weighting=latitude_weighting,
            level_weighting=level_weighting,
        )
        if post_reduce_fn is not None:
            result = post_reduce_fn(result)
        return result

    ds_iterator = iter(ds)
    future_to_id: dict = {}
    future_to_ds: dict = {}

    def try_submit_one() -> Optional[Any]:
        try:
            id_i, ds_i = next(ds_iterator)
            var = build_reduced_var(ds_i)
            f = client.compute(var)
            future_to_id[f] = id_i
            future_to_ds[f] = ds_i
            return f
        except StopIteration:
            return None

    # Fill the initial concurrency window
    initial_futures = []
    for _ in range(max_concurrent):
        f = try_submit_one()
        if f is None:
            break
        initial_futures.append(f)

    if not initial_futures:
        raise RuntimeError("No init groups found to process")

    init_results: dict[ID, xarray.Dataset] = {}
    count = 0
    phase_start = time.perf_counter()

    ac = dask_as_completed(initial_futures)
    for future in ac:
        id_i = future_to_id.pop(future)
        ds_i = future_to_ds.pop(future)
        result = future.result()
        ds_i.close()

        init_results[id_i] = result
        count += 1

        if count == 1 or count % progress_every == 0:
            elapsed = format_seconds(time.perf_counter() - phase_start)
            suffix = f"/{total_items}" if total_items is not None else ""
            log_progress(f"Prepared {count}{suffix} init datasets in {elapsed}")

        # Keep the window full — submit the next init immediately
        f = try_submit_one()
        if f is not None:
            ac.add(f)

    if not init_results:
        raise RuntimeError("No init results were produced")

    ids = sorted(init_results.keys())
    ds_by_init = xarray.concat(
        [
            init_results[id_i].expand_dims(
                init_time=xarray.DataArray([id_i], dims="init_time")
            )
            for id_i in ids
        ],
        dim="init_time",
    )

    log_progress(
        f"Finished processing {count} init datasets in "
        f"{format_seconds(time.perf_counter() - phase_start)}"
    )
    return ids, ds_by_init


def maybe_build_dask_client() -> Any:
    use_distributed = os.environ.get("USE_DASK_DISTRIBUTED", "1") == "1"
    if not use_distributed:
        raise RuntimeError("USE_DASK_DISTRIBUTED must be 1 (distributed-only mode)")
    if Client is None or LocalCluster is None:
        raise RuntimeError("dask.distributed is not available in this environment")

    n_workers = int(os.environ.get("DASK_N_WORKERS", DASK_N_WORKERS))
    threads_per_worker = int(
        os.environ.get("DASK_THREADS_PER_WORKER", DASK_THREADS_PER_WORKER)
    )
    memory_limit = os.environ.get(
        "DASK_MEMORY_LIMIT_PER_WORKER", DASK_MEMORY_LIMIT_PER_WORKER
    )
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=os.environ.get("DASK_DASHBOARD_ADDRESS", ":8787"),
        processes=True,
    )
    client = Client(cluster)
    return client


def compute_with_progress(obj: Any, client: Any) -> Any:
    log_progress("Starting compute")
    compute_start = time.perf_counter()

    if progress is None:
        raise RuntimeError("distributed progress reporter is unavailable")
    future = client.compute(obj)
    progress(future)
    result = future.result()

    log_progress(
        f"Finished compute in {format_seconds(time.perf_counter() - compute_start)}"
    )
    return result


# Expected filename: Cast_graphcast_amse_2022_01_05_06.nc
# Fields: {module}_{model_name}_{year}_{month}_{day}_{hour}.nc
# Corresponding id: (graphcast_amse, "2022-01-05T06:00:00")
DirPath = str
ModelName = str
InitDT = datetime


def generate_id_from_root(
    root: str, max_files: Optional[int] = 1000
) -> Generator[tuple[DirPath, ModelName, InitDT], None, None]:
    count = 0
    for dirpath in os.listdir(root):
        if dirpath.endswith(".nc"):
            dirpath_cut = dirpath[:-3]  # Remove .nc extension
            parts = dirpath_cut.split("_")
            if len(parts) == 6:
                model_name = parts[1]
                dt = datetime.fromisoformat(
                    f"{parts[2]}-{parts[3]}-{parts[4]}T{parts[5]}:00:00"
                )
            elif len(parts) == 7:
                model_name = f"{parts[1]}_{parts[2]}"
                dt = datetime.fromisoformat(
                    f"{parts[3]}-{parts[4]}-{parts[5]}T{parts[6]}:00:00"
                )
            else:
                raise ValueError(
                    f"Unexpected filename format: {dirpath}. Expected 6 or 7 parts separated by underscores."
                )
            dirpath = os.path.join(root, dirpath)
            yield dirpath, model_name, dt

            count += 1
            if max_files is not None and count >= max_files:
                break


def group_datasets_by_init_dt(
    ds_gen: Generator[tuple[DirPath, ModelName, InitDT], None, None],
) -> dict[InitDT, list[tuple[DirPath, ModelName]]]:
    inits: dict[InitDT, list[tuple[DirPath, ModelName]]] = {}
    for file_path, model_name, dt in ds_gen:
        if dt in inits:
            inits[dt].append((file_path, model_name))
        else:
            inits[dt] = [(file_path, model_name)]
    return inits


def subset_variables(
    ds: xarray.Dataset, variable: Optional[str], level: Optional[int] = None
) -> xarray.Dataset:
    variable_list = [variable]

    if variable is None:
        return ds

    KE_10M = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
    KE = ["u_component_of_wind", "v_component_of_wind"]

    if variable == "10m_kinetic_energy":
        variable_list = KE_10M
    if variable == "kinetic_energy":
        variable_list = KE

    ds = ds[variable_list]
    if level is not None and "pressure_level" in ds.dims:
        ds = ds.sel(pressure_level=level, drop=True)

    if variable == "kinetic_energy":
        # Compute kinetic energy from u and v components
        u = ds[KE[0]]
        v = ds[KE[1]]
        ke = 0.5 * (u**2 + v**2)
        ke.name = variable
        ds[variable] = ke
        ds = ds.drop(KE)

    if variable == "10m_kinetic_energy":
        # Compute kinetic energy from 10m u and v components
        u = ds[KE_10M[0]]
        v = ds[KE_10M[1]]
        ke = 0.5 * (u**2 + v**2)
        ke.name = variable
        ds[variable] = ke
        ds = ds.drop(KE_10M)

    return ds


def normalize_dataset_coords(ds: xarray.Dataset) -> xarray.Dataset:
    if "lat" in ds.dims:
        log_progress("Dataset uses 'lat' dimension. Renaming to 'latitude'.")
        ds = ds.rename({"lat": "latitude"})

    lat0 = float(ds["latitude"].values[0])
    lat1 = float(ds["latitude"].values[1])
    if lat1 < lat0:
        ds = ds.isel({"latitude": slice(None, None, -1)})

    if "lon" in ds.dims:
        log_progress("Dataset uses 'lon' dimension. Renaming to 'longitude'.")
        ds = ds.rename({"lon": "longitude"})

    lon0 = float(ds["longitude"].values[0])
    lon1 = float(ds["longitude"].values[1])
    if lon1 < lon0:
        ds = ds.isel({"longitude": slice(None, None, -1)})

    return ds


def reindex_valid_time(ds: xarray.Dataset, init_dt: datetime) -> xarray.Dataset:
    # Valid-time transform to timedeltas indicating forecast lead time
    forecast_time = ds["valid_time"].values - numpy.datetime64(init_dt)
    assert forecast_time.dtype == numpy.dtype("timedelta64[ns]")
    ds = ds.assign_coords(valid_time=forecast_time)
    return ds


def build_preprocess(
    init_dt: datetime,
    variable: Optional[str],
    level: Optional[int],
    fixed_valid_time: Optional[np.timedelta64] = None,
) -> Callable[[xarray.Dataset], xarray.Dataset]:
    def _preprocess(ds: xarray.Dataset) -> xarray.Dataset:
        ds = normalize_dataset_coords(ds)
        ds = reindex_valid_time(ds, init_dt)
        ds = subset_variables(ds, variable, level)
        if fixed_valid_time is not None:
            ds = ds.sel(valid_time=fixed_valid_time)  # expect exact match
        return ds

    return _preprocess


def generate_mfdatasets_by_model(
    file_paths_by_init: dict[InitDT, list[tuple[DirPath, ModelName]]],
    variable: str,
    level: Optional[int] = None,
    chunks: Optional[dict[str, int]] | str = None,
    preprocess_factory: Optional[
        Callable[
            [InitDT, str, Optional[int]], Callable[[xarray.Dataset], xarray.Dataset]
        ]
    ] = None,
) -> Generator[tuple[InitDT, xarray.Dataset], None, None]:
    total_inits = len(file_paths_by_init)
    progress_every = max(1, int(os.environ.get("PROGRESS_EVERY", PROGRESS_EVERY)))
    for index, (dt, list_of_files) in enumerate(file_paths_by_init.items(), start=1):
        # Sort by model_name to ensure consistent ordering across inits
        list_of_files.sort(key=lambda x: x[1])
        model_dim = xarray.DataArray(
            list(x[1] for x in list_of_files),
            dims=MODEL_DIM,
            name=MODEL_DIM,
        )
        files = [x[0] for x in list_of_files]
        preprocess_fn = (
            preprocess_factory(dt, variable, level)
            if preprocess_factory is not None
            else build_preprocess(dt, variable, level)
        )
        ds = xarray.open_mfdataset(
            files,
            chunks=chunks,
            concat_dim=model_dim,
            combine="nested",
            engine="h5netcdf",
            preprocess=preprocess_fn,
            join="exact",
        )
        if index == 1 or index % progress_every == 0 or index == total_inits:
            log_progress(
                f"Opened {index}/{total_inits} init groups; current init={dt.isoformat()}"
            )
            log_progress(f"Size: {ds.nbytes / 1e9:.2f} GB.")
        yield dt, ds


def get_plot_data(
    root: str,
    data_save_path: str,
    variable: str,
    level: Optional[int],
    year: int = YEAR,
    month: Optional[int] = None,
) -> xarray.Dataset:

    return _get_plot_data_impl(
        root, data_save_path, variable, level, year, month, preprocess_factory=None
    )


def get_plot_data_fixed_valid_time(
    root: str,
    data_save_path: str,
    variable: str,
    valid_time: Union[np.timedelta64, timedelta, pd.Timedelta, str],
    level: Optional[int] = None,
    year: int = YEAR,
    month: Optional[int] = None,
) -> xarray.Dataset:
    """Like get_plot_data but selects a fixed lead time per-init before the spatial
    reduction, returning a dataset with only (init_time,) as dimension."""
    target_valid_time = _as_timedelta64ns(valid_time)

    def _fixed_lead_preprocess_factory(
        init_dt: InitDT,
        variable_i: str,
        level_i: Optional[int],
    ) -> Callable[[xarray.Dataset], xarray.Dataset]:
        # For this task, valid_time is selected in preprocess after reindexing.
        return build_preprocess(
            init_dt,
            variable_i,
            level_i,
            fixed_valid_time=target_valid_time,
        )

    return _get_plot_data_impl(
        root,
        data_save_path,
        variable,
        level,
        year,
        month,
        preprocess_factory=_fixed_lead_preprocess_factory,
    )


def get_plot_data_fixed_valid_time_init_averaged_spatial(
    root: str,
    data_save_path: str,
    variable: str,
    valid_time: Union[np.timedelta64, timedelta, pd.Timedelta, str],
    level: Optional[int] = None,
    year: int = YEAR,
    month: Optional[int] = None,
) -> xarray.DataArray:
    """Select a fixed lead time, keep spatial dimensions, and average over init_time.

    This computes model variance at each init, preserving spatial dimensions
    (e.g. latitude/longitude and optional pressure_level), then averages across
    init_time.
    """
    if os.path.exists(data_save_path):
        plot_data = xarray.open_dataset(data_save_path)
        log_progress(f"Reusing existing NetCDF {data_save_path}")

    else:
        target_valid_time = _as_timedelta64ns(valid_time)

        def _fixed_lead_preprocess_factory(
            init_dt: InitDT,
            variable_i: str,
            level_i: Optional[int],
        ) -> Callable[[xarray.Dataset], xarray.Dataset]:
            return build_preprocess(
                init_dt,
                variable_i,
                level_i,
                fixed_valid_time=target_valid_time,
            )

        plot_data = _get_plot_data_impl(
            root,
            data_save_path,
            variable,
            level,
            year,
            month,
            preprocess_factory=_fixed_lead_preprocess_factory,
            mean_dim=(),
            average_init_time=True,
        )

    assert variable in plot_data, f"Variable '{variable}' not found in dataset"

    da = plot_data[variable]
    if level is not None and "pressure_level" in da.dims:
        da = da.sel(pressure_level=level, method="nearest")
    da = da.squeeze(drop=True)

    expected_dims = {"latitude", "longitude"}
    actual_dims = set(da.dims)
    assert (
        actual_dims == expected_dims
    ), f"Expected dims {expected_dims} for spatial plotting, got {actual_dims}"

    return da


def _get_plot_data_impl(
    root: str,
    data_save_path: str,
    variable: str,
    level: Optional[int],
    year: int,
    month: Optional[int],
    preprocess_factory: Optional[
        Callable[
            [InitDT, str, Optional[int]], Callable[[xarray.Dataset], xarray.Dataset]
        ]
    ],
    mean_dim: tuple[str, ...] = ("latitude", "longitude", "pressure_level"),
    average_init_time: bool = False,
) -> xarray.Dataset:
    if os.path.exists(data_save_path):
        log_progress(f"Reusing existing NetCDF {data_save_path}")
        return xarray.open_dataset(data_save_path)

    try:
        client = maybe_build_dask_client()
        log_progress(f"Dask dashboard: {client.dashboard_link}")

        log_progress(f"Scanning files under {root}")
        id_gen = generate_id_from_root(root, max_files=None)
        dt_to_file_path_model = group_datasets_by_init_dt(id_gen)

        dt_to_file_path_model = {
            dt: file_paths
            for dt, file_paths in dt_to_file_path_model.items()
            if dt.year == year and (month is None or dt.month == month)
        }
        log_progress(f"Selected {len(dt_to_file_path_model)} init groups for {year}")

        ds_gen = generate_mfdatasets_by_model(
            dt_to_file_path_model,
            variable,
            level,
            chunks=CHUNKS,
            preprocess_factory=preprocess_factory,
        )

        _, ds_result = compute_mean_var_from_generator(
            ds_gen,
            mean_dim=mean_dim,
            var_dim=(MODEL_DIM,),
            latitude_weighting=True,
            level_weighting=True,
            total_items=len(dt_to_file_path_model),
            client=client,
        )

        if average_init_time and "init_time" in ds_result.dims:
            ds_result = ds_result.mean(dim="init_time")

        if is_dask_backed(ds_result):
            ds_result = compute_with_progress(ds_result, client)

        log_progress(f"Writing NetCDF to {data_save_path}")
        log_progress(
            f"Dimensions: {ds_result.dims}, size: {ds_result.nbytes / 1e9:.2f} GB"
        )
        write_start = time.perf_counter()
        ds_result.to_netcdf(data_save_path)
        log_progress(
            f"Finished writing NetCDF in {format_seconds(time.perf_counter() - write_start)}"
        )

        client.close()
        return ds_result

    finally:
        pass


def np_array_to_timedelta_list(np_array: np.ndarray) -> list[timedelta]:
    """Converts a 0d or 1d numpy array of timedelta64 objects to a list of timedelta objects."""
    assert np_array.dtype.kind == "m", "Input array must be of timedelta64 dtype."
    assert np_array.ndim <= 1, "Input array must be 1-dimensional."
    if np_array.ndim == 0:
        np_array = np_array.reshape((1,))
    return [pd.Timedelta(x).to_pytimedelta() for x in np_array]


def _as_timedelta64ns(
    valid_time: Union[np.timedelta64, timedelta, pd.Timedelta, str],
) -> np.timedelta64:
    if isinstance(valid_time, np.timedelta64):
        return valid_time.astype("timedelta64[ns]")
    if isinstance(valid_time, timedelta):
        return np.timedelta64(pd.Timedelta(valid_time).value, "ns")
    if isinstance(valid_time, pd.Timedelta):
        return np.timedelta64(valid_time.value, "ns")
    if isinstance(valid_time, str):
        td = pd.to_timedelta(valid_time)
        return np.timedelta64(td.value, "ns")
    raise TypeError(
        "valid_time must be one of: np.timedelta64, datetime.timedelta, "
        "pandas.Timedelta, or a pandas-parseable timedelta string"
    )


def build_mean_var_vs_timestep_data(
    ds_mean_var: xarray.Dataset,
    variable: str,
    average_init_time: bool = True,
) -> xarray.DataArray:
    plot_data = ds_mean_var[variable]
    if average_init_time and "init_time" in plot_data.dims:
        plot_data = plot_data.mean(dim="init_time")
    return plot_data.squeeze()


def select_mean_var_for_fixed_valid_time(
    ds_mean_var: xarray.Dataset,
    variable: str,
    valid_time: Union[np.timedelta64, timedelta, pd.Timedelta, str],
    method: str = "nearest",
) -> xarray.DataArray:
    """Select mean variance at a fixed lead time, returning a DataArray over init_time.

    If valid_time is already absent from the dataset (it was selected upstream),
    the DataArray is returned as-is.
    """
    da = ds_mean_var[variable]
    if "valid_time" in da.dims:
        target_valid_time = _as_timedelta64ns(valid_time)
        da = da.sel(valid_time=target_valid_time, method=method)
    return da


def plot_mean_var_for_fixed_valid_time(
    fixed_time_data: xarray.Dataset,
    variable: str,
    valid_time: Union[np.timedelta64, timedelta, pd.Timedelta, str],
    fig_path: str,
    level: Optional[int] = None,
    fixed_time_data2: Optional[xarray.Dataset] = None,
) -> xarray.DataArray:
    """Plot mean variance (already averaged over lat/lon/pressure_level) as a
    function of init_time for a single fixed lead time."""

    if "init_time" not in fixed_time_data.dims:
        raise KeyError(
            "Expected 'init_time' dimension in the mean-variance dataset. "
            "Ensure data was computed across multiple init times."
        )

    target_valid_time = pd.Timedelta(_as_timedelta64ns(valid_time))
    log_progress(f"Rendering fixed-valid-time figure to {fig_path}")

    fig, ax = plt.subplots()
    ylabel = variable if level is None else f"{variable} ({level} hPa)"
    ax.set_title(
        f"Model ensemble variance at lead time {target_valid_time}, avg. over (lat, lon, level)"
    )
    ax.set_xlabel("Init time")
    ax.set_ylabel(f"{ylabel} : Mean variance across models")
    ax.plot(fixed_time_data["init_time"].values, fixed_time_data.values)
    if fixed_time_data2 is not None:
        ax.plot(fixed_time_data2["init_time"].values, fixed_time_data2.values)
    ax.set_ylim(0, 1)

    plt.savefig(fig_path)
    log_progress("Finished fixed-valid-time plot generation")
    return fixed_time_data


def mean_var_vs_timestep(
    root: str,
    save_root: str,
    variable: str,
    level: Optional[int] = None,
    year: int = YEAR,
    month: Optional[int] = None,
    root2: Optional[str] = None,
    save_root2: Optional[str] = None,
) -> xarray.Dataset:
    output_file = os.path.join(
        save_root,
        (
            f"mean_var_by_forecast_time_{year}_{variable}.nc"
            if month is None
            else f"mean_var_by_forecast_time_{year}_{month:02d}_{variable}.nc"
        ),
    )
    fig_path = os.path.join(
        save_root,
        (
            f"mean_var_by_forecast_time_{year}_{variable}.png"
            if month is None
            else f"mean_var_by_forecast_time_{year}_{month:02d}_{variable}.png"
        ),
    )
    os.makedirs(save_root, exist_ok=True)

    ds_mean_var = get_plot_data(
        root, output_file, variable, level, year=year, month=month
    )
    plot_data = build_mean_var_vs_timestep_data(
        ds_mean_var,
        variable,
        average_init_time=True,
    )

    if root2 is not None:
        assert (
            save_root2 is not None
        ), "save_root2 must be provided if root2 is provided"
        output_file2 = os.path.join(
            save_root2,
            (
                f"mean_var_by_forecast_time_{year}_{variable}.nc"
                if month is None
                else f"mean_var_by_forecast_time_{year}_{month:02d}_{variable}.nc"
            ),
        )
        ds2_mean_var = get_plot_data(
            root2, output_file2, variable, level, year=year, month=month
        )
        plot_data2 = build_mean_var_vs_timestep_data(
            ds2_mean_var,
            variable,
            average_init_time=True,
        )

    log_progress(f"Rendering figure to {fig_path}")
    fig, ax = plt.subplots()
    ax.set_title("Model ensemble variance, avg. over (lat, lon, level)")
    ax.set_xlabel("Forecast steps (6h)")
    ylabel = variable if level is None else f"{variable} ({level} hPa)"
    ax.set_ylabel(f"{ylabel} : Mean variance across models")

    ax.plot(plot_data)
    if root2 is not None:
        ax.plot(plot_data2)
        ax.legend(["Dataset 1", "Dataset 2"])
    plt.savefig(fig_path)
    log_progress("Finished plot generation")
    return ds_mean_var


def _build_land_mask(
    latitude: xarray.DataArray,
    longitude: xarray.DataArray,
) -> xarray.DataArray:
    land_feature = cfeature.NaturalEarthFeature("physical", "land", "110m")
    land_geometry = unary_union(list(land_feature.geometries()))

    lon_values = np.asarray(longitude.values, dtype=float)
    lon_values = ((lon_values + 180.0) % 360.0) - 180.0
    lat_values = np.asarray(latitude.values, dtype=float)

    land_mask = np.zeros((lat_values.size, lon_values.size), dtype=bool)
    for lat_index, lat_value in enumerate(lat_values):
        for lon_index, lon_value in enumerate(lon_values):
            land_mask[lat_index, lon_index] = land_geometry.covers(
                Point(float(lon_value), float(lat_value))
            )

    return xarray.DataArray(
        land_mask,
        coords={"latitude": latitude, "longitude": longitude},
        dims=("latitude", "longitude"),
        name="land_mask",
    )


def _masked_surface_mean(
    da: xarray.DataArray,
    mask: xarray.DataArray,
) -> xarray.DataArray:
    surface_dims = tuple(dim for dim in ("latitude", "longitude") if dim in da.dims)
    if not surface_dims:
        return da.squeeze()

    lat_weights = get_latitude_weights(0.25, lat_dim="latitude", lon_dim="longitude")
    if "latitude" in da.coords and "latitude" in lat_weights.dims:
        lat_weights = lat_weights.reindex({"latitude": da["latitude"]})

    weighted_mask = lat_weights.where(mask)
    denominator = weighted_mask.sum(dim=surface_dims, skipna=True)
    if float(denominator) == 0.0:
        raise ValueError("No grid points were selected for the requested mask")

    numerator = (da * weighted_mask).sum(dim=surface_dims, skipna=True)
    return numerator / denominator


def mean_var_vs_timestep_land_sea(
    root: str,
    save_root: str,
    variable: str,
    level: Optional[int] = None,
    year: int = YEAR,
    month: Optional[int] = None,
    root2: Optional[str] = None,
    save_root2: Optional[str] = None,
) -> xarray.Dataset:
    output_file = os.path.join(
        save_root,
        (
            f"mean_var_by_forecast_time_land_sea_{year}_{variable}.nc"
            if month is None
            else f"mean_var_by_forecast_time_land_sea_{year}_{month:02d}_{variable}.nc"
        ),
    )
    fig_path = os.path.join(
        save_root,
        (
            f"mean_var_by_forecast_time_land_sea_{year}_{variable}.png"
            if month is None
            else f"mean_var_by_forecast_time_land_sea_{year}_{month:02d}_{variable}.png"
        ),
    )
    os.makedirs(save_root, exist_ok=True)

    def _compute_land_sea_dataset(
        input_root: str,
        output_path: str,
    ) -> xarray.Dataset:
        if os.path.exists(output_path):
            log_progress(f"Reusing existing NetCDF {output_path}")
            return xarray.open_dataset(output_path)

        client = maybe_build_dask_client()
        log_progress(f"Dask dashboard: {client.dashboard_link}")

        log_progress(f"Scanning files under {input_root}")
        id_gen = generate_id_from_root(input_root, max_files=None)
        dt_to_file_path_model = group_datasets_by_init_dt(id_gen)
        dt_to_file_path_model = {
            dt: file_paths
            for dt, file_paths in dt_to_file_path_model.items()
            if dt.year == year and (month is None or dt.month == month)
        }
        log_progress(f"Selected {len(dt_to_file_path_model)} init groups for {year}")

        # Build the land mask from a single sample file so we can fold spatial
        # masking into each per-init Dask future.  This avoids accumulating full
        # (init_time × valid_time × lat × lon) arrays in memory.
        first_dt, first_files = next(iter(dt_to_file_path_model.items()))
        first_file_path = first_files[0][0]
        with xarray.open_dataset(first_file_path, engine="h5netcdf") as ds_sample:
            ds_sample = normalize_dataset_coords(ds_sample)
            ds_sample = subset_variables(ds_sample, variable, level)
            land_mask = _build_land_mask(ds_sample["latitude"], ds_sample["longitude"])
        sea_mask = ~land_mask
        land_name = f"{variable}_land"
        sea_name = f"{variable}_sea"

        def _land_sea_post_reduce(ds_i: xarray.Dataset) -> xarray.Dataset:
            da_i = ds_i[variable]
            return xarray.Dataset(
                {
                    land_name: _masked_surface_mean(da_i, land_mask),
                    sea_name: _masked_surface_mean(da_i, sea_mask),
                }
            )

        ds_gen = generate_mfdatasets_by_model(
            dt_to_file_path_model,
            variable,
            level,
            chunks=CHUNKS,
            preprocess_factory=None,
        )

        _, ds_result = compute_mean_var_from_generator(
            ds_gen,
            mean_dim=("pressure_level",),
            var_dim=(MODEL_DIM,),
            latitude_weighting=True,
            level_weighting=True,
            total_items=len(dt_to_file_path_model),
            client=client,
            post_reduce_fn=_land_sea_post_reduce,
        )

        land_series = ds_result[land_name]
        sea_series = ds_result[sea_name]

        if "init_time" in land_series.dims:
            land_series = land_series.mean(dim="init_time")
        if "init_time" in sea_series.dims:
            sea_series = sea_series.mean(dim="init_time")

        ds_land_sea = xarray.Dataset(
            {
                f"{variable}_land": land_series.squeeze(),
                f"{variable}_sea": sea_series.squeeze(),
            }
        )

        log_progress(f"Writing NetCDF to {output_path}")
        log_progress(
            f"Dimensions: {ds_land_sea.dims}, size: {ds_land_sea.nbytes / 1e9:.2f} GB"
        )
        write_start = time.perf_counter()
        ds_land_sea.to_netcdf(output_path)
        log_progress(
            f"Finished writing NetCDF in {format_seconds(time.perf_counter() - write_start)}"
        )

        client.close()
        return ds_land_sea

    ds_land_sea = _compute_land_sea_dataset(root, output_file)
    if root2 is not None:
        assert (
            save_root2 is not None
        ), "save_root2 must be provided if root2 is provided"
        os.makedirs(save_root2, exist_ok=True)
        output_file2 = os.path.join(
            save_root2,
            (
                f"mean_var_by_forecast_time_land_sea_{year}_{variable}.nc"
                if month is None
                else f"mean_var_by_forecast_time_land_sea_{year}_{month:02d}_{variable}.nc"
            ),
        )
        ds_land_sea2 = _compute_land_sea_dataset(root2, output_file2)
    else:
        ds_land_sea2 = None

    log_progress(f"Rendering figure to {fig_path}")
    fig, ax = plt.subplots()
    ax.set_title(
        "Model ensemble variance, avg. over (lat, lon, level), split by land/sea"
    )
    ax.set_xlabel("Forecast steps (6h)")
    ylabel = variable if level is None else f"{variable} ({level} hPa)"
    ax.set_ylabel(f"{ylabel} : Mean variance across models")

    land_name = f"{variable}_land"
    sea_name = f"{variable}_sea"
    ax.plot(
        ds_land_sea[land_name]["valid_time"].values,
        ds_land_sea[land_name].values,
        label="Land",
    )
    ax.plot(
        ds_land_sea[sea_name]["valid_time"].values,
        ds_land_sea[sea_name].values,
        label="Sea",
    )
    if ds_land_sea2 is not None:
        ax.plot(
            ds_land_sea2[land_name]["valid_time"].values,
            ds_land_sea2[land_name].values,
            label="Land (Dataset 2)",
        )
        ax.plot(
            ds_land_sea2[sea_name]["valid_time"].values,
            ds_land_sea2[sea_name].values,
            label="Sea (Dataset 2)",
        )
    ax.legend()

    plt.savefig(fig_path)
    log_progress("Finished land/sea plot generation")
    return ds_land_sea


def mean_var_fixed_valid_time(
    root: str,
    save_root: str,
    variable: str,
    valid_time: Union[np.timedelta64, timedelta, pd.Timedelta, str],
    level: Optional[int] = None,
    year: int = YEAR,
    month: Optional[int] = None,
    root2: Optional[str] = None,
    save_root2: Optional[str] = None,
) -> xarray.DataArray:
    """Compute and plot mean ensemble variance (averaged over lat/lon/pressure_level)
    as a function of init_time at a fixed forecast lead time."""
    fixed_valid_time_label = str(pd.Timedelta(_as_timedelta64ns(valid_time))).replace(
        " ", "_"
    )
    os.makedirs(save_root, exist_ok=True)
    output_file = os.path.join(
        save_root,
        (
            f"mean_var_by_init_time_{year}_{variable}_{fixed_valid_time_label}.nc"
            if month is None
            else f"mean_var_by_init_time_{year}_{month:02d}_{variable}_{fixed_valid_time_label}.nc"
        ),
    )
    fig_path = os.path.join(
        save_root,
        (
            f"mean_var_by_init_time_{year}_{variable}_{fixed_valid_time_label}_zoom.png"
            if month is None
            else f"mean_var_by_init_time_{year}_{month:02d}_{variable}_{fixed_valid_time_label}_zoom.png"
        ),
    )

    ds_mean_var = get_plot_data_fixed_valid_time(
        root, output_file, variable, valid_time, level=level, year=year, month=month
    )
    fixed_time_data = select_mean_var_for_fixed_valid_time(
        ds_mean_var, variable, valid_time
    )

    if root2 is not None:
        assert (
            save_root2 is not None
        ), "save_root2 must be provided if root2 is provided"
        output_file2 = os.path.join(
            save_root2,
            (
                f"mean_var_by_init_time_{year}_{variable}_{fixed_valid_time_label}.nc"
                if month is None
                else f"mean_var_by_init_time_{year}_{month:02d}_{variable}_{fixed_valid_time_label}.nc"
            ),
        )
        ds2_mean_var = get_plot_data_fixed_valid_time(
            root2,
            output_file2,
            variable,
            valid_time,
            level=level,
            year=year,
            month=month,
        )
        fixed_time_data2 = select_mean_var_for_fixed_valid_time(
            ds2_mean_var, variable, valid_time
        )
    else:
        fixed_time_data2 = None

    return plot_mean_var_for_fixed_valid_time(
        fixed_time_data,
        variable,
        valid_time,
        fig_path,
        level=level,
        fixed_time_data2=fixed_time_data2,
    )


def mean_var_fixed_valid_time_init_averaged_spatial(
    root: str,
    save_root: str,
    variable: str,
    valid_time: Union[np.timedelta64, timedelta, pd.Timedelta, str],
    level: Optional[int] = None,
    year: int = YEAR,
    month: Optional[int] = None,
    root2: Optional[str] = None,
    save_root2: Optional[str] = None,
    vmin: int = 0,
    vmax: int = 10,
) -> xarray.Dataset:
    """Compute model variance at a fixed lead time, keep spatial dimensions,
    and average across init_time."""
    fixed_valid_time_label = str(pd.Timedelta(_as_timedelta64ns(valid_time))).replace(
        " ", "_"
    )
    os.makedirs(save_root, exist_ok=True)
    output_file = os.path.join(
        save_root,
        (
            f"mean_var_spatial_init_avg_{year}_{variable}_{fixed_valid_time_label}.nc"
            if month is None
            else f"mean_var_spatial_init_avg_{year}_{month:02d}_{variable}_{fixed_valid_time_label}.nc"
        ),
    )
    fig_path = os.path.join(
        save_root,
        (
            f"mean_var_spatial_init_avg_{year}_{variable}_{fixed_valid_time_label}.png"
            if month is None
            else f"mean_var_spatial_init_avg_{year}_{month:02d}_{variable}_{fixed_valid_time_label}.png"
        ),
    )

    da = get_plot_data_fixed_valid_time_init_averaged_spatial(
        root=root,
        data_save_path=output_file,
        variable=variable,
        valid_time=valid_time,
        level=level,
        year=year,
        month=month,
    )

    if root2 is not None:
        assert (
            save_root2 is not None
        ), "save_root2 must be provided if root2 is provided"
        output_file2 = os.path.join(
            save_root2,
            (
                f"mean_var_spatial_init_avg_{year}_{variable}_{fixed_valid_time_label}.nc"
                if month is None
                else f"mean_var_spatial_init_avg_{year}_{month:02d}_{variable}_{fixed_valid_time_label}.nc"
            ),
        )
        da2 = get_plot_data_fixed_valid_time_init_averaged_spatial(
            root=root2,
            data_save_path=output_file2,
            variable=variable,
            valid_time=valid_time,
            level=level,
            year=year,
            month=month,
        )

    target_valid_time = pd.Timedelta(_as_timedelta64ns(valid_time))
    ylabel = variable if level is None else f"{variable} ({level} hPa)"

    log_progress(f"Rendering spatial heatmap to {fig_path}")

    if root2 is not None:
        fig, ax = plt.subplots(
            1, 2, figsize=(24, 6), subplot_kw={"projection": ccrs.PlateCarree()}
        )

        heatmap1 = ax[0].pcolormesh(
            da["longitude"],
            da["latitude"],
            da.values,
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
        ax[0].coastlines(linewidth=0.8)
        ax[0].add_feature(cfeature.BORDERS, linewidth=0.4)
        ax[0].add_feature(
            cfeature.LAND, facecolor="none", edgecolor="black", linewidth=0.2
        )
        ax[0].set_global()
        ax[0].set_title(
            "Model ensemble variance spatial map "
            f"at lead time {target_valid_time} (init-time averaged)"
        )

        heatmap2 = ax[1].pcolormesh(
            da2["longitude"],
            da2["latitude"],
            da2.values,
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
        ax[1].coastlines(linewidth=0.8)
        ax[1].add_feature(cfeature.BORDERS, linewidth=0.4)
        ax[1].add_feature(
            cfeature.LAND, facecolor="none", edgecolor="black", linewidth=0.2
        )
        ax[1].set_global()
        ax[1].set_title(
            "DS2 - Model ensemble variance spatial map "
            f"at lead time {target_valid_time} (init-time averaged)"
        )
        cbar = fig.colorbar(heatmap2, ax=ax[1], orientation="vertical", pad=0.02)
        cbar.set_label(f"{ylabel} : Mean variance across models")

    else:
        fig = plt.figure(figsize=(12, 6))
        ax: Any = plt.axes(projection=ccrs.PlateCarree())
        heatmap = ax.pcolormesh(
            da["longitude"],
            da["latitude"],
            da,
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(
            cfeature.LAND, facecolor="none", edgecolor="black", linewidth=0.2
        )
        ax.set_global()
        ax.set_title(
            "Model ensemble variance spatial map "
            f"at lead time {target_valid_time} (init-time averaged)"
        )
        cbar = fig.colorbar(heatmap, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label(f"{ylabel} : Mean variance across models")
        plt.tight_layout()

    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    log_progress("Finished spatial heatmap generation")


if __name__ == "__main__":

    root = "/home/users/f/froelicm/share_scratch/turing_calibration_hres_t0/outputs"
    save_root = "/home/users/f/froelicm/WeatherPlots/outputs/turing/fc0"

    # For comparison
    root2 = "/home/users/f/froelicm/share_scratch/turing_calibration/outputs"
    save_root2 = "/home/users/f/froelicm/WeatherPlots/outputs/turing"

    variable = "2m_temperature"
    level = None
    max_lead_days = 5

    # mean_var_vs_timestep(
    #    root=root,
    #    save_root=save_root,
    #    variable=variable,
    #    level=level,
    #    year=YEAR,
    #    month=MONTH,
    #    root2=root2,
    #    save_root2=save_root2,
    # )

    # mean_var_fixed_valid_time(
    #    root=root,
    #    save_root=save_root,
    #    variable=variable,
    #    valid_time=timedelta(hours=24 * max_lead_days),
    #    level=level,
    #    year=YEAR,
    #    month=MONTH,
    #    root2=root2,
    #    save_root2=save_root2,
    # )

    # mean_var_fixed_valid_time_init_averaged_spatial(
    #    root=root,
    #    save_root=save_root,
    #    variable=variable,
    #    valid_time=timedelta(hours=24 * max_lead_days),
    #    level=level,
    #    year=YEAR,
    #    month=MONTH,
    #    root2=root2,
    #    save_root2=save_root2,
    #    vmin=0,
    #    vmax=5,
    # )

    # Land-sea masking:
    mean_var_vs_timestep_land_sea(
        root=root2,
        save_root=save_root2,
        variable=variable,
        level=level,
        year=YEAR,
        month=MONTH,
    )
