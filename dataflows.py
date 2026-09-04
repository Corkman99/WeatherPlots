import glob
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Optional, overload

import numpy as np
import pandas as pd
import xarray

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default per-variable weights (empty → all variables weighted equally at 1)
# ---------------------------------------------------------------------------

DEFAULT_PER_VARIABLE_WEIGHTS: dict[str, float] = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1,
    "geopotential": 1.0,
    "vertical_velocity": 1.0,
    "temperature": 1.0,
    "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0,
    "specific_humidity": 1.0,
}


def load_netcdf_collection(
    path: str,
    pattern: str,
    dim_name: str = "epoch",
    first_target_datetime: Optional[datetime] = None,
    chunked: str = "auto",
) -> xarray.Dataset:
    """Load multiple NetCDF files and concatenate along dim_name with canonical grid."""
    files = glob.glob(os.path.join(path, pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {path} matching pattern {pattern}")

    regex = re.compile(r"(\d+)")
    file_tuples = []
    for f in files:
        match = regex.search(os.path.basename(f))
        if match:
            idx = int(match.group(1))
            file_tuples.append((idx, f))
    if not file_tuples:
        raise ValueError(f"No indexed files found in {path} matching pattern {pattern}")

    file_tuples.sort()
    indices, sorted_files = zip(*file_tuples)
    datasets = [
        load_dataset(f, first_target_datetime=first_target_datetime, chunks=chunked)
        for f in sorted_files
    ]

    merged = xarray.concat(datasets, dim=pd.Index(indices, name=dim_name))

    return merged


@overload
def load_dataset(
    source: str | os.PathLike,
    first_target_datetime: Optional[datetime] = None,
    chunks: str | dict = "auto",
) -> xarray.Dataset:
    pass


@overload
def load_dataset(
    source: xarray.Dataset,
    first_target_datetime: Optional[datetime] = None,
    chunks: str | dict = "auto",
) -> xarray.Dataset:
    pass


def load_dataset(
    source: str | os.PathLike | xarray.Dataset,
    first_target_datetime: Optional[datetime] = None,
    chunks: str | dict = "auto",
    tres: timedelta = timedelta(hours=6),
) -> xarray.Dataset:
    """Load a dataset from path or pass through existing dataset.
    Normalizes data format.
    """
    if isinstance(source, os.PathLike | str):
        ds = _load_and_validate(str(source), chunks=chunks)
    else:
        assert isinstance(source, xarray.Dataset), f"type: {type(source)}"
        ds = source

    # Format normalization
    if is_aiwm2(ds):
        ds = aiwm2_preprocess(ds, first_target_datetime, tres)

    ds = _normalize_lat_lon(ds)

    if "batch" in ds.dims:
        ds = ds.squeeze("batch", drop=True)

    _validate_format(ds)

    return ds


def _load_and_validate(
    path: str, chunks: Optional[str | dict] = "auto"
) -> xarray.Dataset:
    import hdf5plugin  # specifies path to look for alternative hdf5 decompression algo

    def _validate_opened_dataset(ds: xarray.Dataset) -> None | Warning:
        if ds.nbytes == 0:
            return Warning(f"Empty object: 0 bytes of data.")

        var = list(ds.data_vars)[0]
        indexers = {d: 0 for d in ds[var].dims}

        try:
            # May trigger full data read if not chunked.
            ds[var].isel(indexers).load()
        except Exception as e:
            return Warning("Failed loading data.")

    engines = ["netcdf4", "h5netcdf", "scipy"]
    for eng in engines:
        ds = xarray.open_dataset(path, engine=eng, chunks=chunks)
        if _validate_opened_dataset(ds):
            continue
        return ds

    raise ValueError(
        f"Failed to open dataset at {path} with any of the engines: {engines}"
    )


def is_aiwm2(ds: xarray.Dataset) -> bool:
    return "valid_time" in ds.coords


def aiwm2_preprocess(
    data: xarray.Dataset,
    first_target_datetime=None,
    tres: timedelta = timedelta(hours=6),
) -> xarray.Dataset:

    AIWM2_TO_GCAST_DIMS = {
        "latitude": "lat",
        "longitude": "lon",
        "pressure_level": "level",
        "valid_time": "time",
    }
    data = data.rename(
        {k: v for k, v in AIWM2_TO_GCAST_DIMS.items() if k in data.coords}
    )

    if "total_precipitation" in data.data_vars:
        precip_name = f"total_precipitation_{int(tres.total_seconds() / 3600)}hr"
        data = data.rename({"total_precipitation": precip_name})

    dt = [datetime.fromisoformat(str(x)) for x in data["time"].values]

    first_target_datetime = first_target_datetime or dt[2]
    td = [t - (first_target_datetime - tres) for t in dt]

    data = data.assign_coords(time=td)
    data.coords["datetime"] = xarray.DataArray(dt, dims=("time",))

    return data


def _normalize_lat_lon(ds: xarray.Dataset) -> xarray.Dataset:
    """Ensure lat/lon are in canonical ascending order."""
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})

    # Skip normalization if this is a wraparound subset (lon already properly ordered)
    is_wraparound = ds.attrs.get("is_wraparound", False)
    print(
        f"[DEBUG _normalize_lat_lon] is_wraparound flag: {is_wraparound}, attrs: {ds.attrs}"
    )
    if is_wraparound:
        print(
            f"[DEBUG _normalize_lat_lon] Skipping normalization for wraparound dataset"
        )
        return ds

    # lat: -90 -> 90
    print(f"[DEBUG _normalize_lat_lon] Applying normalization")
    ds = ds.sortby("lat")

    # lon: 0 -> 360
    if ds["lon"].min() < 0:
        ds = ds.assign_coords(lon=((ds["lon"] + 360) % 360))

    return ds


def _validate_format(ds: xarray.Dataset) -> None:
    assert ds.nbytes > 0

    assert "time" in ds.dims
    assert "datetime" in ds.coords

    assert "lat" in ds.dims
    assert ds["lat"][0] < ds["lat"][-1]

    assert "lon" in ds.dims
    assert ds["lon"].min() >= 0

    assert "batch" not in ds.dims


def datetime_to_np(dt: datetime) -> np.datetime64:
    return pd.to_datetime(dt).to_datetime64()


def select_datetime(
    data: xarray.Dataset,
    datetimes: list[datetime],
) -> xarray.Dataset:
    typed_datetimes = datetimes
    if data["datetime"].dtype.kind == "M":
        typed_datetimes = [datetime_to_np(dt) for dt in datetimes]
    assert set(typed_datetimes) <= set(data["datetime"].values), (
        f"Data does not contain requested datetimes."
        f"Requested: {typed_datetimes}, available: {data['datetime'].values}"
    )
    data = (
        data.swap_dims({"time": "datetime"})
        .sel(datetime=typed_datetimes, drop=False)
        .swap_dims({"datetime": "time"})
    )
    data = data.sortby("time")
    return data
