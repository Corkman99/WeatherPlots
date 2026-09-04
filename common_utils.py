import glob
import logging
import os
import re
from typing import Callable, Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from pandas import Index

RegionSpec = tuple[Optional[float], Optional[float], Optional[float], Optional[float]]

LEVEL_13 = (
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


# Utility: Standardize xarray dimension names
def standardize_xarray_dims(ds):
    """
    If 'valid_time' is a dimension, rename it to 'time',
    and also rename 'latitude' to 'lat', 'longitude' to 'lon'.
    Works for both xarray.Dataset and xarray.DataArray.
    """
    rename_dict = {}
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        rename_dict["valid_time"] = "time"
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"
    if rename_dict:
        return ds.rename(rename_dict)
    return ds


def _lon_to_360(lon):
    return np.mod(lon, 360.0)


def _lon_to_180(lon: float) -> float:
    return ((float(lon) + 180.0) % 360.0) - 180.0


def wrapped_longitude_delta(lon_end: float, lon_start: float) -> float:
    """Return shortest signed longitudinal delta in degrees."""
    return ((float(lon_end) - float(lon_start) + 540.0) % 360.0) - 180.0


def region_for_dataset_lon_convention(
    ds: Union[xr.Dataset, xr.DataArray],
    region: RegionSpec,
    lon_name: str = "lon",
) -> RegionSpec:
    """Convert region lon bounds to match dataset lon convention."""
    minlat, minlon, maxlat, maxlon = region
    if minlon is None or maxlon is None or lon_name not in ds.coords:
        return region

    lon_min = float(ds[lon_name].min().item())
    lon_max = float(ds[lon_name].max().item())

    if lon_min >= 0 and (minlon < 0 or maxlon < 0):
        minlon = _lon_to_360(minlon)
        maxlon = _lon_to_360(maxlon)
    elif lon_max <= 180 and (minlon > 180 or maxlon > 180):
        minlon = _lon_to_180(minlon)
        maxlon = _lon_to_180(maxlon)

    return (minlat, minlon, maxlat, maxlon)


def select_region(
    ds: Union[xr.Dataset, xr.DataArray],
    region: RegionSpec,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> Union[xr.Dataset, xr.DataArray]:
    """Select a lat/lon region and support lon wraparound windows."""
    minlat, minlon, maxlat, maxlon = region_for_dataset_lon_convention(
        ds, region, lon_name=lon_name
    )

    out = ds

    if minlat is not None and maxlat is not None and lat_name in out.coords:
        lat0 = float(out[lat_name].values[0])
        lat1 = float(out[lat_name].values[-1])
        lat_slice = slice(minlat, maxlat) if lat0 <= lat1 else slice(maxlat, minlat)
        out = out.sel({lat_name: lat_slice})

    if minlon is None or maxlon is None or lon_name not in out.coords:
        return out

    # Select in 0..360 space so wrap windows (minlon > maxlon) can be split safely.
    out = out.assign_coords({lon_name: _lon_to_360(out[lon_name])}).sortby(lon_name)
    minlon_360 = _lon_to_360(minlon)
    maxlon_360 = _lon_to_360(maxlon)

    if minlon_360 <= maxlon_360:
        return out.sel({lon_name: slice(minlon_360, maxlon_360)})

    left = out.sel({lon_name: slice(minlon_360, None)})
    right = out.sel({lon_name: slice(None, maxlon_360)})
    out = xr.concat([left, right], dim=lon_name)
    return out.assign_attrs({**out.attrs, "is_wraparound": True})


def region_to_cartopy_extent(
    region: RegionSpec,
    ds: Optional[Union[xr.Dataset, xr.DataArray]] = None,
) -> Optional[list[float]]:
    """Return [west, east, south, north] in [-180, 180] lon convention."""
    minlat, minlon, maxlat, maxlon = region

    if (
        minlat is not None
        and minlon is not None
        and maxlat is not None
        and maxlon is not None
    ):
        return [_lon_to_180(minlon), _lon_to_180(maxlon), minlat, maxlat]

    if ds is None or "lat" not in ds.coords or "lon" not in ds.coords:
        return None

    lons = np.asarray(ds["lon"].values)
    lats = np.asarray(ds["lat"].values)
    if lons.size == 0 or lats.size == 0:
        return None

    if bool(ds.attrs.get("is_wraparound", False)):
        west = _lon_to_180(float(lons[0]))
        east = _lon_to_180(float(lons[-1]))
    else:
        west = _lon_to_180(float(np.min(lons)))
        east = _lon_to_180(float(np.max(lons)))

    return [west, east, float(np.min(lats)), float(np.max(lats))]


def prep_data(
    ds: xr.Dataset,
    variables: Optional[Union[list[str], dict[str, str]]] = None,
    levels: Optional[list[int]] = None,
    region: Optional[RegionSpec] = None,
    time_range: Union[tuple[Optional[int], Optional[int]], list[int]] = (None, None),
    transform: Optional[dict[str, Callable]] = None,
    reduce: Optional[dict[str, Callable]] = None,
    remove_levels: bool = True,
) -> xr.Dataset:
    """
    Prepares an xarray Dataset by subselecting variables, levels, region, and time,
    and applying transformations or reductions to the data.

    Parameters:
    - ds (xarray.Dataset): input dataset
    - variables (list[str] or dict[str, str]): variable names to select; if dict, keys are original names and values are new names
    - levels (list[int]): pressure levels to select
    - region (RegionSpec): bounding box for subsetting (minlat, minlon, maxlat, maxlon)
    - time_range (tuple[Optional[int], Optional[int]] or list[int]): time indices to select; if tuple, first element is start index and second is end index
    - transform (Optional[dict[str, Callable]]): transformations to apply to variables; keys are variable names and values are functions
    - reduce (Optional[dict[str, Callable]]): reductions to apply to variables; keys are variable names and values are functions

    Returns:
    - xarray.Dataset: processed dataset
    """

    if variables is not None:
        # Handle variable subselection
        if isinstance(variables, dict):
            variable_names = list(variables.keys())
        else:
            variable_names = variables
        ds = ds[variable_names]

    # Subselect levels
    if levels is not None:
        ds = ds.sel(level=levels)

    # Subselect region and time
    if region is not None:
        minlat, minlon, maxlat, maxlon = region
        if "lon" in ds.coords:
            if minlon is None:
                minlon = float(ds["lon"].min().item())
            if maxlon is None:
                maxlon = float(ds["lon"].max().item())
        region = (minlat, minlon, maxlat, maxlon)
        ds = select_region(ds, region)

    if time_range is not None:
        if isinstance(time_range, tuple):
            ds = ds.isel(time=slice(*time_range))
        else:
            ds = ds.isel(time=time_range, drop=False)

    # Apply transformations if any
    if transform:
        for var, t in transform.items():
            result = t(ds[var])
            ds[var] = result

    if reduce:
        for var, r in reduce.items():
            result = r(ds[var])
            ds[var] = result

    if isinstance(variables, dict):
        ds = ds.rename(variables)

    # Remove level dimension if needed:
    if levels is not None and remove_levels:
        for var in ds.data_vars:
            if "level" in ds[var].dims:
                for level in levels:
                    name = f"{var}{level}"
                    dims = [x for x in ds[var].dims if x != "level"]
                    ds[name] = (dims, ds[var].sel(level=level, drop=True).data)
                ds = ds.drop_vars([var])
        ds = ds.drop_dims("level")

    if "batch" in ds.dims:
        ds = ds.squeeze("batch", drop=True)
    return ds


def merge_netcdf_files(
    path: str, pattern: str, dim_name: str = "epoch", chunked: str = "auto"
) -> xr.Dataset:
    """
    Merges multiple NetCDF files in a directory into a single xarray Dataset.
    The index for concatenation is extracted from the integer in the filename.

    Parameters:
    - path (str): directory containing the NetCDF files
    - pattern (str): glob pattern to match the files and from which the dim index is extracted, e.g., "regional_ep-*.nc"
    - dim_name (str): name of the dimension to use for concatenation

    Returns:
    - xarray.Dataset: merged dataset
    """
    files = glob.glob(os.path.join(path, pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {path} matching pattern {pattern}")
    # Extract integer from each filename
    file_tuples = []
    regex = re.compile(r"(\d+)")
    for file in files:
        match = regex.search(os.path.basename(file))
        if match:
            idx = int(match.group(1))
            file_tuples.append((idx, file))
    # Sort by the extracted integer
    file_tuples.sort()
    indices, sorted_files = zip(*file_tuples)
    datasets = [xr.open_dataset(f, chunks=chunked) for f in sorted_files]
    merged_ds = xr.concat(datasets, Index(indices, name=dim_name))
    return merged_ds


def create_multi_panel_figure(
    plot_funcs: list[Callable[[Union[Axes, GeoAxes]], Union[Axes, GeoAxes]]],
    nrows: int,
    ncols: int,
    figsize: tuple[int, int] = (10, 5),
    subplot_kw: dict = {},
    panel_labels: Optional[dict[str, list]] = None,
    colormap: Optional[
        dict[str, tuple[plt.cm.ScalarMappable, tuple[float, float, float, float]]]
    ] = None,
    font_size: int = 12,
) -> Figure:
    """
    Creates a multi-panel figure from a list of plotting functions.

    Parameters:
    - plot_funcs: list of functions that take an ax and return an ax
    - nrows: number of subplot rows
    - ncols: number of subplot columns
    - figsize: size of the entire figure
    - colormap: dictionary of colormap title to tuple of colormap object and position

    Returns:
    - matplotlib Figure
    """
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw=subplot_kw,
        # constrained_layout=True,
    )
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for func, ax in zip(plot_funcs, axes):
        func(ax)

    if panel_labels is not None:
        assert "row" in panel_labels and "col" in panel_labels
        assert len(panel_labels["row"]) == nrows
        assert len(panel_labels["col"]) == ncols

        for row in range(nrows):
            for col in range(ncols):
                ax = axes[row * ncols + col]
                # Row label on first column
                if col == 0:
                    ax.text(
                        -0.1,
                        0.5,
                        panel_labels["row"][row],
                        va="center",
                        ha="right",
                        transform=ax.transAxes,
                        fontsize=font_size,
                        rotation=90,
                    )
                # Column label on top row
                if row == 0:
                    ax.set_title(panel_labels["col"][col], fontsize=font_size)

    if colormap is not None:
        for title, (scalar_map, pos) in colormap.items():
            cbar_ax = fig.add_axes(pos)  # (x, y, width, height)
            cbar = fig.colorbar(
                scalar_map,
                cax=cbar_ax,
                orientation="vertical",
            )
            cbar.set_label(title, fontsize=font_size)
            cbar.ax.yaxis.label.set_size(font_size)
            cbar.ax.tick_params(labelsize=max(font_size - 2, 6))

    return fig


def gencast_like_configs(
    ntruth: int, noptimized: int, ngen: int, ntimesteps: list[int]
) -> dict:

    seq = ["g"] * ngen + ["o"] * noptimized + ["t"] * ntruth

    truth_color = "#000000"  # or #3BC64E green
    optimized_color = "#DE370D"
    gen_color = "#125CC4"

    truth_marker_sizes = 2
    optimized_marker_sizes = 2
    gen_marker_sizes = 0.5

    markers = []
    marker_sizes = []
    alphas = []
    for i, nt in enumerate(ntimesteps):
        markers.append([None] * nt + ["o"])
        if seq[i] == "t":
            ms = truth_marker_sizes
        elif seq[i] == "o":
            ms = optimized_marker_sizes
        else:
            ms = gen_marker_sizes
        marker_sizes.append([None] * nt + [ms])
        alphas.append(list(np.linspace(0.25, 1, nt)))

    colors = (
        [gen_color] * ngen + [optimized_color] * noptimized + [truth_color] * ntruth
    )
    linewidths = [0.5] * ngen + [1] * noptimized + [1] * ntruth

    return {
        "land_color": "#E3DFBF",
        "draw_labels": False,
        "grid": False,
        "coastline_linewidth": 0.5,
        "smooth": False,
        "color": colors,
        "marker": markers,
        "markersize": marker_sizes,
        "linewidth": linewidths,
        "fillstyle": None,
        "alpha": alphas,
        "legend": False,
    }


def gencast_like_configs_color_variation(
    ntruth: int, noptimized: int, ngen: int, ntimesteps: list[int]
) -> dict:
    import matplotlib.cm as cm

    seq = ["g"] * ngen + ["o"] * noptimized + ["t"] * ntruth

    # truth_color = "#000000"  # or #3BC64E green
    # optimized_color = "#DE370D"
    # gen_color = "#125CC4"

    truth_cmap = "Greys"
    optim_cmap = "Reds"
    gen_cmap = "Blues"

    markers = [None] * len(seq)
    marker_sizes = [None] * len(seq)
    alphas = [1.0] * len(seq)
    colors = [gen_cmap] * ngen + [optim_cmap] * noptimized + [truth_cmap] * ntruth
    linewidths = [0.5] * ngen + [1] * noptimized + [1] * ntruth

    dict = {
        "legend": False,
        "plot_kwargs": {
            "land_color": "#E3DFBF",
            "draw_labels": False,
            "grid": False,
            "coastline_linewidth": 0.5,
            "smooth": False,
            "color": colors,
            "cmap": True,
            "marker": markers,
            "markersize": marker_sizes,
            "linewidth": linewidths,
            "alpha": alphas,
            "fillstyle": None,
        },
    }
    return dict


# From graphcast.losses
def normalized_latitude_weights(
    data: xr.DataArray, res, lat_dim_name: str = "lat"
) -> xr.DataArray:
    latitudes_full = np.array(np.arange(-90, 90 + res, res), dtype=np.float32)
    weights = _weight_for_latitude_vector_with_poles(latitudes_full, res)
    weights = weights / weights.mean()

    # Data may not be on the full grid, so find indices of the data to subsample.
    latitudes = np.asarray(data.coords[lat_dim_name].values, dtype=np.float32)
    idx = np.abs(latitudes_full[:, None] - latitudes[None, :]).argmin(axis=0)
    if not np.allclose(latitudes_full[idx], latitudes, atol=max(1e-6, abs(res) * 1e-3)):
        raise ValueError(
            "Data latitude coordinates do not align with the expected uniform grid."
        )

    return xr.DataArray(
        weights[idx],
        dims=lat_dim_name,
        coords={lat_dim_name: data.coords[lat_dim_name].values},
    )


def _weight_for_latitude_vector_with_poles(latitude, res):
    """Weights for uniform latitudes of the form [+- 90, ..., -+90]."""
    if not np.isclose(np.max(latitude), 90.0) or not np.isclose(
        np.min(latitude), -90.0
    ):
        raise ValueError(
            f"Latitude vector {latitude} does not start/end at +- 90 degrees."
        )
    weights = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(res / 2))
    # The two checks above enough to guarantee that latitudes are sorted, so
    # the extremes are the poles
    weights[[0, -1]] = np.sin(np.deg2rad(res / 4)) ** 2
    return weights


def normalized_level_weights(data: xr.DataArray) -> xr.DataArray:
    """Weights proportional to pressure at each level."""
    if "level" in data.coords:
        level = data.coords["level"]
    elif "pressure_level" in data.coords:
        level = data.coords["pressure_level"]
    elif len(data.dims) == 1 and data.dims[0] in data.coords:
        level = data.coords[data.dims[0]]
    else:
        raise KeyError(
            "Expected a level-like coordinate named 'level' or 'pressure_level'."
        )
    return level / level.mean(skipna=False)


# From graphcast-AMSE
def get_model_coords(
    resolution: float, lat_dim_name: str = "lat", lon_dim_name: str = "lon"
) -> tuple[xr.DataArray, xr.DataArray]:
    model_latitude = xr.DataArray(
        np.linspace(-90, 90, int(1 + 180 / resolution), dtype=np.float32),
        dims=lat_dim_name,
    )
    model_latitude = model_latitude.assign_coords({lat_dim_name: model_latitude})
    model_longitude = xr.DataArray(
        np.linspace(
            0,
            360 - resolution,
            int(360 / resolution),
            dtype=np.float32,
        ),
        dims=lon_dim_name,
    )
    model_longitude = model_longitude.assign_coords({lon_dim_name: model_longitude})
    return (model_latitude, model_longitude)


def get_weights(ds, resolution: float) -> tuple[xr.DataArray, xr.DataArray]:
    model_latitude, _ = get_model_coords(resolution)
    latitude_weights = normalized_latitude_weights(model_latitude, resolution)
    latitude_weights = latitude_weights / latitude_weights.mean()
    level_weights = normalized_level_weights(ds)
    return latitude_weights, level_weights


def get_latitude_weights(
    resolution: float, lat_dim: str = "lat", lon_dim: str = "lon"
) -> xr.DataArray:
    model_latitude, _ = get_model_coords(
        resolution, lat_dim_name=lat_dim, lon_dim_name=lon_dim
    )
    weights = normalized_latitude_weights(
        model_latitude, resolution, lat_dim_name=lat_dim
    )
    return weights / weights.mean()


def get_level_weights(levels: Iterable[int], dim: str = "level") -> xr.DataArray:
    level_values = list(levels)
    level = xr.DataArray(level_values, dims=(dim,), coords={dim: level_values})
    return normalized_level_weights(level)


def normalize_dataset(
    ds: xr.Dataset,
    scales: Optional[xr.Dataset],
    locations: Optional[xr.Dataset],
) -> xr.Dataset:
    """Normalize *ds* using per-variable *scales* (stddev) and *locations* (mean).

    Mimics GraphCast's ``normalize`` but works on plain numpy / dask arrays.
    """
    out_vars = {}
    for name in ds.data_vars:
        arr = ds[name]
        if locations is not None and name in locations:
            arr = arr - locations[name].astype(arr.dtype)
        elif locations is not None:
            logging.warning("No normalization location for %s", name)
        if scales is not None and name in scales:
            arr = arr / scales[name].astype(arr.dtype)
        elif scales is not None:
            logging.warning("No normalization scale for %s", name)
        out_vars[name] = arr
    return xr.Dataset(out_vars, coords=ds.coords, attrs=ds.attrs)


def get_max_epoch_in_dir(
    dir: str,
    startswith: str = "output_epoch-",
    endswith: str = ".nc",
) -> tuple[str, int]:
    files = os.listdir(dir)
    print(f"Files in dir {dir}:")
    max_epoch = 0
    file = startswith + "0" + endswith  # default to epoch 0 if no files found
    for f in files:
        if f.startswith(startswith) and f.endswith(endswith):
            epoch_str = f[len(startswith) : -len(endswith)]
            epoch = int(epoch_str)
            if epoch > max_epoch:
                max_epoch = epoch
                file = f
    print(f"Max file: {file} with epoch {max_epoch}")
    return os.path.join(dir, file), max_epoch


def distance_on_sphere(
    point1: tuple[float, float], point2: tuple[float, float]
) -> float:
    """Compute the great-circle distance between two points on a sphere."""
    from math import atan2, cos, radians, sin, sqrt

    lat1, lon1 = point1
    lat2, lon2 = point2

    # Convert latitude and longitude from degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of Earth in kilometers (mean radius)
    R = 6371.0
    distance = R * c
    return distance


def _extract_latlon_tuple(latlon_idx: int):
    """
    Robustly extract (lat, lon) as floats from any xarray return type:
    - tuple/list: return first two elements as floats
    - 0-d array: flatten and take first two as floats
    - scalar: return (float(val), float(val))
    """

    if isinstance(latlon_idx, (tuple, list)):

        if len(latlon_idx) >= 2:
            if isinstance(latlon_idx[0], (tuple, list, np.ndarray)):
                logging.info(f"extract_latlon_tuple: tuple of tuples {latlon_idx}")
                return latlon_idx[0][0], latlon_idx[0][1]
            return latlon_idx[0], latlon_idx[1]

        elif len(latlon_idx) == 1:
            assert isinstance(latlon_idx[0], (tuple, list, np.ndarray))
            logging.info(f"extract_latlon_tuple: single tuple {latlon_idx}")
            return latlon_idx[0][0], latlon_idx[0][1]
        else:
            raise ValueError("latlon_idx tuple/list is empty")
    elif hasattr(latlon_idx, "shape"):
        arr = np.ravel(latlon_idx)
        if arr.size >= 2:
            if isinstance(arr[0], (tuple, list, np.ndarray)):
                logging.info(f"extract_latlon_tuple: array of tuples {arr}")
                return arr[0][0], arr[0][1]
            return arr[0], arr[1]
        elif arr.size == 1:
            assert isinstance(arr[0], (tuple, list, np.ndarray))
            logging.info(f"extract_latlon_tuple: single array tuple {arr}")
            return arr[0][0], arr[0][1]
        else:
            raise ValueError("latlon_idx array is empty")
    raise ValueError(
        f"latlon_idx is sclar {latlon_idx} of type {type(latlon_idx)}, cannot extract lat/lon tuple."
    )


def _extract_hurricane_centers(
    ds: xarray.DataArray, search_region: RegionSpec
) -> list[tuple[float, float]]:
    subregion = select_region(ds, search_region)
    min_coords = []

    if "time" not in subregion.dims:
        subregion = subregion.expand_dims("time")
    for t in subregion.time.values:
        slice_t = subregion.sel(time=t, drop=True)
        stacked = slice_t.stack(points=("lat", "lon"))
        if stacked.size == 0:
            logging.debug("No points in subregion for time %s; skipping", t.values)
            continue
        # Find the index of the minimum value
        argmin_idx = stacked.compute().argmin("points")

        # Get the corresponding lat/lon
        latlon_idx = stacked.points[argmin_idx].values
        min_coords.append(_extract_latlon_tuple(latlon_idx))

    return min_coords


# ---------------------------------------------------------------------------
# Latitude / level weights
# ---------------------------------------------------------------------------


def get_dim_weights(
    ds: Optional[xr.Dataset] = None,
    level_weight_map: Optional[dict[int, float]] = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Derive latitude and level weights directly from an xarray Dataset.

    Returns ``(latitude_weights, level_weights)`` as DataArrays aligned to
    the coordinates of *ds*.
    """
    if ds is None:
        lats = np.arange(-90, 90.1, 0.25)
        lons = np.arange(0, 360, 0.25)
        levels = np.array(LEVEL_13)
        ds = xr.DataArray(
            np.zeros((len(lats), len(lons), len(levels)), dtype=np.float32),
            dims=["lat", "lon", "level"],
            coords={"lat": lats, "lon": lons, "level": levels},
        ).to_dataset(name="dummy")

    # -- Latitude weights ----------------------------------------------------
    if "lat" in ds.coords:
        lat_weights = normalized_latitude_weights(ds["lat"], res=0.25)
    else:
        lat_weights = xr.DataArray(1.0)

    # -- Level weights -------------------------------------------------------
    if "level" in ds.dims:
        levels = ds["level"].values.astype(float)
        if level_weight_map:
            weights = np.zeros_like(levels, dtype=np.float32)
            for lvl, w in level_weight_map.items():
                idx = np.where(levels == int(lvl))[0]
                if idx.size > 0:
                    weights[idx[0]] = float(w)
            total = weights.sum()
            if total > 0:
                weights = weights / total
            level_weights = xr.DataArray(
                weights, dims=("level",), coords={"level": ds["level"].values}
            )
        else:
            # Uniform
            n = len(levels)
            level_weights = xr.DataArray(
                np.full(n, 1.0 / n, dtype=np.float32),
                dims=("level",),
                coords={"level": ds["level"].values},
            )
    else:
        level_weights = xr.DataArray(1.0)

    return lat_weights, level_weights


from datetime import datetime
from typing import Any


def extract_from_experiment_config(
    dir: str,
    spec: tuple[Optional[type], tuple[str, ...]],
    config_file_name: str = "experiment.json",
) -> Any:
    import json

    with open(os.path.join(dir, config_file_name), "r") as f:
        config = json.load(f)
    for key in spec[1]:
        config = config[key]

    expected = spec[0]
    if expected is datetime:
        if isinstance(config, datetime):
            return config
        if isinstance(config, str):
            return datetime.fromisoformat(config)
        raise TypeError(
            f"Expected datetime/string at {spec[1]} in config, got {type(config)}"
        )

    if expected is not None and not isinstance(config, expected):
        raise TypeError(
            f"Expected type {expected} at {spec[1]} in config, got {type(config)}"
        )

    return config
