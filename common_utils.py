import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Union, Dict, Optional, Callable, List, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from cartopy.mpl.geoaxes import GeoAxes
from pandas import Index
import glob
import re


def prep_data(
    ds: xr.Dataset,
    variables: Union[List[str], Dict[str, str]],
    levels: Optional[List[int]],
    region: Optional[
        Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]
    ],
    time_range: Union[Tuple[Optional[int], Optional[int]], List[int]],
    transform: Optional[Dict[str, Callable]] = None,
    reduce: Optional[Dict[str, Callable]] = None,
) -> xr.Dataset:
    """
    Prepares an xarray Dataset by subselecting variables, levels, region, and time,
    and applying transformations or reductions to the data.

    Parameters:
    - ds (xarray.Dataset): input dataset
    - variables (List[str] or Dict[str, str]): variable names to select; if dict, keys are original names and values are new names
    - levels (List[int]): pressure levels to select
    - region (Tuple[float, float, float, float]): bounding box for subsetting (minlat, minlon, maxlat, maxlon)
    - time_range (Tuple[Optional[int], Optional[int]] or List[int]): time indices to select; if tuple, first element is start index and second is end index
    - transform (Optional[Dict[str, Callable]]): transformations to apply to variables; keys are variable names and values are functions
    - reduce (Optional[Dict[str, Callable]]): reductions to apply to variables; keys are variable names and values are functions

    Returns:
    - xarray.Dataset: processed dataset
    """

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
        ds = ds.sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))

    if isinstance(time_range, Tuple):
        ds = ds.isel(time=slice(*time_range))
    else:
        ds = ds.isel(time=time_range)

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
    if levels is not None:
        for var in ds.data_vars:
            if "level" in ds[var].dims:
                for level in levels:
                    name = f"{var}{level}"
                    dims = [x for x in ds[var].dims if x != "level"]
                    ds[name] = (dims, ds[var].sel(level=level, drop=True).data)
                ds = ds.drop_vars([var])
        ds = ds.drop_dims("level")
    return ds.squeeze()


def merge_netcdf_files(path: str, pattern: str, dim_name: str = "epoch") -> xr.Dataset:
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
    datasets = [xr.open_dataset(f) for f in sorted_files]
    merged_ds = xr.concat(datasets, Index(indices, name=dim_name))
    return merged_ds


def create_multi_panel_figure(
    plot_funcs: List[Callable[[Union[Axes, GeoAxes]], Union[Axes, GeoAxes]]],
    nrows: int,
    ncols: int,
    figsize: Tuple[int, int] = (10, 5),
    subplot_kw: Dict = {},
    panel_labels: Optional[Dict[str, List]] = None,
    colormap: Optional[
        Dict[str, Tuple[plt.cm.ScalarMappable, Tuple[float, float, float, float]]]
    ] = None,
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
                        fontsize=12,
                        rotation=90,
                    )
                # Column label on top row
                if row == 0:
                    ax.set_title(panel_labels["col"][col], fontsize=12)

    if colormap is not None:
        for title, (scalar_map, pos) in colormap.items():
            cbar_ax = fig.add_axes(pos)  # (x, y, width, height)
            cbar = fig.colorbar(
                scalar_map,
                cax=cbar_ax,
                orientation="vertical",
            )
            cbar.set_label(title, fontsize=12)

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
