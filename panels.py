from typing import Union, Dict, Optional, Callable, List, Tuple
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes

"""
Plotting Functions for xarrays

The following function is a signature for the other functions contained in this document. They take as argument 
at least a matplotlib axes and a xarray dataset, dataarray or list/dict/tuple of xarray elements, and return 
the matplotlib axes element with drawn elements.
"""


def plot_func(
    ax: Axes,
    ds: Union[xr.Dataset, xr.DataArray, Dict[Union[xr.Dataset, xr.DataArray], str]],
    **kwargs,
) -> Axes:
    """
    Plot a single panel on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        The axes to plot on.

    ds : xarray.Dataset or xarray.Dataarray
        Dataset containing the variables to plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot drawn on it.
    """
    return ax


def plot_time_variable_panel(
    ax: Axes,
    ds: Dict[Union[xr.Dataset, xr.DataArray], str],
    variable: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    **plot_kwargs,
) -> Axes:
    """
    Plots 2D time-series of variables on a given axes, for different datasets.

    Parameters:
    - ax: matplotlib axes object to plot on
    - ds: dictionary of {xarray Dataset or DataArray: label}
    - variable: name of the variable to extract if inputs are Datasets
    - plot_kwargs: additional kwargs for ax.plot()

    Returns:
    - ax: matplotlib axes object with plot
    """

    color_list = plot_kwargs.pop("color", None)  # find a better solution to this
    alpha_list = plot_kwargs.pop("alpha", None)  # find a better solution to this

    for i, (label, x) in enumerate(ds.items()):
        # Extract variable if needed
        if isinstance(x, xr.Dataset):
            if variable is not None:
                data_array = x[variable]
            else:
                raise ValueError(
                    "If input is an xarray.Dataset, 'variable' must be provided."
                )
        else:
            data_array = x

        # Collapse spatial dimensions if present
        if {"lat", "lon"} <= set(data_array.dims):
            data_array = data_array.mean(dim=["lat", "lon"])
        ax.plot(
            data_array["datetime"],
            data_array,
            label=label,
            color=color_list[i],
            alpha=alpha_list[i],
        )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel("Time")

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(variable if variable else "Value")

    if legend:
        ax.legend()
    return ax


def plot_map_panel(
    ax: Axes,
    ds: Union[xr.Dataset, xr.DataArray],
    fcontour: Optional[dict] = None,
    contour: Optional[dict] = None,
    arrows: Optional[dict] = None,
    region: Optional[Tuple[float, float, float, float]] = None,
    title: Optional[str] = None,
    projection: Optional[ccrs.Projection] = ccrs.PlateCarree(),
    legend_object: bool = True,
):
    """
    Plot a map from an xarray.Dataset with optional filled contours, line contours, and arrows.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing all the variables to be plotted.

    region : dict, optional
        tuple of minlat, minlon, maxlat, maxlon

    fcontour : dict, optional
        Dictionary with two entries: {'variable': var_name, 'specs': plot_kwargs},
        where plot_kwargs is a dictionary and can include: cmap, norm, levels, extend, etc.

    contour : dict, optional
        Dictionary with two entries: {'variable': var_name, 'specs': plot_kwargs},
        where plot_kwargs is a dictionary and can include: colors, levels, linewidths, linestyles, etc.

    arrows : dict, optional
        Dictionary with two entries: {'variable': [var_name_u, var_name_v], 'specs': plot_kwargs},
        where plot_kwargs is a dictionary and can include: color, scale, regrid_slice
        (e.g., (slice(None, None, 5), slice(None, None, 5))) for downsampling.

    projection : cartopy.crs, optional
        The projection for the plot. Default is PlateCarree.

    Returns
    -------
    ax : matplotlib Axes
        The axes with the plotted data.
    """

    if all([fcontour, contour, arrows]) is None:
        raise ValueError("At least one component must be specified")

    # Subset region
    if region is not None:
        lat_min, lon_min, lat_max, lon_max = region
        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # Plot filled contour
    if fcontour is not None:
        pc = ax.contourf(
            ds.lon,
            ds.lat,
            ds[fcontour["variable"]],
            transform=projection,
            **fcontour["specs"],
        )

    # Plot contour lines
    if contour is not None:
        spec = contour["specs"].copy()
        lab = False
        if "label" in spec.keys():
            lab = spec.pop("label")

        cs = ax.contour(
            ds.lon,
            ds.lat,
            ds[contour["variable"]],
            transform=projection,
            **spec,
        )
        if lab:
            ax.clabel(cs, inline=True, fontsize=8, fmt="%d")  # You can customize this

    # Plot arrows
    if arrows is not None:
        var_u = ds[arrows["variable"][0]]
        var_v = ds[arrows["variable"][1]]
        spec = arrows["specs"].copy()
        regrid = spec.pop("regrid_slice", (slice(None, None), slice(None, None)))

        quiv = ax.quiver(
            ds.lon.values[regrid[1]],
            ds.lat.values[regrid[0]],
            var_u.values[regrid],
            var_v.values[regrid],
            transform=projection,
            **spec,
        )

    # Add map features
    if isinstance(ax, GeoAxes):
        ax.coastlines()
        ax.set_extent(
            [ds.lon.min(), ds.lon.max(), ds.lat.min(), ds.lat.max()],
            crs=projection,
        )

    if title is not None:
        ax.set_title(title)

    return ax


# In the same style as the other functions in this file,
# the function plots in 2D time on the x-axis and level on the y-axis.
# If time and level are not present in dimensions, it raises an error.
# fcontour takes a dictionary of variable to be plotted and matplotlib specifications,
# which will be passed to ax.contourf.
# same for contour, but passed to ax.contour.
def plot_temporal_vertical_profile(
    ax: Axes,
    ds: Union[xr.Dataset, xr.DataArray],
    fcontour: Optional[dict] = None,
    contour: Optional[dict] = None,
    title: Optional[str] = None,
):

    if all([fcontour, contour]) is None:
        raise ValueError("At least one component must be specified")

    # Plot filled contour
    if fcontour is not None:

        assert "level" in ds[fcontour["variable"]].dims
        assert "time" in ds[fcontour["variable"]].dims

        if "lat" in ds[fcontour["variable"]].dims:
            # If lat is present, we average over it
            ds = ds[fcontour["variable"]].mean(dim="lat")
        if "lon" in ds[fcontour["variable"]].dims:
            # If lon is present, we average over it
            ds = ds[fcontour["variable"]].mean(dim="lon")

        ax.contourf(
            ds.time,
            ds.level,
            ds[fcontour["variable"]],
            transform=ccrs.PlateCarree(),
            **fcontour["specs"],
        )

    # Plot contour lines
    if contour is not None:

        assert "level" in ds[contour["variable"]].dims
        assert "time" in ds[contour["variable"]].dims

        if "lat" in ds[contour["variable"]].dims:
            # If lat is present, we average over it
            ds = ds[contour["variable"]].mean(dim="lat")
        if "lon" in ds[contour["variable"]].dims:
            # If lon is present, we average over it
            ds = ds[contour["variable"]].mean(dim="lon")

        spec = contour["specs"].copy()
        lab = False
        if "label" in spec.keys():
            lab = spec.pop("label")

        cs = ax.contour(
            ds.time,
            ds.level,
            ds[contour["variable"]],
            transform=ccrs.PlateCarree(),
            **spec,
        )

        if lab:
            ax.clabel(cs, inline=True, fontsize=8, fmt="%d")  # You can customize this

    # Set labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Pressure level (hPa)")
    if title:
        ax.set_title(title)

    return ax


def plot_tropical_hurricane_track(
    ax: Union[Axes, GeoAxes],
    ds: List[Union[xr.Dataset, xr.DataArray]],
    region: Tuple[float, float, float, float],
    title: Optional[str] = None,
    legend: bool = True,
    **plot_kwargs,
):
    def _extract_hurricane_centers(mslp, minlat, minlon, maxlat, maxlon):
        subregion = mslp.sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))
        min_coords = []
        for t in subregion.time:
            slice_t = subregion.sel(time=t, drop=True).drop_vars(
                [v for v in subregion.coords if v not in ("lat", "lon")],
                errors="ignore",
            )
            # Find the indices of the minimum value
            min_idx = np.unravel_index(
                np.argmin(slice_t.values, axis=None), slice_t.shape
            )
            # Get the corresponding lat/lon values
            min_lat = float(slice_t.lat.values[min_idx[0]])
            min_lon = float(slice_t.lon.values[min_idx[1]])
            min_coords.append((min_lon, min_lat))
        return min_coords

    n = len(ds)
    if n == 0:
        raise ValueError("No datasets provided for plotting.")

    centers = []
    for d in ds:
        if isinstance(d, xr.Dataset):
            var_name = list(d.data_vars)[0]
            d = d[var_name]
        centers.append(
            _extract_hurricane_centers(d, region[0], region[1], region[2], region[3])
        )

    # plot_kwargs are lists of colors, markers, labels, alpha, etc. Set defaults if not provided
    if "color" not in plot_kwargs:
        # default is tab10 colors
        plot_kwargs["color"] = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
        ][: len(centers)]
    if "marker" not in plot_kwargs:
        plot_kwargs["marker"] = ["o"] * n  # default is all the same markers
    if "label" not in plot_kwargs:
        # default labels are Dataset 1, Dataset 2, etc.
        plot_kwargs["label"] = [f"Dataset {i+1}" for i in range(n)]
    if "markersize" not in plot_kwargs:
        plot_kwargs["markersize"] = [5] * n  # default markersize
    if "alpha" not in plot_kwargs:
        plot_kwargs["alpha"] = [0.7] * n  # default alpha

    if isinstance(ax, GeoAxes):
        ax.set_extent(
            [region[1], region[3], region[0], region[2]], crs=ccrs.PlateCarree()
        )
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.gridlines(draw_labels=True)

    for i, center in enumerate(centers):
        lons, lats = zip(*center)
        ax.plot(
            lons,
            lats,
            marker=plot_kwargs["marker"][i],
            color=plot_kwargs["color"][i],
            markersize=plot_kwargs["markersize"][i],
            alpha=plot_kwargs["alpha"][i],
            label=plot_kwargs["label"][i],
            transform=ccrs.PlateCarree(),
        )

    if legend:
        ax.legend(loc="upper right")
    if title:
        ax.set_title(title)

    return ax


def plot_timeseries_losses(
    ax: Union[Axes, GeoAxes],
    ds: np.ndarray,
    stride: Optional[int] = 5,
    title: Optional[str] = None,
):
    """
    Input is a 2D np array. One the first axis, epoch. On the second axis, time.
    stride gives the option to skip over the time axis, e.g., to plot every 5th time step.
    Return axes with plot with time on the x-axis and loss on the y-axis.
    """

    # If stride is not specified, default to 1 (no skipping)
    if stride is None:
        stride = 1

    # each ds row should be plotted as a separate line
    # calculate the number of epochs to plot according to the stride
    line_idx = np.arange(0, ds.shape[0], stride)
    for i in line_idx:
        ax.plot(
            np.arange(0, ds.shape[1]),
            ds[i, :],
            label=f"Epoch {i + 1}",
            alpha=0.7,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Loss")
    ax.legend(title="Epoch", loc="upper right")

    if title is not None:
        ax.set_title(title)

    return ax
