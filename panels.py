from typing import Union, Dict, Optional, Callable, List, Tuple, Any
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


def colored_line_between_pts(x, y, c, ax, **lc_kwargs):
    """
    Copied from: https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

    Plot a line with a color specified between (x, y) points by a third value.

    It does this by creating a collection of line segments between each pair of
    neighboring points. The color of each segment is determined by the
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should have a size one less than that of x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    from matplotlib.collections import LineCollection

    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=c, **lc_kwargs)
    # Set the values used for colormapping
    # lc.set_array(c)

    return ax.add_collection(lc)


def plot_tropical_hurricane_track(
    ax: GeoAxes,
    ds: Tuple[Union[xr.Dataset, xr.DataArray], ...],
    search_region: Tuple[float, float, float, float],
    plot_region: Tuple[float, float, float, float],
    title: Optional[str] = None,
    legend: bool = True,
    plot_kwargs: Dict[str, Any] = {},
):
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.cm as cm

    def _extract_hurricane_centers(mslp, minlat, minlon, maxlat, maxlon, tol=20):
        assert mslp.lat.min() <= minlat
        assert mslp.lat.max() >= maxlat
        assert mslp.lon.min() <= minlon
        assert mslp.lon.max() >= maxlon

        subregion = mslp.sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))
        min_coords = []

        for t in subregion.time:
            slice_t = subregion.sel(time=t, drop=True)

            # Stack lat/lon into a single dimension, find the min, then map back
            stacked = slice_t.stack(points=("lat", "lon"))
            min_val = stacked.min("points")
            min_points = stacked.where(stacked == min_val, drop=True)
            indices = min_points.points.values
            """if len(indices) > 1:
                min_value = stacked.min()
                min_points = slice_t.where(slice_t == min_value)
                min_lat = float(np.mean(min_points.coords["lat"].values))
                min_lon = float(np.mean(min_points.coords["lon"].values))
                min_point = (min_lat, min_lon)
            else:"""

            min_point = stacked.idxmin(
                "points"
            ).item()  # gives a point index (lat, lon tuple)

            # item() returns a tuple (lat_value, lon_value) because our index is MultiIndex
            min_lat, min_lon = map(float, min_point)

            if min_coords:
                prev_lon, prev_lat = min_coords[-1]
                dist = np.sqrt((min_lon - prev_lon) ** 2 + (min_lat - prev_lat) ** 2)
                if dist > tol:
                    min_lon, min_lat = prev_lon, prev_lat

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
            _extract_hurricane_centers(
                d,
                search_region[0],
                search_region[1],
                search_region[2],
                search_region[3],
            )
        )

    # plot_kwargs are lists of colors, markers, labels, alpha, etc. Set defaults if not provided

    n = len(centers)

    if "color" not in plot_kwargs:
        c = cm.get_cmap("Paired", n)
        plot_kwargs["color"] = [c(x) for x in range(n)]
        plot_kwargs["cmap"] = False
    if "alpha" not in plot_kwargs:
        plot_kwargs["alpha"] = [1.0] * n
    if "marker" not in plot_kwargs:
        plot_kwargs["marker"] = ["o"] * n
    if "markersize" not in plot_kwargs:
        plot_kwargs["markersize"] = [1] * n
    if "label" not in plot_kwargs:
        plot_kwargs["label"] = [f"Dataset {i+1}" for i in range(n)]
    if "linestyle" not in plot_kwargs:
        plot_kwargs["linestyle"] = ["-"] * n
    if "linewidth" not in plot_kwargs:
        plot_kwargs["linewidth"] = [1] * n

    if isinstance(ax, GeoAxes):
        ax.set_extent(
            [plot_region[1], plot_region[3], plot_region[0], plot_region[2]],
            crs=ccrs.PlateCarree(),
        )
        ax.coastlines(linewidth=plot_kwargs.get("coastline_linewidth", 1))
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(
            cfeature.LAND, facecolor=plot_kwargs.get("land_color", "lightgray")
        )

        if plot_kwargs.get("grid", False):
            ax.gridlines(draw_labels=plot_kwargs.get("draw_labels", True))

    for i, center in enumerate(centers):
        lons, lats = zip(*center)
        if plot_kwargs.get("smooth", False):
            from scipy.signal import savgol_filter

            lons = savgol_filter(
                lons, window_length=plot_kwargs.get("smoothing_window", 7), polyorder=3
            )
            lats = savgol_filter(
                lats, window_length=plot_kwargs.get("smoothing_window", 7), polyorder=3
            )

        if plot_kwargs["cmap"]:
            color_spec = cm.get_cmap(plot_kwargs["color"][i])
            col = [color_spec(j) for j in np.linspace(0, 1, len(lons) - 1)]
            line = colored_line_between_pts(
                lons,
                lats,
                col,
                ax,
                # marker=plot_kwargs["marker"][i],
                # markersize=plot_kwargs["markersize"][i],
                label=plot_kwargs["label"][i],
                # fillstyle=plot_kwargs.get("fillstyle", "none"),
                linestyle=plot_kwargs["linestyle"][i],
                linewidth=plot_kwargs["linewidth"][i],
                alpha=plot_kwargs["alpha"][i],
                transform=ccrs.PlateCarree(),
            )
        else:
            ax.plot(
                lons,
                lats,
                marker=plot_kwargs["marker"][i],
                color=plot_kwargs["color"][i],
                markersize=plot_kwargs["markersize"][i],
                label=plot_kwargs["label"][i],
                fillstyle=plot_kwargs.get("fillstyle", "none"),
                linestyle=plot_kwargs["linestyle"][i],
                linewidth=plot_kwargs["linewidth"][i],
                alpha=plot_kwargs["alpha"][i],
                transform=ccrs.PlateCarree(),
            )

    if legend:
        ax.legend(loc="upper right")
    if title:
        ax.set_title(title)

    return ax, centers


def plot_timeseries_losses(
    ax: Union[Axes, GeoAxes],
    ds: np.ndarray,
    stride: Optional[int] = 5,
    title: Optional[str] = None,
    **kwargs: dict,
):
    """
    Input is a 2D np array. One the first axis, epoch. On the second axis, time.
    stride gives the option to skip over the time axis, e.g., to plot every 5th time step.
    Return axes with plot with time on the x-axis and loss on the y-axis.
    """

    if ds.ndim == 1:
        ax.plot(np.arange(0, ds.shape[0]), ds, label="Total Loss")
        if "ylim" in kwargs:
            assert isinstance(kwargs["ylim"], tuple) and len(kwargs["ylim"]) == 2
            ax.set_ylim(kwargs["ylim"])
        if "xlim" in kwargs:
            assert isinstance(kwargs["xlim"], tuple) and len(kwargs["xlim"]) == 2
            ax.set_xlim(kwargs["xlim"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        return ax

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
        if "ylim" in kwargs:
            assert isinstance(kwargs["ylim"], tuple) and len(kwargs["ylim"]) == 2
            ax.set_ylim(kwargs["ylim"])
        if "xlim" in kwargs:
            assert isinstance(kwargs["xlim"], tuple) and len(kwargs["xlim"]) == 2
            ax.set_xlim(kwargs["xlim"])

    ax.set_xlabel("Time")
    ax.set_ylabel("Loss")
    ax.legend(title="Epoch", loc="upper right")

    if title is not None:
        ax.set_title(title)

    return ax


def plot_variable_as_line(
    ax: Union[Axes, GeoAxes],
    ds: xr.Dataset,
):
    # x-axis is time
    # y-axis is some score
    for line in ds.data_vars:
        ax.plot(ds[line].time, ds[line], label=line)

    # add a horizontal line at y=0
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    ax.legend()
    return ax
