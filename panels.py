import logging

_logger = logging.getLogger(__name__)


def extract_latlon_tuple(latlon_idx):
    """
    Robustly extract (lat, lon) as floats from any xarray return type:
    - tuple/list: return first two elements as floats
    - 0-d array: flatten and take first two as floats
    - scalar: return (float(val), float(val))
    """

    def to_360(lon):
        """Convert longitude to 0–360 convention."""
        lon = float(lon)
        return lon if 0 <= lon <= 360 else (lon + 360 if lon < 0 else lon)

    if isinstance(latlon_idx, (tuple, list)):
        if len(latlon_idx) >= 2:
            if isinstance(latlon_idx[0], (tuple, list, np.ndarray)):
                logging.info(f"extract_latlon_tuple: tuple of tuples {latlon_idx}")
                lat = float(latlon_idx[0][0])
                lon = to_360(latlon_idx[0][1])
                return lat, lon
            lat = float(latlon_idx[0])
            lon = to_360(latlon_idx[1])
            return lat, lon
        elif len(latlon_idx) == 1:
            if isinstance(latlon_idx[0], (tuple, list, np.ndarray)):
                logging.info(f"extract_latlon_tuple: single tuple {latlon_idx}")
                lat = float(latlon_idx[0][0])
                lon = to_360(latlon_idx[0][1])
                return lat, lon
            lat = float(latlon_idx[0])
            lon = to_360(latlon_idx[0])
            return lat, lon
        else:
            raise ValueError("latlon_idx tuple/list is empty")
    elif hasattr(latlon_idx, "shape"):
        arr = np.ravel(latlon_idx)
        if arr.size >= 2:
            if isinstance(arr[0], (tuple, list, np.ndarray)):
                logging.info(f"extract_latlon_tuple: array of tuples {arr}")
                lat = float(arr[0][0])
                lon = to_360(arr[0][1])
                return lat, lon
            lat = float(arr[0])
            lon = to_360(arr[1])
            return lat, lon
        elif arr.size == 1:
            if isinstance(arr[0], (tuple, list, np.ndarray)):
                logging.info(f"extract_latlon_tuple: single array tuple {arr}")
                lat = float(arr[0][0])
                lon = to_360(arr[0][1])
                return lat, lon
            lat = float(arr[0])
            lon = to_360(arr[0])
            return lat, lon
        else:
            raise ValueError("latlon_idx array is empty")
    else:
        # Scalar
        logging.info(f"extract_latlon_tuple: scalar {latlon_idx}")
        lat = float(latlon_idx)
        lon = to_360(latlon_idx)
        return lat, lon


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes

from common_utils import get_weights, region_to_cartopy_extent, select_region


def _prepare_lon_for_plot(
    ds: Union[xr.Dataset, xr.DataArray], lon_name: str = "lon"
) -> Union[xr.Dataset, xr.DataArray]:
    """Return dataset with continuous, monotonic longitudes for contour plotting."""
    if lon_name not in ds.coords:
        return ds

    lons = np.asarray(ds[lon_name].values)
    if lons.ndim != 1 or lons.size < 2:
        return ds

    is_monotonic = bool(np.all(np.diff(lons) >= 0))
    if is_monotonic and not bool(ds.attrs.get("is_wraparound", False)):
        return ds

    out = ds.assign_coords({lon_name: ((ds[lon_name] + 180.0) % 360.0) - 180.0})
    out = out.sortby(lon_name)
    return out.assign_attrs({**out.attrs, "is_wraparound": False})


def plot_tropical_hurricane_track_2(
    ax: GeoAxes,
    ds: Union[xr.DataArray, Tuple[Union[xr.Dataset, xr.DataArray], ...]],
    search_region: Tuple[float, float, float, float],
    plot_region: Tuple[float, float, float, float],
    title: Optional[str] = None,
    plot_kwargs: Optional[dict] = None,
    extend: Optional[str] = None,
):
    """
    Improved hurricane track plotting:
    - Accepts single or multiple datasets
    - Accepts colormap or color for each dataset
    - Handles label externally (no legend inside)
    - Robust to single dataset/colormap
    """
    import matplotlib.cm as cm
    from matplotlib.colors import to_rgba

    def _extract_hurricane_centers(var, minlat, minlon, maxlat, maxlon, tol=20):
        _logger.info(
            f"Available lat: {var.lat.values}; available lon: {var.lon.values}"
        )
        _logger.info("Requesting slice({minlat}, {maxlat}), ({minlon}, {maxlon})")
        subregion = select_region(var, (minlat, minlon, maxlat, maxlon))
        min_coords = []
        for t in subregion.time:
            slice_t = subregion.sel(time=t, drop=True)
            stacked = slice_t.stack(points=("lat", "lon"))
            if stacked.size == 0:
                logging.debug("No points in subregion for time %s; skipping", t.values)
                continue
            # Find the index of the minimum value
            argmin_idx = (
                stacked.argmin("points").compute()
                if hasattr(stacked, "argmin")
                else stacked.argmin("points")
            )
            # Get the corresponding lat/lon
            latlon_idx = stacked.points[argmin_idx].values
            min_lat, min_lon = extract_latlon_tuple(latlon_idx)
            if min_coords:
                prev_lon, prev_lat = min_coords[-1]
                # Use shortest wrapped longitudinal distance to avoid false jumps across dateline
                lon_diff = ((min_lon - prev_lon + 540) % 360) - 180
                dist = np.sqrt(lon_diff**2 + (min_lat - prev_lat) ** 2)
                if dist > tol:
                    min_lon, min_lat = prev_lon, prev_lat
            min_coords.append((min_lon, min_lat))
        return min_coords

    # Accept single or tuple of datasets
    if isinstance(ds, (xr.DataArray, xr.Dataset)):
        ds = (ds,)

    n = len(ds)
    logging.info(f"plot_tropical_hurricane_track_2: called with {n} datasets")

    centers = []
    for i, d in enumerate(ds):
        logging.info(f"_extract_hurricane_centers: processing dataset {i}")
        if isinstance(d, xr.Dataset):
            var_name = list(d.data_vars)[0]
            d = d[var_name]
        raw_centers = _extract_hurricane_centers(
            d,
            search_region[0],
            search_region[1],
            search_region[2],
            search_region[3],
        )
        logging.info(
            f"_extract_hurricane_centers: found {len(raw_centers)} points for dataset {i}"
        )
        centers.append(raw_centers)

    # --- PATCH: Robust per-dataset plot_kwargs handling ---
    # plot_kwargs can be a single dict (applied to all) or a list of dicts (one per dataset)
    if plot_kwargs is None:
        plot_kwargs = {}
    if extend is not None:
        if isinstance(plot_kwargs, list):
            for d in plot_kwargs:
                d["extend"] = extend
        else:
            plot_kwargs["extend"] = extend

    # Helper to merge a list of dicts into a dict of lists
    def merge_kwargs_list(kwargs_list, n):
        merged = {}
        for i in range(n):
            for k, v in kwargs_list[i].items():
                merged.setdefault(k, [None] * n)
                merged[k][i] = v
        # Fill missing with None
        for k in merged:
            for i in range(n):
                if merged[k][i] is None:
                    merged[k][i] = merged[k][0]  # fallback to first value
        return merged

    # Accept both single dict or list of dicts for plot_kwargs
    if isinstance(plot_kwargs, list):
        # Merge into dict of lists for per-dataset kwargs
        merged_kwargs = merge_kwargs_list(plot_kwargs, n)

        def get_kw(key, default):
            v = merged_kwargs.get(key, [default] * n)
            # If not a list, broadcast
            if not isinstance(v, list):
                return [v] * n
            if len(v) < n:
                v = v + [v[0]] * (n - len(v))
            return v

        cmaps = merged_kwargs.get("cmap", True)
    else:
        # Single dict: fallback to old behavior
        def get_kw(key, default):
            v = plot_kwargs.get(key, default)
            if isinstance(v, list):
                return v
            return [v] * n

        cmaps = plot_kwargs.get("cmap", True)

    colormaps = get_kw("color", "coolwarm")
    markers = get_kw("marker", "o")
    linestyles = get_kw("linestyle", "-")
    linewidths = get_kw("linewidth", 1)
    alphas = get_kw("alpha", 1.0)
    marker_alphas = get_kw("marker_alpha", 1.0)
    marker_sizes = get_kw("markersize", None)
    marker_size_aliases = get_kw("marker_size", None)
    marker_sizes = [
        marker_sizes[i] if marker_sizes[i] is not None else marker_size_aliases[i]
        for i in range(n)
    ]
    # --- END PATCH ---

    # Always use PlateCarree(central_longitude=0) for 0–360° convention
    pc_crs = ccrs.PlateCarree(central_longitude=0)

    def to_minus180_180(lon):
        """Convert 0–360 longitude to -180 to 180."""
        return lon - 360 if lon > 180 else lon

    if isinstance(ax, GeoAxes):
        extent = region_to_cartopy_extent(plot_region)
        assert extent is not None
        logging.info(f"Setting plot extent: {extent}")
        ax.set_extent(extent, crs=pc_crs)
        ax.coastlines(linewidth=plot_kwargs.get("coastline_linewidth", 1))
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(
            cfeature.LAND, facecolor=plot_kwargs.get("land_color", "lightgray")
        )
        if plot_kwargs.get("grid", False):
            ax.gridlines(draw_labels=plot_kwargs.get("draw_labels", True))

    for i, center in enumerate(centers):
        # Skip series with no extracted points
        if not center:
            logging.warning("No track points extracted for dataset %s; skipping", i)
            continue

        # Unpack and convert lons from 0–360 to -180 to 180 for plotting
        lons, lats = zip(*center)
        lons = [to_minus180_180(lon) for lon in lons]
        logging.info(f"Plotting track lons: {lons}")
        if len(lons) == 0:
            logging.warning(
                f"No longitude coordinates extracted for center number {i+1}"
            )
            continue
        if plot_kwargs.get("smooth", False):
            from scipy.signal import savgol_filter

            lons = savgol_filter(
                lons, window_length=plot_kwargs.get("smoothing_window", 7), polyorder=3
            )
            lats = savgol_filter(
                lats, window_length=plot_kwargs.get("smoothing_window", 7), polyorder=3
            )
        if cmaps:
            # Use colormap for time progression
            color_spec = (
                cm.get_cmap(colormaps[i])
                if isinstance(colormaps[i], str)
                else colormaps[i]
            )
            col = [color_spec(j) for j in np.linspace(0, 1, len(lons) - 1)]
            colored_line_between_pts(
                lons,
                lats,
                col,
                ax,
                linestyle=linestyles[i],
                linewidth=linewidths[i],
                alpha=alphas[i],
                transform=pc_crs,
            )
            # plot markers separately to allow different alpha for markers
            try:
                marker_color = color_spec(0.5)
            except Exception:
                marker_color = color_spec(0.0)
            scatter_kwargs = {
                "marker": markers[i],
                "color": marker_color,
                "transform": pc_crs,
                "alpha": marker_alphas[i],
                "zorder": 3,
            }
            if marker_sizes[i] is not None:
                scatter_kwargs["s"] = marker_sizes[i]
            ax.scatter(
                [lons[-1]],
                [lats[-1]],
                **scatter_kwargs,
            )
        else:
            # plot line and markers separately so markers can have distinct alpha
            line_color = (
                colormaps[i] if isinstance(colormaps[i], str) else to_rgba(colormaps[i])
            )
            ax.plot(
                lons,
                lats,
                linestyle=linestyles[i],
                linewidth=linewidths[i],
                color=line_color,
                alpha=alphas[i],
                transform=pc_crs,
            )
            marker_kwargs = {
                "linestyle": "None",
                "marker": markers[i],
                "color": line_color,
                "alpha": marker_alphas[i],
                "transform": pc_crs,
            }
            if marker_sizes[i] is not None:
                marker_kwargs["markersize"] = marker_sizes[i]
            ax.plot(
                [lons[-1]],
                [lats[-1]],
                **marker_kwargs,
            )
    if title:
        ax.set_title(title)
    return ax, centers


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes

from common_utils import get_weights

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
    **kwargs,
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
        ds = select_region(ds, region)

    ds = _prepare_lon_for_plot(ds)

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
            contour_fontsize = kwargs.get(
                "contour_fontsize", kwargs.get("font_size", 8)
            )
            ax.clabel(
                cs,
                inline=True,
                fontsize=contour_fontsize,
                fmt="%d",
            )  # You can customize this

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
        ax.coastlines(linewidth=kwargs.get("coastline_linewidth", 1))
        extent = (
            region_to_cartopy_extent(region, ds)
            if region is not None
            else region_to_cartopy_extent((None, None, None, None), ds)
        )
        if extent is not None:
            ax.set_extent(extent, crs=projection)
        if kwargs.get("border", False):
            ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, facecolor=kwargs.get("land_color", "lightgray"))
        # Add US state boundaries
        try:
            ax.add_feature(cfeature.STATES, linewidth=0.7)
        except AttributeError:
            pass  # Some projections or Cartopy installs may not support STATES

    if title is not None:
        title_fontsize = kwargs.get("title_fontsize", kwargs.get("font_size", None))
        if title_fontsize is not None:
            ax.set_title(title, fontsize=title_fontsize)
        else:
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
    import matplotlib.cm as cm
    from matplotlib.colors import LinearSegmentedColormap

    def _extract_hurricane_centers(mslp, minlat, minlon, maxlat, maxlon, tol=20):
        assert mslp.lat.min() <= minlat
        assert mslp.lat.max() >= maxlat
        subregion = select_region(mslp, (minlat, minlon, maxlat, maxlon))
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
        extent = region_to_cartopy_extent(plot_region)
        assert extent is not None
        ax.set_extent(extent, crs=ccrs.PlateCarree())
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


# import xrft


def plot_PSD_Coh(
    ax: Union[Axes, GeoAxes],
    ds: xr.Dataset,
    ref_ds: xr.Dataset,
    normalizations: xr.Dataset,
    resolution: float = 0.25,
    per_variable_weighting: Optional[Dict[str, float]] = None,
    **plot_kwargs,
):

    if per_variable_weighting is None:
        per_variable_weighting = {
            "geopotential": 1.0,
            "specific_humidity": 1.0,
            "temperature": 1.0,
            "u_component_of_wind": 1.0,
            "v_component_of_wind": 1.0,
            "vertical_velocity": 1.0,
            "2m_temperature": 1.0,
            "10m_u_component_of_wind": 0.1,
            "10m_v_component_of_wind": 0.1,
            "mean_sea_level_pressure": 0.1,
            "total_precipitation_6hr": 0.0,
        }

    _, lvl_weights = get_weights(ds, resolution)

    # Normalize
    # mult by lat_weights is not done by authors
    ds = ds / normalizations
    ref_ds = ref_ds / normalizations

    # Code from ChatGPT
    per_var = []
    for var in ds.data_vars:
        iso_spec = xrft.isotropic_power_spectrum(
            ds[var], dim=["lon", "lat"], scaling="density", truncate=True
        )
        iso_spec_ref = xrft.isotropic_power_spectrum(
            ref_ds[var], dim=["lon", "lat"], scaling="density", truncate=True
        )
        iso_cross_spec = xrft.isotropic_cross_spectrum(
            ds[var], ref_ds[var], dim=["lon", "lat"], scaling="density", truncate=True
        )

        amp_ratio = np.sqrt(np.abs(iso_spec_ref)) / np.sqrt(
            np.abs(iso_spec)
        )  # Amplitude ratio
        coherence = np.real(iso_cross_spec) / np.sqrt(
            iso_spec * iso_spec_ref
        )  # according to paper

        if "level" in iso_spec.dims:
            amp_ratio = (amp_ratio * lvl_weights).mean(dim="level")
            coherence = (coherence * lvl_weights).mean(dim="level")

        if "time" in iso_spec.dims:
            amp_ratio = amp_ratio.mean(dim="time")
            coherence = coherence.mean(dim="time")

        print(f"{var} ... {coherence.mean().item()}")
        per_var.append(
            xr.Dataset(
                {
                    "amp": amp_ratio * per_variable_weighting.get(str(var), 1.0),
                    "coh": coherence * per_variable_weighting.get(str(var), 1.0),
                }
            )
        )

    total = xr.concat(per_var, dim="variable", join="exact").mean(
        "variable", skipna=False
    )

    # Plotting (PSD only)
    ax.plot(
        total["freq_r"],
        total["amp"],
        label=f"{plot_kwargs.get('label','')} PSD",
        color=plot_kwargs.get("color", "k"),
    )
    if plot_kwargs.get("log_axis", True):
        ax.set_xscale("log")  # optional
    # ax.set_xlim(1, total["freq_r"].max())
    # ax.set_ylim(0, 1.1)
    ax.set_xlabel(plot_kwargs.get("xlabel", "Total wavenumber"))
    ax.set_ylabel(plot_kwargs.get("ylabel", "Power spectral density"))
    if plot_kwargs.get("legend", True):
        ax.legend()

    return ax
