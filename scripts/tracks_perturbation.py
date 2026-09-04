import os
import sys
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize

# Ensure project root is importable when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_utils import get_dim_weights, normalize_dataset, prep_data

# -------------------- USER CONFIGURABLE DEFAULTS --------------------
# List of parent directories to process, with optional plotting kwargs
DATASETS = [
    {
        "path": "/home/users/f/froelicm/scratch/output/ian_hresf0_rerun/12step_0001/minus0.5",
        "linestyle": "-",
        "linewidth": 5,
    },
    {
        "path": "/home/users/f/froelicm/scratch/output/ian_hresf0_rerun/12step_0001/minus2.0",
        "linestyle": "-",
        "linewidth": 5,
    },
    {
        "path": "/home/users/f/froelicm/scratch/output/ian_hresf0_rerun/12step_0001/minus3.0/",
        "linestyle": "-",
        "linewidth": 2,
    },
    {
        "path": "/home/users/f/froelicm/scratch/output/ian_hresf0_rerun/12step_0001/plus0.5/",
        "linestyle": "-",
        "linewidth": 2,
    },
    {
        "path": "/home/users/f/froelicm/scratch/output/ian_hresf0_rerun/12step_0001/plus1.0/",
        "linestyle": "-",
        "linewidth": 2,
    },
    {
        "path": "/home/users/f/froelicm/scratch/output/ian_hresf0_rerun/12step_0001/plus2.0",
        "linestyle": "-",
        "linewidth": 2,
    },
    {
        "path": "/home/users/f/froelicm/scratch/hresf0-fc0/weatherbench2_hres_t0_2022_09.nc",
        "color": "#000000",
        "linestyle": "-",
        "linewidth": 4,
        "marker": "D",
        "timeframe": [
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
        ],
        "label": "HRES analysis",
    },
    {
        "label": "GraphCast-AMSE",
        "path": "/home/users/f/froelicm/scratch/output/ian_hresf0_rerun/12step_0001/minus0.5/output_epoch-0.nc",
        "color": "#000000",
        "linestyle": "--",
        "linewidth": 4,
    },
]

# Search region for extracting hurricane centers (minlat, minlon, maxlat, maxlon)
SEARCH_REGION = (15, -90, 35, -60)

# Plot region for map display (minlat, minlon, maxlat, maxlon)
PLOT_REGION = (15, -90, 35, -60)

# Colormap and levels for time progression (if used)
COLORMAP = "viridis"
COLORMAP_LEVELS = 10
COLORBAR_FRACTION = 0.07  # Fraction of axis width/height
COLORBAR_LOCATION = "right"  # 'right', 'bottom', etc.

# Legend placement
LEGEND_LOC = "lower right"

# Figure output path
FIGURE_PATH = "hurricane_tracks.png"

# Figure size (in inches)
FIGSIZE = (8, 6)

# -------------------------------------------------------------------


def mse_loss(
    prediction: xr.Dataset,
    analysis: xr.Dataset,
    weights_latitude: xr.DataArray,
    weights_per_level: xr.DataArray,
    weights_per_variable: dict[str, float],
    scales: Optional[xr.Dataset] = None,
    locations: Optional[xr.Dataset] = None,
) -> tuple[float, xr.Dataset]:
    """Compute weighted MSE loss between *prediction* and *analysis*.

    Both datasets are first **normalized** (if *scales* / *locations* are
    provided), then the MSE is computed exactly as in GraphCast:

    1. ``(diff**2).mean(dim='lon') * lat_weights``  → area-weight
    2. ``.mean(dim='lat') * level_weights``          → level-weight
    3. ``.sum(dim='level')``
    4. ``.mean(dim='time')``                         → average over time
    5. Multiply by per-variable weight and sum.

    Returns ``(total_loss, mse_by_variable)`` where *mse_by_variable* is an
    ``xr.Dataset`` with one scalar per variable (before variable weighting).
    """
    # Determine which variables to include
    all_vars = list(analysis.data_vars)
    if weights_per_variable:
        subset = [v for v in all_vars if weights_per_variable.get(v, 0) > 0]
    else:
        subset = all_vars  # all with weight 1

    # Align
    analysis, prediction = xr.align(analysis, prediction, join="inner")
    analysis = analysis[subset]
    prediction = prediction[subset]

    # Normalize
    analysis = normalize_dataset(analysis, scales, locations)
    prediction = normalize_dataset(prediction, scales, locations)

    diffs = analysis - prediction

    mse_per_var = {}
    total = 0.0
    for var in subset:
        d = diffs[var]
        var_dims = set(d.dims)
        mse = d**2
        if "lon" in var_dims:
            mse = mse.mean(dim="lon") * weights_latitude
        if "lat" in var_dims:
            mse = mse.mean(dim="lat")
        if "level" in var_dims:
            mse = (mse * weights_per_level).sum(dim="level")
        if "time" in mse.dims:
            mse = mse.mean(dim="time")
        # Also average over batch if present
        if "batch" in mse.dims:
            mse = mse.mean(dim="batch")
        # Collapse any remaining dimensions
        if hasattr(mse, "dims") and len(mse.dims) > 0:
            mse = mse.mean()
        scalar = float(mse.values)
        mse_per_var[var] = scalar
        w = weights_per_variable.get(var, 0.0)
        total += scalar * w

    mse_ds = xr.Dataset({v: xr.DataArray(val) for v, val in mse_per_var.items()})
    print(f"Computed MSE loss: total={total:.6f}")
    return total, mse_ds


def extract_hurricane_centers(mslp, search_region, tol=20):
    """Extract hurricane centers (min pressure) at each time step within search_region."""
    minlat, minlon, maxlat, maxlon = search_region
    # Subset region
    lats = mslp["lat"]
    lons = mslp["lon"]
    region = mslp.sel(
        lat=slice(minlat, maxlat) if lats[0] < lats[-1] else slice(maxlat, minlat),
        lon=slice(minlon, maxlon) if lons[0] < lons[-1] else slice(maxlon, minlon),
    )
    centers = []
    for t in region["time"]:
        slice_t = region.sel(time=t)
        stacked = slice_t.stack(points=("lat", "lon"))
        if stacked.size == 0:
            centers.append((np.nan, np.nan))
            continue
        argmin_idx = stacked.argmin("points").item()
        lat = float(stacked["lat"].values[argmin_idx])
        lon = float(stacked["lon"].values[argmin_idx])
        # Optionally: filter out jumps
        if centers:
            prev_lat, prev_lon = centers[-1]
            if not np.isnan(prev_lon):
                lon_diff = ((lon - prev_lon + 540) % 360) - 180
                dist = np.sqrt(lon_diff**2 + (lat - prev_lat) ** 2)
                if dist > tol:
                    lon, lat = prev_lon, prev_lat
        centers.append((lat, lon))
    return np.array(centers)


def in_plot_region(lat, lon, plot_region):
    minlat, minlon, maxlat, maxlon = plot_region
    if maxlon < minlon:
        in_lon = lon >= minlon or lon <= maxlon
    else:
        in_lon = minlon <= lon <= maxlon
    return (minlat <= lat <= maxlat) and in_lon


def _get_max_epoch_in_dir(
    dir: str,
    startswith: str = "output_epoch-",
    endswith: str = ".nc",
) -> int:
    files = os.listdir(dir)

    max_epoch = 0
    for f in files:
        if f.startswith(startswith) and f.endswith(endswith):
            epoch_str = f[len(startswith) : -len(endswith)]
            epoch = int(epoch_str)
            if epoch > max_epoch:
                max_epoch = epoch
    return max_epoch


def main():
    fig = plt.figure(figsize=FIGSIZE)
    ax = plt.axes(projection=ccrs.PlateCarree())
    # Cartopy GeoAxes methods (may show false errors in some editors)
    ax.set_extent(
        [PLOT_REGION[1], PLOT_REGION[3], PLOT_REGION[0], PLOT_REGION[2]],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="lightgray")

    # For colorbar if using time progression
    norm = Normalize(0, COLORMAP_LEVELS - 1)
    cmap = plt.get_cmap(COLORMAP)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    perturb_sizes = []
    dir_entries = []
    file_entries = []
    for spec in DATASETS:
        path = spec["path"]
        if os.path.isdir(path):
            dir_entries.append(spec)
        elif os.path.isfile(path):
            file_entries.append(spec)
        else:
            print(f"Warning: {path} is not a valid file or directory, skipping.")

    # Handle directories (perturbation logic)
    for ds_idx, spec in enumerate(dir_entries):
        parent_dir = spec["path"]
        max_epoch = _get_max_epoch_in_dir(parent_dir)
        input0_path = os.path.join(parent_dir, f"input_epoch-0.nc")
        inputN_path = os.path.join(parent_dir, f"input_epoch-{max_epoch}.nc")
        outputN_path = os.path.join(parent_dir, f"output_epoch-{max_epoch}.nc")

        # Load input datasets
        input0 = xr.open_dataset(input0_path)
        inputN = xr.open_dataset(inputN_path)

        # Compute un-normalized max/min/mean abs diffs for each field
        stats = {}
        for var in input0.data_vars:
            arr0 = input0[var].values
            arrN = inputN[var].values
            absdiff = np.abs(arrN - arr0)
            if arr0.ndim == 3:  # (time, level, lat/lon) or (level, lat, lon)
                for i in range(arr0.shape[1]):
                    key = f"{var}_lev{i}" if arr0.shape[1] > 1 else var
                    absdiff_i = (
                        absdiff[:, i]
                        if absdiff.shape[1] == arr0.shape[1]
                        else absdiff[i]
                    )
                    stats[key] = {
                        "max": float(np.nanmax(absdiff_i)),
                        "min": float(np.nanmin(absdiff_i)),
                        "mean": float(np.nanmean(absdiff_i)),
                    }
            else:
                stats[var] = {
                    "max": float(np.nanmax(absdiff)),
                    "min": float(np.nanmin(absdiff)),
                    "mean": float(np.nanmean(absdiff)),
                }
        print(f"Un-normalized abs diff stats for {parent_dir}:")
        print(stats)

        # Compute perturbation size (MSE loss, normalized)
        weights_lat, weights_lev = get_dim_weights(input0)
        weights_per_var = {v: 1.0 for v in input0.data_vars}
        input0_norm = normalize_dataset(input0, None, None)
        inputN_norm = normalize_dataset(inputN, None, None)
        # Ensure weights_per_var keys are str
        weights_per_var_str = {str(k): float(v) for k, v in weights_per_var.items()}
        pert_size, _ = mse_loss(
            inputN_norm, input0_norm, weights_lat, weights_lev, weights_per_var_str
        )
        perturb_sizes.append(pert_size)

    # Normalize perturbation sizes for colormap (for directories)
    pert_array = np.array(perturb_sizes) if perturb_sizes else np.array([0.0])
    vmin = pert_array.min()
    vmax = pert_array.max()
    norm = Normalize(vmin, vmax)
    cmap = plt.get_cmap(COLORMAP)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Plot tracks for directories
    for ds_idx, spec in enumerate(dir_entries):
        parent_dir = spec["path"]
        max_epoch = _get_max_epoch_in_dir(parent_dir)
        outputN_path = os.path.join(parent_dir, f"output_epoch-{max_epoch}.nc")
        ds = xr.open_dataset(outputN_path)
        if "mean_sea_level_pressure" in ds:
            mslp = ds["mean_sea_level_pressure"]
        else:
            mslp = list(ds.data_vars.values())[0]
        centers = extract_hurricane_centers(mslp, SEARCH_REGION)
        # Filter to plot region, break line at out-of-bounds
        lats, lons = [], []
        for lat, lon in centers:
            if np.isnan(lat) or np.isnan(lon):
                lats.append(np.nan)
                lons.append(np.nan)
            elif in_plot_region(lat, lon, PLOT_REGION):
                lats.append(lat)
                lons.append(lon)
            else:
                lats.append(np.nan)
                lons.append(np.nan)
        # Color by perturbation size
        pert_size = perturb_sizes[ds_idx] if ds_idx < len(perturb_sizes) else 0.0
        color = cmap(norm(pert_size))
        # Plot track
        ax.plot(
            lons,
            lats,
            color=color,
            label=spec.get("label", f"Track {ds_idx+1}"),
            linewidth=spec.get("linewidth", 1),
            linestyle=spec.get("linestyle", "-"),
            marker=spec.get("marker", None),
        )
        # Optionally: plot start/end markers
        if lons and lats and not np.isnan(lons[0]) and not np.isnan(lats[0]):
            ax.plot(lons[0], lats[0], marker="o", color=color)
        if lons and lats and not np.isnan(lons[-1]) and not np.isnan(lats[-1]):
            ax.plot(lons[-1], lats[-1], marker="s", color=color)

    # Plot tracks for direct files (no perturbation coloring)
    for spec in file_entries:
        file_path = spec["path"]
        ds = xr.open_dataset(file_path)
        if "mean_sea_level_pressure" in ds:
            mslp = ds["mean_sea_level_pressure"]
        else:
            mslp = list(ds.data_vars.values())[0]
        centers = extract_hurricane_centers(mslp, SEARCH_REGION)
        lats, lons = [], []
        for lat, lon in centers:
            if np.isnan(lat) or np.isnan(lon):
                lats.append(np.nan)
                lons.append(np.nan)
            elif in_plot_region(lat, lon, PLOT_REGION):
                lats.append(lat)
                lons.append(lon)
            else:
                lats.append(np.nan)
                lons.append(np.nan)
        ax.plot(
            lons,
            lats,
            color=spec.get("color", None),
            label=spec.get("label", os.path.basename(file_path)),
            linewidth=spec.get("linewidth", 1),
            linestyle=spec.get("linestyle", "-"),
            marker=spec.get("marker", None),
        )
        if lons and lats and not np.isnan(lons[0]) and not np.isnan(lats[0]):
            ax.plot(lons[0], lats[0], marker="o", color=spec.get("color", None))
        if lons and lats and not np.isnan(lons[-1]) and not np.isnan(lats[-1]):
            ax.plot(lons[-1], lats[-1], marker="s", color=spec.get("color", None))
    # ax.set_title('')

    plt.savefig(FIGURE_PATH, bbox_inches="tight", dpi=150)
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
