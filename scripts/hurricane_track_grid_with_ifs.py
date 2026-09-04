"""Same as hurricane_track_grid.py but with an IFS ensemble MSLP hull overlay.

The overlay draws the convex hull of mean_sea_level_pressure minima locations
from the IFS ensemble dataset, evaluated at each timestep within the map region.

Config: same JSON/YAML schema as hurricane_track_grid.py, with additional
optional fields defined in HurricaneTrackGridWithIfsConfig below.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, cast

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray
from cartopy.mpl.geoaxes import GeoAxes, config
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from pydantic import Field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hurricane_track_grid import (
    EXPERIMENT_CONFIG_FILE,
    HurricaneTrackGridConfig,
    TargetPoint,
    compute_target_points,
    plot_target_grid,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
_logger = logging.getLogger(__name__)

MSLP_VARIABLE = "mean_sea_level_pressure"


# ─── IFS loading (user-implemented) ─────────────────────────────────────────────


def load_ifs_ens_data(path: str, ftdt: datetime) -> xarray.Dataset:
    ds = xarray.load_dataset(path, engine="cfgrib")
    dts = [t + np.datetime64(ftdt) - np.timedelta64(6, "h") for t in ds.step.values]
    ds = ds.drop_vars(["time", "surface", "valid_time"])
    ds = ds.rename({"step": "time"})
    ds = ds.assign_coords(valid_time=dts)
    ds = ds.rename(
        {"msl": "mean_sea_level_pressure", "latitude": "lat", "longitude": "lon"}
    )
    ds = ds.sortby("lat")
    ds = ds.isel(time=1).expand_dims("time")
    return ds


# ─── Extended config ───────────────────────────────────────────────────────


class HurricaneTrackGridWithIfsConfig(HurricaneTrackGridConfig):
    overlay: str = Field(
        default="hull",
        description="IFS minima overlay: 'hull', 'density', or 'quantiles' (KDE quantile contours and minima scatter)",
    )
    minima_jitter_std: float = Field(
        default=0.05,
        description="Standard deviation of Gaussian noise added to IFS minima locations (degrees).",
    )
    minima_kde_bw: float = Field(
        default=0.2,
        description="Bandwidth for KDE of minima locations (scipy.stats.gaussian_kde 'bw_method').",
    )
    center_only_mode: bool = Field(
        default=False,
        description=(
            "If True, plot the normal target grid and map layout, but skip the "
            "IFS hull overlay."
        ),
    )
    center_color: str = Field(
        default="gray",
        description="Marker color for center_only_mode points.",
    )
    center_alpha: float = Field(
        default=0.4,
        description="Marker alpha for center_only_mode points.",
    )
    center_jitter_degrees: float = Field(
        default=0.0,
        description=(
            "Random jitter magnitude in degrees applied independently to lat/lon "
            "in center_only_mode. Set >0 to separate overlapping points."
        ),
    )
    center_jitter_seed: Optional[int] = Field(
        default=0,
        description="Random seed for jitter in center_only_mode. Use None for non-deterministic jitter.",
    )
    center_marker_size: Optional[float] = Field(
        default=None,
        description="Optional marker size override for center_only_mode.",
    )
    ifs_ens_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to IFS ensemble dataset. "
            "If None, the MSLP hull overlay is skipped."
        ),
    )
    hull_color: str = Field(
        default="cyan",
        description="Color of the MSLP hull overlay.",
    )
    hull_linewidth: float = Field(
        default=1.5,
        description="Line width of the hull polygon edge.",
    )
    hull_alpha: float = Field(
        default=0.8,
        description="Alpha of the hull polygon edge.",
    )
    hull_fill_alpha: float = Field(
        default=0.1,
        description="Alpha of the hull polygon fill. Set to 0 for no fill.",
    )
    hull_label: str = Field(
        default="IFS MSLP hull",
        description="Legend label for the hull overlay.",
    )


def load_config(config_path: str) -> HurricaneTrackGridWithIfsConfig:
    if config_path.endswith(".json"):
        with open(config_path) as f:
            data = json.load(f)
    else:
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f)
    return HurricaneTrackGridWithIfsConfig.model_validate(data)


# ─── Convex hull (pure numpy, Andrew's monotone chain) ───────────────────────


def _cross(o: tuple, a: tuple, b: tuple) -> float:
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Compute the convex hull of *points* (N×2, columns [lon, lat]) using
    Andrew's monotone chain algorithm.

    Returns hull vertices in counter-clockwise order as (M×2) array.
    Returns all points unchanged for N ≤ 2.
    """
    pts = sorted(set(map(tuple, points)))  # deduplicate and sort (lon, lat)
    n = len(pts)
    if n <= 2:
        return np.array(pts)

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1])


# ─── MSLP minima hull ──────────────────────────────────────────────────────


def _map_region(
    config: HurricaneTrackGridConfig,
    points: list[TargetPoint],
) -> tuple[float, float, float, float]:
    """
    Return (min_lat, min_lon, max_lat, max_lon) for the plot region.
    Uses map_extent if configured; otherwise derives from target points with a buffer.
    """
    if config.map_extent is not None:
        return config.map_extent  # (min_lat, min_lon, max_lat, max_lon)
    lats = [p.target_lat for p in points]
    lons = [p.target_lon for p in points]
    buf = 5.0
    return min(lats) - buf, min(lons) - buf, max(lats) + buf, max(lons) + buf


def mslp_minima_hull(
    ifs_ds: xarray.Dataset,
    region: tuple[float, float, float, float],
) -> Optional[np.ndarray]:
    """
    For each (time, ensemble member) combination in *ifs_ds*, find the (lat, lon)
    of the MSLP minimum within *region* = (min_lat, min_lon, max_lat, max_lon).

    The dataset is expected to have dimensions (number, time, lat, lon).

    Returns the convex hull of all those minima as an (N×2) array of [lon, lat]
    columns, or None if no valid points are found.
    """
    min_lat, min_lon, max_lat, max_lon = region

    da = ifs_ds[MSLP_VARIABLE]
    lat_mask = (da.lat >= min_lat) & (da.lat <= max_lat)
    lon_mask = (da.lon >= min_lon) & (da.lon <= max_lon)
    da_region = da.where(lat_mask & lon_mask)

    lat_vals = da_region.lat.values  # 1-D
    lon_vals = da_region.lon.values  # 1-D
    nx = len(lon_vals)

    minima: list[tuple[float, float]] = []
    for member in da_region.number:
        for t in da_region.time:
            arr = da_region.sel(number=member, time=t).values  # 2-D (lat × lon)
            if np.all(np.isnan(arr)):
                continue
            lat_i, lon_i = divmod(int(np.nanargmin(arr)), nx)
            minima.append((float(lon_vals[lon_i]), float(lat_vals[lat_i])))
    if not minima:
        _logger.warning("No MSLP minima found within the plot region.")
        return None
    return np.array(minima)


# ─── IFS minima quantile contours and scatter ─────────────────────────────


def plot_ifs_minima_quantiles(
    # (function unchanged)
    ax_geo: GeoAxes,
    minima: np.ndarray,
    region: tuple[float, float, float, float],
    config: HurricaneTrackGridWithIfsConfig,
) -> None:
    """Plot IFS minima as black dots (with jitter) and quantile contours (0.25, 0.75, 0.9) using KDE."""
    from scipy.stats import gaussian_kde

    # Add Gaussian noise for clarity
    rng = np.random.default_rng(42)
    jittered = minima + rng.normal(0, config.minima_jitter_std, minima.shape)
    # Plot black dots
    ax_geo.scatter(
        jittered[:, 0],
        jittered[:, 1],
        color="black",
        s=18,
        alpha=0.7,
        zorder=6,
        transform=ccrs.PlateCarree(),
        label="IFS minima",
    )
    # KDE
    kde = gaussian_kde(jittered.T, bw_method=config.minima_kde_bw)
    min_lat, min_lon, max_lat, max_lon = region
    xgrid = np.linspace(min_lon, max_lon, 200)
    ygrid = np.linspace(min_lat, max_lat, 200)
    xx, yy = np.meshgrid(xgrid, ygrid)
    coords = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(coords).reshape(xx.shape)
    # Compute contour levels for quantiles
    flat = zz.flatten()
    idx = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[idx])
    cumsum /= cumsum[-1]

    def find_level(q):
        # Find the density threshold for quantile q
        i = np.searchsorted(cumsum, q)
        return flat[idx[i]]

    levels = [find_level(q) for q in (0.25, 0.75, 0.9)]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green
    labels = ["0.25 quantile", "0.75 quantile", "0.9 quantile"]
    # Sort levels and reorder colors/labels accordingly
    zipped = sorted(zip(levels, colors, labels))
    sorted_levels, sorted_colors, sorted_labels = zip(*zipped)
    cs = ax_geo.contour(
        xx,
        yy,
        zz,
        levels=sorted_levels,
        colors=sorted_colors,
        linewidths=2,
        alpha=0.8,
        transform=ccrs.PlateCarree(),
        zorder=7,
    )
    # Robustly label contours, fallback if not enough collections
    if hasattr(cs, "collections") and len(cs.collections) >= len(sorted_labels):
        for i, l in enumerate(sorted_labels):
            cs.collections[i].set_label(l)
    else:
        # Fallback: add proxy artists for legend if contours missing
        from matplotlib.lines import Line2D

        proxy_lines = [
            Line2D([0], [0], color=sorted_colors[i], lw=2)
            for i in range(len(sorted_labels))
        ]
        ax_geo.legend(proxy_lines, sorted_labels, loc="lower left")


# ─── MSLP minima density ──────────────────────────────────────────────────────


def plot_mslp_minima_density(
    ax_geo: GeoAxes,
    ifs_ds: xarray.Dataset,
    region: tuple[float, float, float, float],
    config: HurricaneTrackGridWithIfsConfig,
) -> None:
    """Plot a 2D density of MSLP minima locations on ax_geo."""
    from scipy.stats import gaussian_kde

    min_lat, min_lon, max_lat, max_lon = region
    da = ifs_ds[MSLP_VARIABLE]
    lat_mask = (da.lat >= min_lat) & (da.lat <= max_lat)
    lon_mask = (da.lon >= min_lon) & (da.lon <= max_lon)
    da_region = da.where(lat_mask & lon_mask)
    lat_vals = da_region.lat.values
    lon_vals = da_region.lon.values
    nx = len(lon_vals)
    minima = []
    for member in da_region.number:
        for t in da_region.time:
            arr = da_region.sel(number=member, time=t).values
            if np.all(np.isnan(arr)):
                continue
            lat_i, lon_i = divmod(int(np.nanargmin(arr)), nx)
            minima.append((float(lon_vals[lon_i]), float(lat_vals[lat_i])))
    if not minima:
        _logger.warning("No MSLP minima found for density plot.")
        return
    minima = np.array(minima)
    # 2D KDE
    kde = gaussian_kde(minima.T)
    # Grid for density
    xgrid = np.linspace(min_lon, max_lon, 200)
    ygrid = np.linspace(min_lat, max_lat, 200)
    xx, yy = np.meshgrid(xgrid, ygrid)
    coords = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(coords).reshape(xx.shape)
    im = ax_geo.pcolormesh(
        xgrid,
        ygrid,
        zz,
        cmap="Blues",
        shading="auto",
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    cbar = plt.colorbar(im, ax=ax_geo, orientation="vertical", pad=0.03, shrink=0.92)
    cbar.set_label("Density of IFS MSLP minima")


# ─── Overlay drawing ─────────────────────────────────────────────────────────


def overlay_mslp_hull(
    ax_geo: GeoAxes,
    hull_pts: np.ndarray,
    config: HurricaneTrackGridWithIfsConfig,
) -> None:
    """Draw the MSLP hull polygon (or degenerate cases) on *ax_geo*."""
    transform = ccrs.PlateCarree()
    n = len(hull_pts)

    if n == 1:
        ax_geo.scatter(
            hull_pts[0, 0],
            hull_pts[0, 1],
            color=config.hull_color,
            s=80,
            zorder=5,
            transform=transform,
            label=config.hull_label,
        )
    elif n == 2:
        ax_geo.plot(
            hull_pts[:, 0],
            hull_pts[:, 1],
            color=config.hull_color,
            linewidth=config.hull_linewidth,
            alpha=config.hull_alpha,
            transform=transform,
            zorder=5,
            label=config.hull_label,
        )
    else:
        closed = np.vstack([hull_pts, hull_pts[0]])  # close the polygon
        ax_geo.plot(
            closed[:, 0],
            closed[:, 1],
            color=config.hull_color,
            linewidth=config.hull_linewidth,
            alpha=config.hull_alpha,
            transform=transform,
            zorder=5,
            label=config.hull_label,
        )
        if config.hull_fill_alpha > 0:
            patch = MplPolygon(
                hull_pts,
                closed=True,
                facecolor=config.hull_color,
                edgecolor="none",
                alpha=config.hull_fill_alpha,
                transform=transform,
                zorder=4,
            )
            ax_geo.add_patch(patch)


def plot_centers_only(
    points: list[TargetPoint],
    config: HurricaneTrackGridWithIfsConfig,
) -> tuple[Figure, GeoAxes, Path]:
    """Plot only target centers in a single gray, optionally jittered style."""
    fig, ax = plt.subplots(
        figsize=(10, 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax_geo = cast(GeoAxes, ax)

    lats = np.array([p.target_lat for p in points], dtype=float)
    lons = np.array([p.target_lon for p in points], dtype=float)

    if config.center_jitter_degrees > 0:
        rng = np.random.default_rng(config.center_jitter_seed)
        jitter = rng.uniform(
            low=-config.center_jitter_degrees,
            high=config.center_jitter_degrees,
            size=(len(points), 2),
        )
        lons = lons + jitter[:, 0]
        lats = lats + jitter[:, 1]

    ax_geo.scatter(
        lons,
        lats,
        color=config.center_color,
        s=(
            config.center_marker_size
            if config.center_marker_size is not None
            else config.marker_size
        ),
        alpha=config.center_alpha,
        edgecolors="none",
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    ax_geo.coastlines(resolution="50m", linewidth=0.8)
    # set land color to grey
    from cartopy import feature as cfeature

    ax_geo.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    gl = ax_geo.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    if config.map_extent is not None:
        min_lat, min_lon, max_lat, max_lon = config.map_extent
        ax_geo.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    ax_geo.set_title(config.title)

    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return fig, ax_geo, output_path


def plot_target_centers(
    points: list[TargetPoint], config: HurricaneTrackGridWithIfsConfig
) -> tuple[Figure, GeoAxes, Path]:
    """Plot target locations as black crosses on a map using the same layout."""
    fig, ax = plt.subplots(
        figsize=(10, 7), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax_geo = cast(GeoAxes, ax)

    lats = [p.target_lat for p in points]
    lons = [p.target_lon for p in points]

    ax_geo.scatter(
        lons,
        lats,
        color="black",
        marker="x",
        s=config.marker_size,
        alpha=config.marker_alpha,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    if config.annotate_points:
        for p in points:
            ax_geo.text(
                p.target_lon,
                p.target_lat,
                p.folder_name,
                fontsize=8,
                transform=ccrs.PlateCarree(),
                zorder=6,
            )

    ax_geo.coastlines(resolution="50m", linewidth=0.8)
    gl = ax_geo.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    if config.map_extent is not None:
        min_lat, min_lon, max_lat, max_lon = config.map_extent
        ax_geo.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    ax_geo.set_title(config.title)

    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return fig, ax_geo, output_path


def main_centers():
    parser = argparse.ArgumentParser(
        description=(
            "Plot all target locations from target_i experiment folders as "
            "black crosses on the configured map."
        )
    )
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    points = compute_target_points(config)
    fig, ax_geo, output_path = plot_target_centers(points, config)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Saved plot to %s", output_path)


# ─── Main ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Same as hurricane_track_grid.py but overlays the convex hull of "
            "IFS ensemble MSLP minima on the map."
        )
    )
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config")
    args = parser.parse_args()

    config = load_config(args.config)

    points = compute_target_points(config)
    fig, ax_geo, output_path = plot_target_grid(points, config)

    ax_geo.scatter(
        -82,
        26.75,
        color="black",
        s=20,
        marker="x",
        transform=ccrs.PlateCarree(),
        zorder=6,
        label="HRES center",
    )

    if config.ifs_ens_path is not None:
        assert config.analysis_first_target_datetime is not None
        ifs_ds = load_ifs_ens_data(
            config.ifs_ens_path, config.analysis_first_target_datetime
        )
        if ifs_ds is not None:
            region = _map_region(config, points)
            minima = mslp_minima_hull(ifs_ds, region)
            if minima is not None:
                if config.overlay == "hull":
                    if config.center_only_mode:
                        _logger.info("center_only_mode enabled; skipping hull overlay.")
                    else:
                        hull_pts = convex_hull(minima)
                        overlay_mslp_hull(ax_geo, hull_pts, config)
                        ax_geo.legend(loc="lower left")
                elif config.overlay == "density":
                    plot_mslp_minima_density(ax_geo, ifs_ds, region, config)
                elif config.overlay == "quantiles":
                    plot_ifs_minima_quantiles(ax_geo, minima, region, config)
                    ax_geo.legend(loc="lower left")
                else:
                    _logger.warning(f"Unknown overlay option: {config.overlay}")
            else:
                _logger.warning("No IFS minima found for overlay plot.")
        else:
            _logger.warning("load_ifs_ens_data returned None; skipping overlay.")
    else:
        _logger.info("No ifs_ens_path configured; skipping IFS overlay.")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Saved plot to %s", output_path)


def main_2():
    # Example config path (replace with your actual config)
    config_path = "/home/users/f/froelicm/WeatherPlots/configs/hurricane_grid/ian8step_penalized.json"
    config = load_config(config_path)
    assert config.ifs_ens_path is not None
    assert config.analysis_first_target_datetime is not None

    # Load IFS ensemble data
    ifs_ds = load_ifs_ens_data(
        config.ifs_ens_path, config.analysis_first_target_datetime
    )
    ifs_ds = ifs_ds.sel({"lon": slice(270, 290), "lat": slice(23, 31)})
    # Select first 5 ensemble members
    ifs_ds_5 = ifs_ds.sel(number=slice(0, 4))

    # Plot tracks for each member
    fig, ax = plt.subplots(
        figsize=(10, 7), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    for member in ifs_ds_5.number.values:
        mslp = ifs_ds_5[MSLP_VARIABLE].sel(number=member)
        # Find min location at each time
        lats, lons = [], []
        if "time" not in mslp.dims:
            arr = mslp.values
            if np.all(np.isnan(arr)):
                continue
            lat_vals = mslp.lat.values
            lon_vals = mslp.lon.values
            nx = len(lon_vals)
            lat_i, lon_i = divmod(int(np.nanargmin(arr)), nx)
            lats.append(float(lat_vals[lat_i]))
            lons.append(float(lon_vals[lon_i]) - 360)
        else:
            for t in mslp.time:
                arr = mslp.sel(time=t).values
                if np.all(np.isnan(arr)):
                    continue
                lat_vals = mslp.lat.values
                lon_vals = mslp.lon.values
                nx = len(lon_vals)
                lat_i, lon_i = divmod(int(np.nanargmin(arr)), nx)
                lats.append(float(lat_vals[lat_i]))
                lons.append(float(lon_vals[lon_i]) - 360)
        print(f"Member {member}: track points: {list(zip(lats, lons))}")
        ax.plot(lons, lats, marker="o", label=f"Member {member}")

    # Set extent from config if present
    if getattr(config, "map_extent", None) is not None:
        min_lat, min_lon, max_lat, max_lon = config.map_extent
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    ax.coastlines(resolution="50m", linewidth=0.8)
    ax.legend()
    ax.set_title("Hurricane tracks for first 5 ensemble members")
    plt.savefig("ifs_test.png", dpi=200, bbox_inches="tight")
    plt.close()


def main_3():
    """
    Plots mean_sea_level_pressure as a heatmap for the second timestep and a single ensemble member.
    Uses config_path as in main_2().
    """
    config_path = "/home/users/f/froelicm/WeatherPlots/configs/hurricane_grid/ian8step_penalized.json"
    config = load_config(config_path)
    assert config.ifs_ens_path is not None
    assert config.analysis_first_target_datetime is not None

    # Load IFS ensemble data
    ifs_ds = load_ifs_ens_data(
        config.ifs_ens_path, config.analysis_first_target_datetime
    )

    # Select second timestep (index 1) and first ensemble member (index 0)
    member = ifs_ds.number.values[0]
    time = ifs_ds.time.values[1]
    mslp = ifs_ds[MSLP_VARIABLE].sel(number=member, time=time)

    fig, ax = plt.subplots(
        figsize=(10, 7), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    pcm = ax.pcolormesh(ifs_ds.lon, ifs_ds.lat, mslp, cmap="viridis")
    plt.colorbar(pcm, ax=ax, label="Mean Sea Level Pressure (Pa)")

    # Set extent from config if present
    if getattr(config, "map_extent", None) is not None:
        min_lat, min_lon, max_lat, max_lon = config.map_extent
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    ax.coastlines(resolution="50m", linewidth=0.8)
    ax.set_title(f"MSLP Heatmap\nMember {member}, Time {str(time)}")
    plt.savefig("ifs_mslp_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
