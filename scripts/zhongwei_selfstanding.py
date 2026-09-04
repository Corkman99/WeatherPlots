"""Minimal standalone GRIB map plot example.

This script intentionally avoids imports from the local WeatherPlots package.
It demonstrates a single-map contourf + contour plot from a GRIB file.

Suggested venv packages:
  pip install numpy xarray matplotlib cartopy cfgrib eccodes
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes

# -------------------------------
# User configuration constants
# -------------------------------
DATA_PATH = Path("")
OUTPUT_PATH = Path("./fig.png")

PROJECTION_TYPE = "PlateCarree"  # "NearsidePerspective"
SATELLITE_HEIGHT = 35_786_000

# Region format: (minlat, minlon, maxlat, maxlon)
REGION = (25, 340.0, 60.0, 40.0)

FIGSIZE = (18, 12)

TARGET_TIME_INDEX = 0
TARGET_LEVEL_HPA = 850.0

FCONTOUR_VARIABLE = "temperature"
FCONTOUR_LEVELS = np.array([x + 273.15 for x in np.arange(-10, 30, 2)])
FCONTOUR_CMAP = "Spectral_r"
FCONTOUR_EXTEND = "max"
COLORBAR_LABEL = "2m temperature (K)"
ADD_COLORBAR = False
ADD_LAT_LON_ANNOTATIONS = False

CONTOUR_VARIABLE = "geopotential"
CONTOUR_MINOR_LEVELS = np.arange(13000, 16500, 200)
CONTOUR_MAJOR_LEVELS = np.arange(13000, 16500, 400)
CONTOUR_COLORS = "black"
CONTOUR_MINOR_LINEWIDTH = 0.7
CONTOUR_MAJOR_LINEWIDTH = 1.2
CONTOUR_MINOR_ALPHA = 0.65


def to_lon_360(lon: np.ndarray) -> np.ndarray:
    return np.mod(lon, 360.0)


def to_lon_180(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def wrap_center_lon_180(minlon: float, maxlon: float) -> float:
    if minlon <= maxlon:
        center_360 = (minlon + maxlon) / 2.0
    else:
        center_360 = (minlon + (maxlon + 360.0)) / 2.0
        center_360 = center_360 % 360.0
    return to_lon_180(center_360)


def create_projection(
    projection_type: str, region: tuple[float, float, float, float]
) -> ccrs.Projection:
    minlat, minlon, maxlat, maxlon = region
    if projection_type == "NearsidePerspective":
        central_lat = (minlat + maxlat) / 2.0
        central_lon = wrap_center_lon_180(minlon, maxlon)
        return ccrs.NearsidePerspective(
            central_latitude=central_lat,
            central_longitude=central_lon,
            satellite_height=SATELLITE_HEIGHT,
        )
    return ccrs.PlateCarree()


def set_standard_names(ds: xr.Dataset) -> xr.Dataset:
    rename_map: dict[str, str] = {}

    if "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    if "isobaricInhPa" in ds.coords:
        rename_map["isobaricInhPa"] = "level"

    if "t" in ds.data_vars:
        rename_map["t"] = "temperature"
    if "z" in ds.data_vars:
        rename_map["z"] = "geopotential"

    if rename_map:
        ds = ds.rename(rename_map)

    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("Expected latitude/longitude coordinates in GRIB dataset.")

    return ds


def select_region(
    ds: xr.Dataset, region: tuple[float, float, float, float]
) -> xr.Dataset:
    minlat, minlon, maxlat, maxlon = region

    ds = ds.assign_coords(lon=to_lon_360(ds["lon"].values))
    ds = ds.sortby("lon")

    lat_slice = slice(minlat, maxlat)

    if minlon <= maxlon:
        subset = ds.sel(lat=lat_slice, lon=slice(minlon, maxlon))
    else:
        left = ds.sel(lat=lat_slice, lon=slice(minlon, 360.0))
        right = ds.sel(lat=lat_slice, lon=slice(0.0, maxlon))
        subset = xr.concat([left, right], dim="lon")

    return subset


def load_and_preprocess(path: Path) -> xr.Dataset:
    ds = xr.load_dataset(path, engine="cfgrib")
    ds = set_standard_names(ds)

    if "time" in ds.dims:
        ds = ds.isel(time=TARGET_TIME_INDEX, drop=True)
    elif "time" in ds.coords:
        ds = ds.isel(time=TARGET_TIME_INDEX)

    if "level" in ds.coords:
        ds = ds.sel(level=TARGET_LEVEL_HPA, method="nearest")

    ds = ds.sortby("lat")
    ds = select_region(ds, REGION)

    if FCONTOUR_VARIABLE not in ds.data_vars:
        raise ValueError(f"Missing variable: {FCONTOUR_VARIABLE}")
    if CONTOUR_VARIABLE not in ds.data_vars:
        raise ValueError(f"Missing variable: {CONTOUR_VARIABLE}")

    lon_plot = to_lon_360(ds["lon"].values)
    lon_plot = np.array([to_lon_180(float(v)) for v in lon_plot])
    ds = ds.assign_coords(lon_plot=("lon", lon_plot)).sortby("lon_plot")

    return ds


def region_extent(region: tuple[float, float, float, float]) -> list[float]:
    minlat, minlon, maxlat, maxlon = region
    west = to_lon_180(minlon)
    east = to_lon_180(maxlon)
    return [west, east, minlat, maxlat]


def plot_map(ds: xr.Dataset, save_path: Path) -> None:
    projection = create_projection(PROJECTION_TYPE, REGION)

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=FIGSIZE,
        subplot_kw={"projection": projection},
    )
    ax = cast(GeoAxes, ax)

    cf = ax.contourf(
        ds["lon_plot"].values,
        ds["lat"].values,
        ds[FCONTOUR_VARIABLE].values,
        levels=FCONTOUR_LEVELS,
        cmap=FCONTOUR_CMAP,
        extend=FCONTOUR_EXTEND,
        transform=ccrs.PlateCarree(),
    )

    contour_data = ds[CONTOUR_VARIABLE].values
    minor_levels = np.setdiff1d(CONTOUR_MINOR_LEVELS, CONTOUR_MAJOR_LEVELS)

    if minor_levels.size > 0:
        ax.contour(
            ds["lon_plot"].values,
            ds["lat"].values,
            contour_data,
            levels=minor_levels,
            colors=CONTOUR_COLORS,
            linewidths=CONTOUR_MINOR_LINEWIDTH,
            alpha=CONTOUR_MINOR_ALPHA,
            transform=ccrs.PlateCarree(),
        )

    cs_major = ax.contour(
        ds["lon_plot"].values,
        ds["lat"].values,
        contour_data,
        levels=CONTOUR_MAJOR_LEVELS,
        colors=CONTOUR_COLORS,
        linewidths=CONTOUR_MAJOR_LINEWIDTH,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs_major, inline=True, fontsize=8)

    ax.coastlines(resolution="110m", linewidth=0.9)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    gl = ax.gridlines(draw_labels=ADD_LAT_LON_ANNOTATIONS, linewidth=0.4, alpha=0.4)
    if ADD_LAT_LON_ANNOTATIONS:
        gl.top_labels = False
        gl.right_labels = False

    ax.set_extent(region_extent(REGION), crs=ccrs.PlateCarree())
    # ax.set_title("850 hPa temperature and geopotential")

    if ADD_COLORBAR:
        cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, shrink=0.9)
        cbar.set_label(COLORBAR_LABEL)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    ds = load_and_preprocess(DATA_PATH)
    plot_map(ds, OUTPUT_PATH)
    print(f"Saved figure to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
