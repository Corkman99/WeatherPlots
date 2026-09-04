import os
import sys
from typing import Optional, Sequence, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_utils import (
    region_to_cartopy_extent,
    select_region,
    standardize_xarray_dims,
)

PLOT_REGION = [15, 260, 50, 290]
BOUNDARIES = np.arange(-1.8, 1.8, 0.05)

DS_1 = "/home/users/f/froelicm/scratch/output/ian_penalized_grid_rerun_0001/target_29/input_epoch-27.nc"
DS_2 = "/home/users/f/froelicm/scratch/output/ian_penalized_grid_rerun_0001/target_52/input_epoch-13.nc"
REF_DS = "/home/users/f/froelicm/scratch/output/ian_penalized_grid_rerun_0001/target_29/input_epoch-0.nc"

FIG_SIZE = (30, 15)


def _load_data(
    source: Union[str, xr.Dataset, xr.DataArray],
) -> Union[xr.Dataset, xr.DataArray]:
    if isinstance(source, str):
        if not source:
            raise ValueError("Dataset path must be provided")
        source = xr.open_dataset(source)
    source = source / 10
    return standardize_xarray_dims(source)


def _prepare_2d_field(
    data: Union[xr.Dataset, xr.DataArray], variable: str
) -> xr.DataArray:
    if isinstance(data, xr.Dataset):
        if variable not in data:
            raise KeyError(f"Variable {variable!r} not found in dataset")
        data = data[variable]

    data = data.squeeze(drop=True)
    for dim in [dim for dim in data.dims if dim not in {"lat", "lon"}]:
        data = data.isel({dim: -1})

    if data.ndim != 2 or "lat" not in data.coords or "lon" not in data.coords:
        raise ValueError("Data must be 2D with lat/lon coordinates after squeezing")

    return data


def difference(
    ref_ds: Union[str, xr.Dataset, xr.DataArray],
    ds: Union[str, xr.Dataset, xr.DataArray],
    variable: str = "mean_sea_level_pressure",
) -> xr.DataArray:
    ref_data = _prepare_2d_field(_load_data(ref_ds), variable)
    data = _prepare_2d_field(_load_data(ds), variable)
    return data - ref_data


def _normalize_longitudes(data: xr.DataArray) -> xr.DataArray:
    """Normalize longitude coordinates to the -180..180 range if needed."""
    if "lon" not in data.coords:
        return data

    lons = data["lon"].values
    if lons.size and float(lons.min()) >= 0 and float(lons.max()) > 180:
        normalized_lon = ((data["lon"] + 180) % 360) - 180
        return data.assign_coords(lon=normalized_lon).sortby("lon")
    return data


def plot_perturbation(
    ds_1: Union[str, xr.Dataset, xr.DataArray] = DS_1,
    ds_2: Union[str, xr.Dataset, xr.DataArray] = DS_2,
    ref_ds: Union[str, xr.Dataset, xr.DataArray] = REF_DS,
    variable: str = "mean_sea_level_pressure",
    plot_region: Sequence[float] = PLOT_REGION,
    boundaries: Sequence[float] = BOUNDARIES,
    output_path: Optional[str] = None,
):
    if not ds_1 or not ds_2 or not ref_ds:
        raise ValueError("ds_1, ds_2 and ref_ds must be provided")

    diff1 = _normalize_longitudes(
        select_region(difference(ref_ds, ds_1, variable), plot_region)
    )
    diff2 = _normalize_longitudes(
        select_region(difference(ref_ds, ds_2, variable), plot_region)
    )

    print(diff1.max().item(), diff1.min().item(), diff1.mean().item())
    print(diff2.max().item(), diff2.min().item(), diff2.mean().item())

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIG_SIZE,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    cf = None
    for ax, diff, title in zip(
        axes,
        (diff1, diff2),
        ("Perturbation 1 − reference", "Perturbation 2 − reference"),
    ):
        extent = region_to_cartopy_extent(plot_region)
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(resolution="110m", linewidth=1.0)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.gridlines(draw_labels=False, linewidth=0.5, color="gray", alpha=0.5)

        cf = ax.contourf(
            diff["lon"],
            diff["lat"],
            diff.values,
            levels=boundaries,
            cmap="bwr",
            extend="both",
            transform=ccrs.PlateCarree(),
            antialiased=True,
        )
        ax.set_title(title)

    if cf is not None:
        cbar = fig.colorbar(
            cf,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.08,
            pad=0.08,
        )
        cbar.set_label(f"{variable} difference")

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=200)

    return fig, axes


if __name__ == "__main__":
    plot_perturbation(
        ds_1=DS_1,
        ds_2=DS_2,
        ref_ds=REF_DS,
        variable="mean_sea_level_pressure",
        plot_region=PLOT_REGION,
        boundaries=BOUNDARIES,
        output_path="perturbation_comparison.png",
    )
