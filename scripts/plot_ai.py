import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize


def load_from_nc(
    input: str, variable: str, time_step: np.datetime64, time_name: str = "valid_time"
) -> xr.DataArray:
    return xr.open_dataset(input)[variable].sel({time_name: time_step}).squeeze()


# function that takes as argument and xarray and generates a plot of this array
# it should have lat and lon dimensions only when squeeze is applied
# use cartopy, add a colormap (argument is cmap) and land-sea-mask
def plot_xarray(da: xr.DataArray, ax, cmap: str = "coolwarm", norm=None) -> None:
    # ax.contourf(da.longitude, da.latitude, da, cmap=cmap, norm=norm)
    da.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)
    ax.coastlines()


IN1 = "/home/users/f/froelicm/scratch/model_comparisons/datasets/gc-pangu-aurora_20220122T12.nc"
IN2 = "/home/users/f/froelicm/scratch/model_comparisons/datasets/gc-pangu-aurora_20220122T12_shuffled.nc"
OUT = "AI_weather_models_2/shuffled_mslp.png"
TIME = np.datetime64("2022-01-22T12:00:00")

if __name__ == "__main__":
    da1 = load_from_nc(IN1, "msl", TIME).compute()
    da2 = load_from_nc(IN2, "msl", TIME).compute()

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axes[0].set_title("Original MSLP")
    axes[1].set_title("Shuffled MSLP")

    cmap = "coolwarm"
    norm = Normalize(
        min(da1.min().item(), da2.min().item()), max(da1.max().item(), da2.max().item())
    )
    plot_xarray(da1, axes[0], cmap=cmap, norm=norm)
    plot_xarray(da2, axes[1], cmap=cmap, norm=norm)

    fig.suptitle("Mean Sea Level Pressure - Un-physical Shuffle")
    plt.savefig(OUT, bbox_inches="tight", dpi=300)
