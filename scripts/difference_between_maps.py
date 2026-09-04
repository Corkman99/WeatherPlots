import matplotlib.pyplot as plt
import xarray
from cartopy import crs as ccrs
from cartopy import feature as cfeature


def load_diff():
    ds_1 = xarray.open_dataset(
        "/home/users/f/froelicm/scratch/output/ian_16step_penalized_lr0001/target_29/output_epoch-23.nc"
    )
    ds_2 = xarray.open_dataset(
        "/home/users/f/froelicm/scratch/output/ian_16step_penalized_lr0001/target_50/output_epoch-15.nc"
    )
    diff = (ds_1 - ds_2)["geopotential"]
    # Pa to decameter
    diff = diff / 9.81 / 10
    return (
        diff.sel(level=500, lat=slice(21, 40), lon=slice(268, 285))
        .isel(time=8)
        .squeeze()
    )


def plot_dff():
    ds_diff = load_diff()
    fig, ax = plt.subplots(
        figsize=(15, 20), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    font_size = 20
    plt.rcParams.update({"font.size": font_size})

    level_boundaries = [-4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4]
    ticks_major = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    ticks_minor = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]

    # set values between -0.5 and 0.5 to nan
    ds_diff.values[(ds_diff.values >= -0.5) & (ds_diff.values <= 0.5)] = float("nan")

    ax.contourf(
        ds_diff.lon,
        ds_diff.lat,
        ds_diff.values,
        levels=level_boundaries,
        cmap="bwr",
        extend="both",
        alpha=0.8,
    )
    # ax.set_title("3day leadtime: East minus North-West trajectories at +2d12h")

    # add coastlines and US states
    ax.coastlines(resolution="10m", linewidth=1.2)
    ax.add_feature(cfeature.STATES, linewidth=0.7)

    # set land color to grey
    ax.add_feature(cfeature.LAND, facecolor="lightgray")

    # Add colormap, horizontal
    cbar = plt.colorbar(
        ax.collections[0], ax=ax, orientation="vertical", pad=0.05, fraction=0.04
    )
    cbar.set_label("500hPa Geopotential height difference (dam)")
    # add zero label to cbar
    cbar.set_ticks(ticks_major)
    cbar.set_ticks(ticks_minor, minor=True)

    plt.savefig("outputs/diff_ian_rerun_12s_geopot500_plus_2.png", bbox_inches="tight")


if __name__ == "__main__":
    plot_dff()
