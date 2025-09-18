import xarray as xr
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

# doesn't seem to work
# plt.rcParams["text.usetex"] = True

import cartopy.crs as ccrs
import cartopy.mpl.contour

legend = False
title = False

forecast = xr.open_dataset(
    "/home/users/f/froelicm/scratch/Data/GraphCast_OP/custom-hres_2022-09-23_res-0.25_levels-13_steps-24.nc"
)[
    ["10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]
].isel(
    time=slice(3, None)
)
field1 = np.sqrt(
    forecast["10m_u_component_of_wind"] ** 2 + forecast["10m_v_component_of_wind"] ** 2
).squeeze()
mslp1 = forecast["mean_sea_level_pressure"].squeeze()

forecast = xr.open_dataset(
    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_6day/miami-track_5e-4/regional_ep-0.nc"
)[["10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]]
field2 = np.sqrt(
    forecast["10m_u_component_of_wind"] ** 2 + forecast["10m_v_component_of_wind"] ** 2
).squeeze()
mslp2 = forecast["mean_sea_level_pressure"].squeeze()

del forecast

fig, axes = plt.subplots(
    1,
    2,
    figsize=(9, 5),
    subplot_kw={
        "projection": ccrs.NearsidePerspective(
            central_latitude=25, central_longitude=-80, satellite_height=1200000
        ),
    },
    dpi=300,
)

fig.subplots_adjust(bottom=0.15)

axes[0].set_global()
axes[0].coastlines()
axes[1].set_global()
axes[1].coastlines()

levels = np.arange(0, 25, 0.5)
cmap = plt.colormaps["coolwarm"]
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

contour_levels = np.arange(950, 1100, 5)
contour_colors = ["grey" if x <= 980 else "black" for x in contour_levels]

mesh1 = axes[0].pcolormesh(
    field1["lon"],
    field1["lat"],
    field1.isel(time=0).values,
    transform=ccrs.PlateCarree(),
    shading="auto",
    cmap="coolwarm",
    norm=norm,
)

mesh2 = axes[1].pcolormesh(
    field2["lon"],
    field2["lat"],
    field2.isel(time=0).values,
    transform=ccrs.PlateCarree(),
    shading="auto",
    cmap="coolwarm",
    norm=norm,
)

# contour1 = axes[0].contour(
#     mslp1["lon"],
#    mslp1["lat"],
#     mslp1.isel(time=0),
#    levels=contour_levels,
#    transform=ccrs.PlateCarree(),
#    colors=contour_colors,
# )
# labels1 = axes[0].clabel(contour1, inline=True, fontsize=8, fmt="%d")
"""
contour2 = axes[1].contour(
    mslp2["lon"],
    mslp2["lat"],
    mslp2.isel(time=0),
    levels=contour_levels,
    transform=ccrs.PlateCarree(),
    colors=contour_colors,
)
labels2 = axes[1].clabel(contour2, inline=True, fontsize=8, fmt="%d")
"""

if legend:
    cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.02])  # custom position and size
    cbar = fig.colorbar(
        mesh1,
        cax=cbar_ax,
        orientation="horizontal",  # ticks=[x * 5 for x in range(6)]
    )
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.set_label("Wind speed (m/s)")

main_title = "Hurricane-Ian, September 2022"
main1 = "HRES-fc0 Analysis"
main2 = "GraphCast-Operational Forecast"

if title:
    fig.suptitle(main_title, fontsize=18)
axes[0].set_title(main1)
axes[1].set_title(main2)

n = field2.sizes["time"]

# def clear_contours_and_labels():
# Remove old contour lines
# print(type(c))
# c.remove()

# Remove old labels
# for txt in label_texts:
#    txt.remove()
# label_texts.clear()

name = "/home/users/f/froelicm/WeatherPlots/PhD-day_plots/Ian_frames_simple/ian2022_"


def update_mesh(t):
    # clear_contours_and_labels()

    # Update mesh color data
    mesh1.set_array(field1.isel(time=t).values.ravel())
    mesh2.set_array(field2.isel(time=t).values.ravel())
    plt.savefig(name + str(t) + ".png", bbox_inches="tight")
    # Create new contours
    # c_new = ax.contour(
    #    field2["lon"],
    #    field2["lat"],
    #    field2.isel(time=t),
    #    levels=np.arange(950, 1050, 5),
    #    transform=ccrs.PlateCarree(),
    #   colors="black",
    # )

    # Create new labels
    # labels_new = ax.clabel(c_new, inline=True, fontsize=8, fmt="%d")
    # label_texts.extend(labels_new)

    # Return all artists that changed
    return [mesh1, mesh2]  # + #[c_new] + label_texts


ani = FuncAnimation(fig, update_mesh, frames=range(n), interval=300)
ani.save(
    "/home/users/f/froelicm/WeatherPlots/PhD-day_plots/ian2022_animation.gif",
    dpi=300,
    writer="pillow",
)
