import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panels import *
from common_utils import *
import xarray as xr

import os

HOME = os.environ["HOME"]

# Dataset definitions:
pattern = "regional_ep-*.nc"
amse = merge_netcdf_files(
    "/home/users/f/froelicm/scratch/AMSE/amse-with-amse-loss_steps-2_normalized",
    pattern=pattern,
)

# Load HRES data
hres = xr.open_dataset(
    "/home/users/f/froelicm/scratch/Data/GraphCast_OP/custom-hres_2022-09-23_res-0.25_levels-13_steps-24.nc"
).isel(time=slice(-2, None))

# Load norms
norms = xr.open_dataset(
    "/home/users/f/froelicm/scratch/Data/GraphCast_small/stats/stddev_by_level.nc"
)


amse = prep_data(
    amse,
    remove_levels=False,
)

hres = prep_data(
    hres,
    remove_levels=False,
)
hres = hres.assign_coords(time=amse.time.values)

# Define plotting specifications
title = "Hurricane Ian - HRES-fc0 Landfall at 2022-09-28 18z"

# Plotting items
dats = [
    amse.sel(epoch=0, drop=True),
    amse.sel(epoch=1, drop=True),
    amse.sel(epoch=3, drop=True),
    amse.sel(epoch=5, drop=True),
    amse.sel(epoch=9, drop=True),
]

labels = [
    "AMSE",
    "AMSE epoch 2",
    "AMSE epoch 4",
    "AMSE epoch 8",
    "AMSE epoch 10",
]

colors = ["k", "olive", "cyan", "purple", "orange"]

# column_titles = [str(6 * (x + 5)) for x in times]
row_titles = [""]  # GraphCast-AMSE", "10ep", "10ep", "20ep", "40ep"]
column_titles = [""]

"""
norm = BoundaryNorm(boundaries=levels, ncolors=len(levels) + 1, extend=extend)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # Required for colorbar
colorbar = (
    {
        "uv10": (
            sm,
            (0.3, 0.02, 0.4, 0.02),  # (x, y, width, height)
        )
    },
)"""

fig, axes = plt.subplots(figsize=(5, 4))

for i, dat in enumerate(dats):
    plot_PSD_Coh(axes, dat, hres, norms, label=labels[i], color=colors[i], legend=False)

axes.legend()

plt.tight_layout()
save_path = HOME + "/WeatherPlots/HurricaneIan_GC-OP_new/amse_spectral_2step.png"
plt.savefig(save_path, bbox_inches="tight")
