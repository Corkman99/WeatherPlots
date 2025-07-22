import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panels import *
from common_utils import *
import numpy as np
import pickle

dataset_names = ["Optim_noprecip", "Optim_noprecip over time"]
dataset_paths = [
    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/optim_noprecip_1e4_ep80/losses.pkl"
]
save_path = "HurricaneIan_GC-OP/optim_noprecip_loss.png"

maps = []
for i, dataset_path in enumerate(dataset_paths):
    # open pickle file
    with open(dataset_path, "rb") as f:
        ds = pickle.load(f)
    ts = ds[1:, :]
    tot = ts.mean(axis=1)
    maps.append(
        lambda ax, ds=tot, j=i: plot_timeseries_losses(ax, ds, 2, dataset_names[j])
    )
    maps.append(
        lambda ax, ds=ts, j=i: plot_timeseries_losses(ax, ds, 10, dataset_names[j])
    )

fig = create_multi_panel_figure(maps, nrows=1, ncols=2, figsize=(12, 6))

plt.savefig(save_path, bbox_inches="tight")
