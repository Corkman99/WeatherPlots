import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panels import *
from common_utils import *
import numpy as np
import pickle

dataset_names = [
    "MSE ckpt 5e-4",
    "Global 5e-4",
    "Global 1e-3",
]  # "LR = 1e-5"]
dataset_paths = [
    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/optimal_no-precip_14steps/losses.pkl",
    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/AMSE/optim_noprecip_5e-4/losses.pkl",
    "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/AMSE/optim_noprecip_1e-3/losses.pkl",
    # "/home/users/f/froelicm/scratch/GraphCast-OP_TC_5day/AMSE/optim_noprecip_5e-4_regional/losses.pkl",
]
save_path = "HurricaneIan_GC-OP/AMSE/loss_amse_5e-4.png"
num_epochs = 10
maps = []
for i, dataset_path in enumerate(dataset_paths):
    # open pickle file
    with open(dataset_path, "rb") as f:
        ds = pickle.load(f)
    ts = ds[1 : (num_epochs + 1), :]
    tot = ts.mean(axis=1)
    maps.append(lambda ax, ds=tot: plot_timeseries_losses(ax, ds, 3, ylim=(2, 3)))
    maps.append(lambda ax, ds=ts: plot_timeseries_losses(ax, ds, 3, ylim=(0, 10)))

fig = create_multi_panel_figure(
    maps,
    nrows=len(dataset_paths),
    ncols=2,
    figsize=(12, 8),
    panel_labels={"row": dataset_names, "col": ["Total Loss", "Per Timestep Loss"]},
)

plt.savefig(save_path, bbox_inches="tight")
