from panels import *
from common_utils import *
import numpy as np
import pickle

dataset_names = ["local", "global"]
dataset_paths = [
    # "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/date-20250704_hurricane-ian_target-miami/losses.pkl",
    # "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/date-20250704_hurricane-ian_target-tallahassee/losses.pkl",
    # "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/date-20250710_hurricane-ian_optimal_local_1e-4_14step/losses.pkl",
    # "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/date-20250710_hurricane-ian_optimal_1e-4_14step/losses.pkl",
    # "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/date-20250704_hurricane-ian_wind-intensification/losses.pkl",
    "DATA/GraphCast_OP/Hurricane_Ian_GC-OP/hurricane-ian_optimal_1e-4_14step_wind-finetune/losses.pkl"
]
save_path = "plotting/Hurricane-Ian/time_losses_14step_finetune.png"

maps = []
for i, dataset_path in enumerate(dataset_paths):
    # open pickle file
    with open(dataset_path, "rb") as f:
        ds = pickle.load(f)
    ds = ds[1:, :]
    maps.append(
        lambda ax, ds=ds, j=i: plot_timeseries_losses(ax, ds, 2, dataset_names[j])
    )

fig = create_multi_panel_figure(maps, nrows=2, ncols=1, figsize=(8, 10))

plt.savefig(save_path, bbox_inches="tight")
