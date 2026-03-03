import argparse
import json
import os
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from common_utils import standardize_xarray_dims
from panels import plot_tropical_hurricane_track_2
from scripts.hurricane_tracks_config import HurricaneTrackConfig


def load_and_prepare_dataset(spec, region):
    # Load only mslp, chunked, restrict to region
    ds = xr.open_dataset(spec.path, chunks="auto")
    ds = standardize_xarray_dims(ds)
    if spec.input_path:
        ds_input = xr.open_dataset(spec.input_path, chunks="auto")
        ds_input = standardize_xarray_dims(ds_input)
        ds = xr.concat([ds, ds_input], dim="time")
        ds = ds.sortby("time")
    # Only keep mean_sea_level_pressure
    if "mean_sea_level_pressure" in ds:
        mslp = ds["mean_sea_level_pressure"]
    else:
        raise ValueError(f"mean_sea_level_pressure not found in {spec.path}")
    # Restrict to region
    minlat, minlon, maxlat, maxlon = region
    mslp = mslp.sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))
    # Subselect time if timeframe is specified
    if getattr(spec, "timeframe", None) is not None:
        mslp = mslp.isel(time=spec.timeframe)
    return mslp


def main(config_path):
    # Load config
    if config_path.endswith(".json"):
        with open(config_path) as f:
            config_dict = json.load(f)
    else:
        import yaml

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    config = HurricaneTrackConfig.model_validate(config_dict)

    # Prepare figure
    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()}, figsize=config.figsize
    )

    # Prepare plotting args
    mslp_datasets = []
    plot_kwargs_list = []
    for legend_name, spec in config.datasets.items():
        print(f"Loading dataset for {legend_name} from {spec.path}")
        mslp = load_and_prepare_dataset(spec, config.region)
        mslp_datasets.append(mslp)
        # Build plot_kwargs for this dataset
        plot_kwargs_i = {
            "marker": spec.marker or "o",
            "linestyle": spec.linestyle or "-",
        }
        if spec.color:
            plot_kwargs_i["color"] = spec.color
            plot_kwargs_i["cmap"] = False
        else:
            plot_kwargs_i["color"] = spec.colormap
            plot_kwargs_i["cmap"] = True
        plot_kwargs_list.append(plot_kwargs_i)

    # Use region for both search and plot
    search_region = config.region
    plot_region = config.region

    # Plot tracks for each dataset with its own plot_kwargs
    for mslp, plot_kwargs_i in zip(mslp_datasets, plot_kwargs_list):
        ax, _ = plot_tropical_hurricane_track_2(
            ax,
            mslp,
            search_region,
            plot_region,
            title=None,
            plot_kwargs=plot_kwargs_i,
        )

    # Add legend manually
    from matplotlib.lines import Line2D

    legend_elements = []
    for i, (legend_name, spec) in enumerate(config.datasets.items()):
        color = spec.color if getattr(spec, "color", None) else "k"
        marker = spec.marker or "o"
        linestyle = spec.linestyle or "-"
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=legend_name,
            )
        )
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    # Title
    ax.set_title("Hurricane Tracks")

    # Save
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(config.output_file, bbox_inches="tight")
    print(f"Saved to {config.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot hurricane tracks from config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (JSON or YAML)",
    )
    args = parser.parse_args()
    main(args.config)
