import argparse
import json
import logging
import os
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
from matplotlib.colors import to_rgba

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from common_utils import select_region, standardize_xarray_dims
from dataflows import aiwm2_preprocess
from panels import plot_tropical_hurricane_track_2
from scripts.hurricane_tracks_config import HurricaneTrackConfig

_logger = logging.getLogger(__name__)


def load_and_prepare_dataset(spec, region):
    # Load only mslp, chunked, restrict to region
    ds = xr.open_dataset(spec.path, chunks="auto")
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = aiwm2_preprocess(ds)
        _logger.info(ds.lat)
    ds = standardize_xarray_dims(ds)
    if spec.input_path:
        ds_input = xr.open_dataset(spec.input_path, chunks="auto")
        if "valid_time" in ds_input.dims or "valid_time" in ds_input.coords:
            ds_input = aiwm2_preprocess(ds_input)
        ds_input = standardize_xarray_dims(ds_input)
        ds = xr.concat([ds, ds_input], dim="time")
        ds = ds.sortby("time")
    # Only keep mean_sea_level_pressure
    if "mean_sea_level_pressure" in ds:
        mslp = ds["mean_sea_level_pressure"]
    else:
        raise ValueError(f"mean_sea_level_pressure not found in {spec.path}")
    mslp = select_region(mslp, region)
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
    datasets_to_plot = config.expand_datasets()

    # Prepare figure
    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()}, figsize=config.figsize
    )

    # Prepare plotting args
    mslp_datasets = []
    plot_kwargs_list = []
    for legend_name, spec in datasets_to_plot.items():
        print(f"Loading dataset for {legend_name} from {spec.path}")
        mslp = load_and_prepare_dataset(spec, config.region)
        mslp_datasets.append(mslp)
        # Build plot_kwargs for this dataset
        plot_kwargs_i = {
            "marker": spec.marker or "o",
            "linestyle": spec.linestyle or "-",
            "alpha": getattr(spec, "line_alpha", 1.0),
            "marker_alpha": getattr(spec, "marker_alpha", 1.0),
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
    for legend_name, mslp, plot_kwargs_i in zip(
        datasets_to_plot.keys(), mslp_datasets, plot_kwargs_list
    ):
        ax, centers = plot_tropical_hurricane_track_2(
            ax,
            mslp,
            search_region,
            plot_region,
            title=None,
            plot_kwargs=plot_kwargs_i,
        )
        track_len = len(centers[0]) if centers else 0
        if track_len == 0:
            print(
                f"Warning: '{legend_name}' produced no track points after region/time filtering."
            )
        else:
            print(
                f"Plotted '{legend_name}' with {track_len} points, color={plot_kwargs_i.get('color')}, linestyle={plot_kwargs_i.get('linestyle')}"
            )

    # Add latitude and longitude gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
        zorder=0,
    )
    gl.top_labels = False
    gl.right_labels = False

    # Add legend manually
    from matplotlib.colors import to_rgba as _to_rgba
    from matplotlib.lines import Line2D

    legend_elements = []
    for i, (legend_name, spec) in enumerate(datasets_to_plot.items()):
        if not getattr(spec, "show_in_legend", True):
            continue
        color = spec.color if getattr(spec, "color", None) else (spec.colormap or "k")
        marker = spec.marker or "o"
        linestyle = spec.linestyle or "-"
        line_rgba = _to_rgba(color, getattr(spec, "line_alpha", 1.0))
        marker_rgba = _to_rgba(color, getattr(spec, "marker_alpha", 1.0))
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=line_rgba,
                marker=marker,
                markerfacecolor=marker_rgba,
                markeredgecolor=marker_rgba,
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
