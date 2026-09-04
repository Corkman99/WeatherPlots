"""Plot hurricane tracks coloured by GraphCast-style weighted MSE input difference.

Datasets whose ``DatasetTrackSpec.color`` is set are drawn with that
fixed colour (intended for ground truth and base-forecast lines).  All
remaining datasets are coloured by the weighted MSE between their
(normalized) input and the (normalized) reference input, using a
``BoundaryNorm`` discrete colormap.  A colorbar is added to the figure
showing the mapping from loss score to colour.

Usage::

    python scripts/hurricane_tracks_by_input_diff.py --config path/to/config.json
"""

import argparse
import json
import logging
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm, colormaps
from matplotlib.colors import BoundaryNorm, LogNorm, to_rgba
from matplotlib.lines import Line2D

# Ensure project root is importable when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_utils import get_dim_weights, normalize_dataset, prep_data
from dataflows import DEFAULT_PER_VARIABLE_WEIGHTS, load_dataset
from panels import plot_tropical_hurricane_track_2
from scripts.hurricane_tracks_by_input_diff_config import InputDiffTrackConfig
from scripts.hurricane_tracks_config import DatasetTrackSpec

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
# Silence verbose Matplotlib internal logger messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)
_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading helpers (mirrors hurricane_tracks.py)
# ---------------------------------------------------------------------------


def load_and_prepare_dataset(
    spec: DatasetTrackSpec,
    region: Tuple[float, float, float, float],
) -> xr.DataArray:
    """Load output mslp data for track plotting, subset to *region* / timeframe.

    Mirrors :func:`hurricane_tracks.load_and_prepare_dataset`.
    """
    ds = load_dataset(spec.path)
    if spec.input_path:
        ds_input = load_dataset(spec.input_path)
        ds = xr.concat([ds, ds_input], dim="time").sortby("time")

    if "mean_sea_level_pressure" not in ds:
        raise ValueError(f"mean_sea_level_pressure not found in {spec.path}")

    time_range = (None, None)
    if spec.timeframe is not None:
        time_range = spec.timeframe

    ds = prep_data(
        ds,
        variables=["mean_sea_level_pressure"],
        region=region,
        time_range=time_range,
    )

    return ds["mean_sea_level_pressure"]


def load_input_dataset(
    spec: DatasetTrackSpec,
    region: Tuple[float, float, float, float],
) -> xr.Dataset:
    """Load the input dataset and subset to *region* / timeframe."""
    if not spec.input_path:
        raise ValueError(f"Dataset spec has no input_path; cannot compute diffs")

    ds = load_dataset(spec.input_path)
    time_range = (None, None)
    if spec.timeframe is not None:
        time_range = spec.timeframe
    ds = prep_data(
        ds,
        region=region,
        time_range=time_range,
    )
    return ds


# ---------------------------------------------------------------------------
# GraphCast-style weighted MSE loss
# ---------------------------------------------------------------------------


def mse_loss(
    prediction: xr.Dataset,
    analysis: xr.Dataset,
    weights_latitude: xr.DataArray,
    weights_per_level: xr.DataArray,
    weights_per_variable: Dict[str, float],
    scales: Optional[xr.Dataset] = None,
    locations: Optional[xr.Dataset] = None,
) -> Tuple[float, xr.Dataset]:
    """Compute weighted MSE loss between *prediction* and *analysis*.

    Both datasets are first **normalized** (if *scales* / *locations* are
    provided), then the MSE is computed exactly as in GraphCast:

    1. ``(diff**2).mean(dim='lon') * lat_weights``  → area-weight
    2. ``.mean(dim='lat') * level_weights``          → level-weight
    3. ``.sum(dim='level')``
    4. ``.mean(dim='time')``                         → average over time
    5. Multiply by per-variable weight and sum.

    Returns ``(total_loss, mse_by_variable)`` where *mse_by_variable* is an
    ``xr.Dataset`` with one scalar per variable (before variable weighting).
    """
    # Determine which variables to include
    all_vars = list(analysis.data_vars)
    if weights_per_variable:
        subset = [v for v in all_vars if weights_per_variable.get(v, 0) > 0]
    else:
        subset = all_vars  # all with weight 1

    # Align
    analysis, prediction = xr.align(analysis, prediction, join="inner")
    analysis = analysis[subset]
    prediction = prediction[subset]

    # Normalize
    analysis = normalize_dataset(analysis, scales, locations)
    prediction = normalize_dataset(prediction, scales, locations)

    diffs = analysis - prediction

    mse_per_var = {}
    total = 0.0
    for var in subset:
        d = diffs[var]
        var_dims = set(d.dims)
        mse = d**2
        if "lon" in var_dims:
            mse = mse.mean(dim="lon") * weights_latitude
        if "lat" in var_dims:
            mse = mse.mean(dim="lat")
        if "level" in var_dims:
            mse = (mse * weights_per_level).sum(dim="level")
        if "time" in mse.dims:
            mse = mse.mean(dim="time")
        # Also average over batch if present
        if "batch" in mse.dims:
            mse = mse.mean(dim="batch")
        # Collapse any remaining dimensions
        if hasattr(mse, "dims") and len(mse.dims) > 0:
            mse = mse.mean()
        scalar = float(mse.values)
        mse_per_var[var] = scalar
        w = weights_per_variable.get(var, 0.0)
        total += scalar * w

    mse_ds = xr.Dataset({v: xr.DataArray(val) for v, val in mse_per_var.items()})
    _logger.info("Computed MSE loss: total=%.6f", total)
    return total, mse_ds


# ---------------------------------------------------------------------------
# Colour / norm helpers
# ---------------------------------------------------------------------------


def _build_boundary_norm(
    scores: Dict[str, float],
    boundaries: List[float] | None,
    colormap_name: str,
    log_scale: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Return *(cmap, norm, bounds)* using ``BoundaryNorm`` or ``LogNorm`` if log_scale.

    If *boundaries* is ``None`` the boundaries are auto-derived from the
    unique sorted score values, padded slightly so every score falls
    inside a bin.
    """
    cmap = colormaps[colormap_name]

    if log_scale:
        # For log scale, use LogNorm and ignore boundaries
        auto_vmin = min(scores.values()) if scores else 1e-6
        auto_vmax = max(scores.values()) if scores else 1.0
        # Use config vmin/vmax if provided
        vmin = vmin if vmin is not None else auto_vmin
        vmax = vmax if vmax is not None else auto_vmax
        # Avoid vmin <= 0
        vmin = max(vmin, 1e-10)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        bounds = None
        return cmap, norm, bounds

    if boundaries is not None:
        bounds = sorted(boundaries)
    else:
        unique = sorted(set(scores.values()))
        if len(unique) == 1:
            delta = 0.1 * (np.abs(unique[0]) + 1e-6)
            bounds = [unique[0] - delta, unique[0] + delta]
        else:
            # Place boundaries midway between successive unique values,
            # plus extra boundaries below the min and above the max.
            bounds = [unique[0] - (unique[1] - unique[0]) * 0.5]
            for a, b in zip(unique[:-1], unique[1:]):
                bounds.append((a + b) / 2.0)
            bounds.append(unique[-1] + (unique[-1] - unique[-2]) * 0.5)

    ncolors = cmap.N
    # Use config vmin/vmax if provided
    norm_vmin = vmin if vmin is not None else (bounds[0] if bounds else None)
    norm_vmax = vmax if vmax is not None else (bounds[-1] if bounds else None)
    norm = BoundaryNorm(bounds, ncolors=ncolors, clip=True)
    if norm_vmin is not None and norm_vmax is not None:
        norm.vmin = norm_vmin
        norm.vmax = norm_vmax
    return cmap, norm, bounds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> InputDiffTrackConfig:
    # ---- Load config -------------------------------------------------------
    if config_path.endswith(".json"):
        with open(config_path) as f:
            config_dict = json.load(f)
    else:
        import yaml

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

    config = InputDiffTrackConfig.model_validate(config_dict)

    expanded_datasets = config.expand_datasets()
    diff_datasets = [s for s in expanded_datasets.values() if not s.color]
    if diff_datasets:
        if config.reference_dataset is None:
            raise ValueError("reference_dataset must be set when computing input diffs")
        if config.reference_dataset not in expanded_datasets:
            raise KeyError(
                f"reference_dataset '{config.reference_dataset}' not found in datasets"
            )
    return config


def _get_max_epoch_in_dir(
    dir: str,
    startswith: str = "output_epoch-",
    endswith: str = ".nc",
) -> int:
    files = os.listdir(dir)

    max_epoch = 0
    for f in files:
        if f.startswith(startswith) and f.endswith(endswith):
            epoch_str = f[len(startswith) : -len(endswith)]
            epoch = int(epoch_str)
            if epoch > max_epoch:
                max_epoch = epoch
    return max_epoch


def update_config_dataset_paths(
    config: InputDiffTrackConfig, pattern: str = "(ep*)"
) -> dict:
    # Auto-resolve epoch numbers for diff datasets if not specified
    spec_update = {}
    for name, spec in config.datasets.items():
        if not isinstance(spec, DatasetTrackSpec):
            spec_update[name] = spec
            continue

        if pattern in name:

            dir = os.path.dirname(spec.path)
            max_epoch = _get_max_epoch_in_dir(dir)

            name = name.replace(pattern, f"(ep{max_epoch})")

            spec_update[name] = spec.model_copy(
                update={
                    "path": os.path.join(dir, f"output_epoch-{max_epoch}.nc"),
                    "input_path": (
                        os.path.join(dir, f"input_epoch-{max_epoch}.nc")
                        if spec.input_path
                        else None
                    ),
                }
            )
        else:
            spec_update[name] = spec  # no change

    return spec_update


def plot_main(ax_main, config, dataset_names, mslp_datasets, plot_kwargs_list):
    """Plot the main hurricane tracks and map on ax_main."""
    search_region = getattr(config, "search_region", None) or config.region
    plot_region = config.region
    for name, mslp, pkw in zip(dataset_names, mslp_datasets, plot_kwargs_list):
        _, centers = plot_tropical_hurricane_track_2(
            ax_main,
            mslp,
            search_region,
            plot_region,
            title=None,
            plot_kwargs=pkw,
        )
        track_len = len(centers[0]) if centers else 0
        if track_len == 0:
            print(
                f"Warning: '{name}' produced no track points after region/time filtering."
            )
        else:
            print(
                f"Plotted '{name}' with {track_len} points, "
                f"color={pkw.get('color')}, linestyle={pkw.get('linestyle')}"
            )

    gl = ax_main.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
        zorder=0,
    )
    gl.top_labels = False
    gl.right_labels = False

    plot_targets(ax_main, config)


def plot_targets(ax_main, config):
    """Plot optional target points defined in config."""
    plot_targets_cfg = getattr(config, "plot_targets", None)
    if not plot_targets_cfg:
        return
    targets = plot_targets_cfg.get("targets", [])
    target_kwargs = plot_targets_cfg.get("kwargs", {})
    for lat, lon in targets:
        norm_lon = lon % 360
        print(
            f"Plotting target point: lat={lat}, lon={lon} (normalized lon={norm_lon}), kwargs={target_kwargs}"
        )
        ax_main.plot(norm_lon, lat, transform=ccrs.PlateCarree(), **target_kwargs)


def plot_colorbar(fig, ax_cbar, norm, cmap, config, bounds=None, log_colorbar=False):
    """Plot the colorbar on ax_cbar. Font size scales with figure size."""
    if cmap is None or norm is None:
        ax_cbar.axis("off")
        return
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig_width, fig_height = fig.get_size_inches()
    base_fontsize = None
    if hasattr(config, "font_size") and config.font_size is not None:
        try:
            base_fontsize = int(config.font_size)
        except Exception:
            base_fontsize = 12
    if base_fontsize is None:
        base_fontsize = max(10, min(14, int((fig_width + fig_height) * 1.0)))
    colorbar_kwargs = dict(
        cax=ax_cbar,
        orientation="vertical",
        pad=0.1,
    )
    if not log_colorbar and bounds is not None:
        colorbar_kwargs["boundaries"] = bounds
        colorbar_kwargs["ticks"] = bounds
    if hasattr(config, "extend") and config.extend is not None:
        colorbar_kwargs["extend"] = config.extend
    cbar = fig.colorbar(sm, **colorbar_kwargs)
    cbar.set_label(config.colorbar_label, fontsize=base_fontsize, labelpad=5)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.tick_params(labelsize=base_fontsize - 2)


def plot_legend(ax_legend, legend_elements, config, fig, ax_main=None):
    """Plot the legend on ax_legend. Font size scales with figure size."""
    if not legend_elements:
        if ax_legend is not None:
            ax_legend.clear()
            ax_legend.axis("off")
        return

    n_entries = len(legend_elements)
    if n_entries <= 4:
        ncol = 1
    elif n_entries <= 10:
        ncol = 2
    elif n_entries <= 18:
        ncol = 3
    else:
        ncol = 4
    nrows = math.ceil(n_entries / ncol)

    fig_width, fig_height = fig.get_size_inches()
    base_fontsize = None
    if hasattr(config, "font_size") and config.font_size is not None:
        try:
            base_fontsize = int(config.font_size)
        except Exception:
            base_fontsize = 12
    if base_fontsize is None:
        base_fontsize = max(
            7,
            min(13, int((fig_width + fig_height) * 0.8) - max(0, nrows - 2)),
        )

    # Always place legend on the map, bottom right
    if ax_main is not None:
        leg = ax_main.legend(
            handles=legend_elements,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.02),
            ncol=ncol,
            frameon=True,
            fontsize=base_fontsize,
            title="Legend",
            title_fontsize=max(10, base_fontsize),
            handlelength=2.8,
            handletextpad=0.8,
            columnspacing=1.0,
            borderaxespad=0.0,
            markerscale=1.5,
        )
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.8)
        leg.get_frame().set_edgecolor("0.4")
        if ax_legend is not None:
            ax_legend.clear()
            ax_legend.axis("off")
        return


def main(config_path: str, get_max_epoch: bool = True, pattern: str = "(ep*)"):
    """
    get_max_epoch: If true, determines max epoch number for each specified dataset that contains (ep*) in the name
    Assumes path and input_path to have the same epoch.
    """
    config = load_config(config_path)

    if get_max_epoch:
        spec_update = update_config_dataset_paths(config, pattern)
        config = config.model_copy(update={"datasets": spec_update})

    datasets_to_plot = config.expand_datasets()
    dataset_names = list(datasets_to_plot.keys())

    fixed_names: List[str] = []
    diff_names: List[str] = []
    for name, spec in datasets_to_plot.items():
        if spec.color:
            if getattr(spec, "show_in_legend", True):
                fixed_names.append(name)
        else:
            diff_names.append(name)

    fixed_count = len(fixed_names)
    legend_loc = str(getattr(config, "legend_loc", "right")).lower()
    legend_on_map = legend_loc in {"on_map", "inside", "map"}
    if fixed_count <= 4:
        legend_width_ratio = 1.0
        bottom_row_ratio = 0.8
    elif fixed_count <= 10:
        legend_width_ratio = 1.5
        bottom_row_ratio = 1.1
    elif fixed_count <= 18:
        legend_width_ratio = 2.0
        bottom_row_ratio = 1.4
    else:
        legend_width_ratio = 2.4
        bottom_row_ratio = 1.7

    # ---- Compute diff scores for diff-coloured datasets --------------------
    # Load normalization statistics (optional)
    scales: Optional[xr.Dataset] = None
    locations: Optional[xr.Dataset] = None
    if config.norm_scales_path:
        scales = xr.open_dataset(config.norm_scales_path)
        print(f"Loaded normalization scales from {config.norm_scales_path}")
    if config.norm_locations_path:
        locations = xr.open_dataset(config.norm_locations_path)
        print(f"Loaded normalization locations from {config.norm_locations_path}")

    # Resolve per-variable weights
    per_variable_weights = (
        config.per_variable_weights
        if config.per_variable_weights
        else DEFAULT_PER_VARIABLE_WEIGHTS
    )

    scores: Dict[str, float] = {}
    if diff_names:
        if config.reference_dataset is None:
            raise ValueError("reference_dataset is required when computing input diffs")
        ref_spec = datasets_to_plot[config.reference_dataset]
        ref_input_ds = load_input_dataset(ref_spec, config.region)

        # Derive latitude / level weights from the reference input
        lat_weights, level_weights = get_dim_weights(
            ref_input_ds, config.level_weight_map
        )

        for name in diff_names:
            spec = datasets_to_plot[name]
            cand_input = load_input_dataset(spec, config.region)
            score, mse_by_var = mse_loss(
                prediction=cand_input,
                analysis=ref_input_ds,
                weights_latitude=lat_weights,
                weights_per_level=level_weights,
                weights_per_variable=per_variable_weights,
                scales=scales,
                locations=locations,
            )
            scores[name] = score
            # print(f"MSE score for '{name}': {score:.6f}")
            # for v, val in mse_by_var.data_vars.items():
            #    print(f"  {v}: {float(val.values):.6f}")

    # ---- Build BoundaryNorm or LogNorm colour mapping ---------------------------------
    log_colorbar = getattr(config, "log_colorbar", False)
    if scores:
        cmap, norm, bounds = _build_boundary_norm(
            scores,
            config.diff_boundaries,
            config.diff_colormap,
            log_scale=log_colorbar,
            vmin=getattr(config, "colorbar_vmin", None),
            vmax=getattr(config, "colorbar_vmax", None),
        )
    else:
        cmap = norm = bounds = None

    # ---- Prepare figure and axes: main plot + vertical colorbar on right ----
    fig = plt.figure(figsize=config.figsize, constrained_layout=True)
    # Main axis
    ax_main = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # Add a vertical colorbar axis to the right, 70% of axis height
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cbar_width = 0.04
    cbar_pad = 0.04
    cbar_height = 0.92  # Even longer colorbar
    cbar_bottom = 0.04
    axpos = ax_main.get_position()
    ax_cbar = fig.add_axes(
        (
            axpos.x1 + cbar_pad,
            axpos.y0 + cbar_bottom * axpos.height,
            cbar_width,
            cbar_height * axpos.height,
        )
    )
    ax_legend = None  # Legend will be on the map

    # ---- Load data & build per-dataset plot_kwargs -------------------------
    search_region = getattr(config, "search_region", None) or config.region
    mslp_datasets: List[xr.DataArray] = []
    plot_kwargs_list: List[Dict] = []

    for name, spec in datasets_to_plot.items():
        print(f"Loading dataset for '{name}' from {spec.path}")
        mslp = load_and_prepare_dataset(spec, search_region)
        mslp_datasets.append(mslp)

        # Determine colour
        if spec.color:
            color = spec.color
        elif cmap is not None and norm is not None:
            color = cmap(norm(scores[name]))
        else:
            color = "tab:blue"

        # Support both 'linewidth' and 'lwd' as aliases
        linewidth = getattr(spec, "linewidth", None)
        if linewidth is None:
            linewidth = getattr(spec, "lwd", 1)
        plot_kwargs_i = {
            "marker": spec.marker or ".",
            "linestyle": spec.linestyle or "-",
            "alpha": getattr(spec, "line_alpha", 1.0),
            "marker_alpha": getattr(spec, "marker_alpha", 1.0),
            "markersize": getattr(spec, "marker_size", None)
            or getattr(spec, "markersize", None),
            "color": color,
            "cmap": False,  # fixed colour per track
            "linewidth": linewidth,
        }
        # Inject global 'extend' if present
        extend_global = getattr(config, "extend", None)
        if extend_global is not None:
            plot_kwargs_i["extend"] = extend_global
        # Allow per-dataset override
        extend_dataset = getattr(spec, "extend", None)
        if extend_dataset is not None:
            plot_kwargs_i["extend"] = extend_dataset
        plot_kwargs_list.append(plot_kwargs_i)

    # ---- Plot tracks, gridlines, and target points ------------------------
    plot_main(ax_main, config, dataset_names, mslp_datasets, plot_kwargs_list)

    # ---- Legend for fixed-colour datasets (prepare only) ---------------------
    legend_elements = []
    for name in fixed_names:
        pkw = plot_kwargs_list[dataset_names.index(name)]
        color = pkw["color"]
        line_rgba = to_rgba(color, pkw.get("alpha", 1.0))
        marker_rgba = to_rgba(color, pkw.get("marker_alpha", 1.0))
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=line_rgba,
                marker=pkw["marker"],
                markerfacecolor=marker_rgba,
                markeredgecolor=marker_rgba,
                linestyle=pkw["linestyle"],
                label=name,
            )
        )

    # ---- Colorbar and Legend below the figure, side-by-side ------------------
    plot_colorbar(
        fig, ax_cbar, norm, cmap, config, bounds=bounds, log_colorbar=log_colorbar
    )
    plot_legend(ax_legend, legend_elements, config, fig, ax_main=ax_main)

    # ---- Title & save -------------------------------------------------------
    # Only set title if config.title is not None
    if getattr(config, "title", None):
        base_fontsize = None
        if hasattr(config, "font_size") and config.font_size is not None:
            try:
                base_fontsize = int(config.font_size)
            except Exception:
                base_fontsize = 12
        if base_fontsize is not None:
            title_fontsize = base_fontsize + 2
        else:
            title_fontsize = max(14, int(sum(config.figsize) * 1.5))
        ax_main.set_title(config.title, fontsize=title_fontsize)
    plt.savefig(config.output_file, bbox_inches="tight")
    print(f"Saved to {config.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot hurricane tracks coloured by mean input differences",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file.",
    )
    args = parser.parse_args()
    main(args.config)
