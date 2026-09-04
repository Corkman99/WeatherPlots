import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, cast

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colors import LogNorm, Normalize
from pert_size_vs_distance import perturbation_magnitude
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_utils import get_dim_weights, get_max_epoch_in_dir, normalize_dataset
from dataflows import DEFAULT_PER_VARIABLE_WEIGHTS, load_dataset

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
_logger = logging.getLogger(__name__)

EXPERIMENT_CONFIG_FILE: str = "experiment_config.json"
EXPERIMENT_TARGET: tuple[Optional[type], tuple[str, ...]] = (
    Iterable,
    ("optimization", "earlystop_kwargs", "target_region"),
)
EXPERIMENT_FTDT: tuple[Optional[type], tuple[str, ...]] = (
    datetime,
    ("data", "first_target_datetime"),
)
EXPERIMENT_EPOCHS: tuple[Optional[type], tuple[str, ...]] = (
    int,
    ("optimization", "num_epochs"),
)
LOG_SCALE_EPSILON: float = 1e-10


class HurricaneTrackGridConfig(BaseModel):
    root_dir: str = Field(
        ..., description="Root directory containing target_i folders."
    )
    analysis_path: str = Field(
        ..., description="Reference analysis dataset used in perturbation_magnitude."
    )
    analysis_first_target_datetime: Optional[datetime] = Field(
        default=None,
        description=(
            "Optional first_target_datetime when loading analysis_path in AIWM2 format."
        ),
    )

    output_file: str
    experiment_filename: str = EXPERIMENT_CONFIG_FILE
    target_dir_regex: str = r"^target_(\d+)$"
    chunks: str | dict[str, int] = "auto"

    norm_scales_path: Optional[str] = None
    norm_locations_path: Optional[str] = None
    level_weight_map: Optional[dict[int, float]] = None
    per_variable_weights: Optional[dict[str, float]] = None

    title: str = "Perturbation Magnitude for Grid of Targets"
    colorbar_label: str = "Perturbation magnitude"
    colormap: str = "viridis"
    marker_size: int = 70
    marker_alpha: float = 0.95
    annotate_points: bool = True
    map_extent: Optional[tuple[float, float, float, float]] = Field(
        default=None,
        description="Optional map extent as (minlat, minlon, maxlat, maxlon).",
    )

    log_scale: bool = Field(
        default=False,
        description="Use a logarithmic (LogNorm) scale for the colorbar. All color-mapped perturbation values must be strictly positive.",
    )
    colorbar_vmin: Optional[float] = Field(
        default=None,
        description="Minimum value for colorbar normalization. Auto-derived from data when None.",
    )
    colorbar_vmax: Optional[float] = Field(
        default=None,
        description="Maximum value for colorbar normalization. Auto-derived from data when None.",
    )

    score: str = Field(
        default="MSE",
    )


@dataclass
class TargetPoint:
    folder_name: str
    target_lat: float
    target_lon: float
    perturbation: float
    goal_met: bool


def load_config(config_path: str) -> HurricaneTrackGridConfig:
    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            data = json.load(f)
    else:
        import yaml

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
    return HurricaneTrackGridConfig.model_validate(data)


def extract_from_experiment_config(
    dir: str,
    spec: tuple[Optional[type], tuple[str, ...]],
    experiment_filename: str = EXPERIMENT_CONFIG_FILE,
) -> Any:
    with open(os.path.join(dir, experiment_filename), "r") as f:
        config = json.load(f)
    for key in spec[1]:
        config = config[key]

    expected = spec[0]
    if expected is datetime:
        if isinstance(config, datetime):
            return config
        if isinstance(config, str):
            return datetime.fromisoformat(config)
        raise TypeError(
            f"Expected datetime/string at {spec[1]} in config, got {type(config)}"
        )

    if expected is not None and not isinstance(config, expected):
        raise TypeError(
            f"Expected type {expected} at {spec[1]} in config, got {type(config)}"
        )

    return config


def _iter_target_dirs(root_dir: str, regex: str) -> list[Path]:
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"root_dir does not exist or is not a directory: {root_dir}")

    pattern = re.compile(regex)
    matches = [p for p in root.iterdir() if p.is_dir() and pattern.match(p.name)]
    matches.sort(key=lambda p: p.name)
    return matches


def compute_target_points(config: HurricaneTrackGridConfig) -> list[TargetPoint]:
    scales = (
        xr.open_dataset(config.norm_scales_path)
        if config.norm_scales_path is not None
        else None
    )
    locations = (
        xr.open_dataset(config.norm_locations_path)
        if config.norm_locations_path is not None
        else None
    )
    if scales is not None:
        _logger.info("Loaded normalization scales from %s", config.norm_scales_path)
    if locations is not None:
        _logger.info(
            "Loaded normalization locations from %s", config.norm_locations_path
        )

    weights_per_variable = (
        config.per_variable_weights
        if config.per_variable_weights
        else DEFAULT_PER_VARIABLE_WEIGHTS
    )

    analysis_ds = load_dataset(
        source=config.analysis_path,
        first_target_datetime=config.analysis_first_target_datetime,
        chunks=config.chunks,
    ).isel(time=-1, drop=False)

    weights_latitude, weights_per_level = get_dim_weights(
        ds=analysis_ds,
        level_weight_map=config.level_weight_map,
    )

    points: list[TargetPoint] = []
    target_dirs = _iter_target_dirs(config.root_dir, config.target_dir_regex)
    if not target_dirs:
        raise ValueError(
            f"No target directories matched regex '{config.target_dir_regex}' in {config.root_dir}"
        )

    for target_dir in target_dirs:
        experiment_path = target_dir / config.experiment_filename
        if not experiment_path.exists():
            _logger.warning(
                "Skipping %s (missing %s)", target_dir, experiment_path.name
            )
            continue

        target_region = extract_from_experiment_config(
            str(target_dir),
            EXPERIMENT_TARGET,
            experiment_filename=config.experiment_filename,
        )
        if not isinstance(target_region, Iterable):
            raise TypeError(
                f"Expected iterable target region in {experiment_path}, got {type(target_region)}"
            )
        target_vals = list(target_region)
        if len(target_vals) < 2:
            raise ValueError(
                f"Expected at least 2 values in target region in {experiment_path}, got {target_vals}"
            )
        target_lat, target_lon = float(target_vals[0]), float(target_vals[1])

        try:
            first_target_datetime = extract_from_experiment_config(
                str(target_dir),
                EXPERIMENT_FTDT,
                experiment_filename=config.experiment_filename,
            )
        except KeyError:
            first_target_datetime = None

        prediction_path, max_epoch = get_max_epoch_in_dir(
            str(target_dir),
            startswith="input_epoch-",
            endswith=".nc",
        )

        config_epochs = extract_from_experiment_config(
            str(target_dir),
            EXPERIMENT_EPOCHS,
            experiment_filename=config.experiment_filename,
        )
        goal_met = max_epoch < int(config_epochs)

        prediction_ds = load_dataset(
            source=prediction_path,
            first_target_datetime=first_target_datetime,
            chunks=config.chunks,
        ).isel(time=-1, drop=False)

        perturb = perturbation_magnitude(
            score=config.score,
            prediction=prediction_ds,
            analysis=analysis_ds,
            weights_latitude=weights_latitude,
            weights_per_level=weights_per_level,
            weights_per_variable=weights_per_variable,
            scales=scales,
            locations=locations,
        )

        points.append(
            TargetPoint(
                folder_name=target_dir.name,
                target_lat=target_lat,
                target_lon=target_lon,
                perturbation=perturb,
                goal_met=goal_met,
            )
        )
        _logger.info(
            "Computed %s: perturbation=%.6f from input_epoch-%d (goal_met=%s)",
            target_dir.name,
            perturb,
            max_epoch,
            goal_met,
        )

    if not points:
        raise ValueError("No valid target points were computed.")

    return points


def _build_norm(
    vals: list[float],
    log_scale: bool,
    vmin: Optional[float],
    vmax: Optional[float],
) -> Normalize:
    """Return a matplotlib Normalize (or LogNorm) instance for the given values."""
    effective_vmin = vmin if vmin is not None else min(vals)
    effective_vmax = vmax if vmax is not None else max(vals)
    if log_scale:
        non_positive = [v for v in vals if v <= 0]
        if non_positive:
            raise ValueError(
                f"log_scale requires all color-mapped perturbation values to be strictly "
                f"positive, but found {len(non_positive)} non-positive value(s): "
                f"{non_positive}. Disable log_scale or ensure all perturbation values are > 0."
            )
        if effective_vmin <= 0:
            raise ValueError(
                f"log_scale requires colorbar_vmin > 0, but effective vmin is {effective_vmin}."
            )
        if effective_vmax <= 0:
            raise ValueError(
                f"log_scale requires colorbar_vmax > 0, but effective vmax is {effective_vmax}."
            )
        return LogNorm(vmin=effective_vmin, vmax=effective_vmax)
    return Normalize(vmin=effective_vmin, vmax=effective_vmax)


def plot_target_grid(
    points: list[TargetPoint], config: HurricaneTrackGridConfig
) -> tuple[plt.Figure, GeoAxes, Path]:
    fig, ax = plt.subplots(
        figsize=(10, 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax_geo = cast(GeoAxes, ax)

    lats = [p.target_lat for p in points]
    lons = [p.target_lon for p in points]
    vals = [p.perturbation for p in points]

    met_lats = [p.target_lat for p in points if p.goal_met]
    met_lons = [p.target_lon for p in points if p.goal_met]
    met_vals = [p.perturbation for p in points if p.goal_met]
    not_met_lats = [p.target_lat for p in points if not p.goal_met]
    not_met_lons = [p.target_lon for p in points if not p.goal_met]

    if config.log_scale:
        vals = [LOG_SCALE_EPSILON if v == 0 else v for v in vals]
        met_vals = [LOG_SCALE_EPSILON if v == 0 else v for v in met_vals]

    norm = _build_norm(
        vals, config.log_scale, config.colorbar_vmin, config.colorbar_vmax
    )

    if met_vals:
        sc = ax_geo.scatter(
            met_lons,
            met_lats,
            c=met_vals,
            cmap=config.colormap,
            norm=norm,
            s=config.marker_size,
            alpha=config.marker_alpha,
            edgecolors="black",
            linewidths=0.4,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
    else:
        # Keep colorbar support even if no goal-met points exist.
        sc = ax_geo.scatter(
            lons,
            lats,
            c=vals,
            cmap=config.colormap,
            norm=norm,
            s=config.marker_size,
            alpha=0.0,
            edgecolors="none",
            linewidths=0.0,
            transform=ccrs.PlateCarree(),
            zorder=1,
        )

    if not_met_lats:
        ax_geo.scatter(
            not_met_lons,
            not_met_lats,
            color="black",
            s=config.marker_size,
            alpha=config.marker_alpha,
            edgecolors="black",
            linewidths=0.4,
            transform=ccrs.PlateCarree(),
            zorder=4,
            label="Goal not met",
        )

    if config.annotate_points:
        for p in points:
            ax_geo.text(
                p.target_lon,
                p.target_lat,
                p.folder_name,
                fontsize=8,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

    ax_geo.coastlines(resolution="50m", linewidth=0.8)
    gl = ax_geo.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    if config.map_extent is not None:
        min_lat, min_lon, max_lat, max_lon = config.map_extent
        ax_geo.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    ax_geo.set_title(config.title)
    cbar = fig.colorbar(sc, ax=ax_geo, orientation="vertical", pad=0.03, shrink=0.92)
    cbar.set_label(config.colorbar_label)

    if not_met_lats:
        ax_geo.legend(loc="lower left")

    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return fig, ax_geo, output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot target points from target_i experiment folders with color set "
            "by perturbation_magnitude against a configured analysis dataset."
        )
    )
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    points = compute_target_points(config)
    fig, ax_geo, output_path = plot_target_grid(points, config)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Saved plot to %s", output_path)


def main_and_ensemble_overlay():
    parser = argparse.ArgumentParser(
        description=(
            "Plot target points from target_i experiment folders with color set "
            "by perturbation_magnitude against a configured analysis dataset."
        )
    )
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    points = compute_target_points(config)
    fig, ax_geo, output_path = plot_target_grid(points, config)

    # Max-min GraphCast-AMSE init from IFS ENS
    #

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Saved plot to %s", output_path)


if __name__ == "__main__":
    main()
