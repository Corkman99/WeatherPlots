import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pydantic import BaseModel, Field, model_validator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from common_utils import (
    RegionSpec,
    _extract_hurricane_centers,
    distance_on_sphere,
    extract_from_experiment_config,
    get_dim_weights,
    get_max_epoch_in_dir,
    normalize_dataset,
)
from dataflows import DEFAULT_PER_VARIABLE_WEIGHTS, load_dataset

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
_logger = logging.getLogger(__name__)

ENSEMBLE_DIM: str = "number"


def perturbation_magnitude(
    score: str,
    prediction: xr.Dataset,
    analysis: xr.Dataset,
    weights_latitude: xr.DataArray,
    weights_per_level: xr.DataArray,
    weights_per_variable: dict[str, float],
    scales: Optional[xr.Dataset] = None,
    locations: Optional[xr.Dataset] = None,
) -> float:

    if score == "MSE":
        analysis = normalize_dataset(analysis, scales, locations)
        prediction = normalize_dataset(prediction, scales, locations)
        mag, _ = weighted_MSE(
            prediction=prediction,
            analysis=analysis,
            weights_latitude=weights_latitude,
            weights_per_level=weights_per_level,
            weights_per_variable=weights_per_variable,
            scales=scales,
            locations=locations,
        )
    elif score == "TotalEnergy":
        analysis = normalize_dataset(analysis, scales, locations)
        prediction = normalize_dataset(prediction, scales, locations)
        mag = total_energy(
            inputs=prediction,
            original_inputs=analysis,
            cq=0.0,  # dry total energy
        )
    else:
        raise ValueError(f"Unsupported score '{score}'")
    return mag


def weighted_MSE(
    prediction: xr.Dataset,
    analysis: xr.Dataset,
    weights_latitude: xr.DataArray,
    weights_per_level: xr.DataArray,
    weights_per_variable: dict[str, float],
    scales: Optional[xr.Dataset] = None,
    locations: Optional[xr.Dataset] = None,
) -> tuple[float, xr.Dataset]:
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
    return total, mse_ds


def total_energy(
    inputs,
    original_inputs,
    cq,
):
    """
    Compute the dry or moist total energy of a perturbation between two
    atmospheric states, by using a weighted sum to approximate the continuous
    integral formulation in:
    https://www.ecmwf.int/en/elibrary/73574-tropical-singular-vectors-computed-linearized-diabatic-physics

    Parameters
    ----------
    inputs : xr.Dataset
        Perturbed atmospheric state.

    original_inputs : xr.Dataset
        Reference atmospheric state.

    cq : float
        Moisture weighting coefficient.
        0 = dry total energy norm.

    Returns
    -------
    xr.DataArray
        Scalar total energy of the perturbation [J].
    """

    # Constants
    CP = 1004.0
    G = 9.80665
    LV = 2.5e6
    TR = 300.0

    if "time" in inputs.dims:
        inputs = inputs.isel(time=-1, drop=True)
    if "time" in original_inputs.dims:
        original_inputs = original_inputs.isel(time=-1, drop=True)

    u_name = "u_component_of_wind"
    v_name = "v_component_of_wind"
    temperature_name = "temperature"
    humidity_name = "specific_humidity"

    # Assume un-normalized fields
    du = inputs[u_name] - original_inputs[u_name]
    dv = inputs[v_name] - original_inputs[v_name]
    dT = inputs[temperature_name] - original_inputs[temperature_name]
    dq = inputs[humidity_name] - original_inputs[humidity_name]

    # lat weights
    lat_radians = np.deg2rad(inputs.lat)
    area_weights = np.cos(lat_radians)

    # ------------------------------------------------------------------
    # Discretizes the vertical integral in pressure coordinates, then convert to mass:
    # - Layer thickness estimated using centered finite differences, with edge layers use one-sided differences
    # - Assume hydrostatic balanc, then mass is pressure thickness divided by g
    # ------------------------------------------------------------------

    p = inputs.level.values

    dp = np.zeros_like(p, dtype=np.float64)

    dp[1:-1] = 0.5 * (p[:-2] - p[2:])
    dp[0] = p[0] - p[1]
    dp[-1] = p[-2] - p[-1]

    mass_weights = xr.DataArray(
        -dp / G,
        coords={"level": inputs.level},
        dims=["level"],
    )

    # ------------------------------------------------------------------
    # Compute kinetic energy contribution: KE = 1/2 (u'^2 + v'^2)
    # - Vertical kinetic energy neglected due to hydrostatic balance assumption
    # and small relative scale.
    # ------------------------------------------------------------------

    kinetic_energy_density = 0.5 * (du**2 + dv**2)

    # ------------------------------------------------------------------
    # Compute available potential energy: APE ≈ 1/2 * (cp / Tr) * T'^2
    # - Small perturbations.
    # - Constant reference temperature Tr.
    # - Linearized dry thermodynamics.
    # ------------------------------------------------------------------

    temperature_energy_density = 0.5 * (CP / TR) * (dT**2)

    # Combine dry total energy density:
    total_energy_density = kinetic_energy_density + temperature_energy_density

    # ------------------------------------------------------------------
    # Moisture contribution: 1/2 * cq * (Lv^2 / (cp Tr)) * q'^2
    # - Quadratic latent-energy approximation.
    # - Condensate neglected.
    # ------------------------------------------------------------------

    moisture_energy_density = 0.5 * cq * (LV**2 / (CP * TR)) * (dq**2)

    total_energy_density = total_energy_density + moisture_energy_density

    # Approximate integral:
    vertically_weighted_energy = total_energy_density * mass_weights
    area_weighted_energy = vertically_weighted_energy * area_weights
    total_energy = area_weighted_energy.sum(dim=["level", "lat", "lon"])

    print("Total energy:", total_energy)

    return total_energy.compute().item()


HURRICANE_VARIABLE: str = "mean_sea_level_pressure"
EXPERIMENT_PREFIX: tuple[str, ...] = ("minus", "plus")
EXPERIMENT_CONFIG_FILE: str = "experiment_config.json"
EXPERIMENT_FTDT: tuple[Optional[type], tuple[str, ...]] = (
    datetime,
    ("data", "first_target_datetime"),
)
EXPERIMENT_TARGET: tuple[Optional[type], tuple[str, ...]] = (
    Iterable,
    ("optimization", "earlystop_kwargs", "target_region"),
)
EXPERIMENT_TARGET_TOL: tuple[Optional[type], tuple[str, ...]] = (
    Iterable,
    ("optimization", "earlystop_kwargs", "tolerance"),
)


class EnsembleNormConfig(BaseModel):
    reference_path: str = Field(
        ..., description="Reference analysis dataset used against ensemble members"
    )
    ensemble_path: str = Field(..., description="Ensemble dataset containing members")
    first_target_datetime: Optional[datetime] = Field(
        default=None,
        description=(
            "first_target_datetime used when loading AIWM2-format files. "
            "If omitted, it is inferred by load_dataset."
        ),
    )


class PertSizeVsDistanceConfig(BaseModel):
    home_paths: dict[str, str] = Field(
        ...,
        description=(
            "Mapping from legend label to experiment-home directory. Each home "
            "directory contains multiple run subdirectories (e.g. minus*/plus*)."
        ),
    )
    ref_target: tuple[float, float] | dict[str, tuple[float, float]] = Field(
        ...,
        description=(
            "Reference target as (lat, lon) used for distance, or mapping from "
            "home_paths label to (lat, lon)."
        ),
    )
    search_region: RegionSpec = Field(
        ...,
        description="(minlat, minlon, maxlat, maxlon) region for storm center search",
    )

    ensemble_norm: Optional[EnsembleNormConfig] = None
    ensemble_norm_by_label: Optional[dict[str, EnsembleNormConfig]] = Field(
        default=None,
        description=(
            "Optional mapping from home_paths label to ensemble normalization config. "
            "When provided, each label is normalized by its own ensemble std-dev."
        ),
    )

    norm_scales_path: Optional[str] = None
    norm_locations_path: Optional[str] = None
    level_weight_map: Optional[dict[int, float]] = None
    per_variable_weights: Optional[dict[str, float]] = None

    chunks: str | dict[str, int] = "auto"
    output_dir: str = "./outputs/Perturb"
    output_file: str = "pert_size_vs_distance.png"
    title: str = "Perturbation Size vs Target Distance"
    xlabel: str = "Distance from reference target (km)"
    ylabel: str = "Normalized perturbation size (sigma_ens units)"
    marker_size: int = 24
    alpha: float = 0.85
    draw_line: bool = False
    grid: bool = True

    @model_validator(mode="after")
    def validate_ref_target_labels(self) -> "PertSizeVsDistanceConfig":
        if isinstance(self.ref_target, dict):
            missing = [
                label for label in self.home_paths if label not in self.ref_target
            ]
            if missing:
                raise ValueError(
                    "ref_target is missing entries for labels: " + ", ".join(missing)
                )
        return self

    def ref_target_for_label(self, label: str) -> tuple[float, float]:
        if isinstance(self.ref_target, dict):
            return self.ref_target[label]
        return self.ref_target


def load_experiment_outputs(
    dir: str,
    ref_target: tuple[float, float],
    search_region: RegionSpec,
    weights_latitude: xr.DataArray,
    weights_per_level: xr.DataArray,
    weights_per_variable: dict[str, float],
    chunks: str | dict[str, int] = "auto",
    scales: Optional[xr.Dataset] = None,
    locations: Optional[xr.Dataset] = None,
    score: str = "MSE",
) -> Generator[tuple[float, float, bool], None, None]:
    # Open each input and output file
    for folder in os.listdir(dir):
        for pre in EXPERIMENT_PREFIX:
            if folder.startswith(pre):
                input_file_path, ep = get_max_epoch_in_dir(
                    os.path.join(dir, folder), "input_epoch-", ".nc"
                )
                ftdt: datetime = extract_from_experiment_config(
                    os.path.join(dir, folder), EXPERIMENT_FTDT
                )

                # Get perturbation magnitude
                input_analysis_file = os.path.join(dir, folder, f"input_epoch-0.nc")
                mag = perturbation_magnitude(
                    score=score,
                    prediction=load_dataset(
                        source=input_file_path,
                        first_target_datetime=ftdt,
                        chunks=chunks,
                    ).isel(time=-1, drop=False),
                    analysis=load_dataset(
                        source=input_analysis_file,
                        first_target_datetime=ftdt,
                        chunks=chunks,
                    ).isel(time=-1, drop=False),
                    weights_latitude=weights_latitude,
                    weights_per_level=weights_per_level,
                    weights_per_variable=weights_per_variable,
                    scales=scales,
                    locations=locations,
                )

                # Get hurricane center distance
                output_file = os.path.join(dir, folder, f"output_epoch-{ep}.nc")
                center = _extract_hurricane_centers(
                    load_dataset(
                        source=output_file, first_target_datetime=ftdt, chunks=chunks
                    )[HURRICANE_VARIABLE].isel(time=-1, drop=False),
                    search_region,
                )[-1]
                distance = distance_on_sphere(point1=ref_target, point2=center)

                # Goal met?
                target_lat, target_lon, _, _ = extract_from_experiment_config(
                    os.path.join(dir, folder), EXPERIMENT_TARGET
                )
                tol_min_lat, tol_min_lon, tol_max_lat, tol_max_lon = (
                    extract_from_experiment_config(
                        os.path.join(dir, folder), EXPERIMENT_TARGET_TOL
                    )
                )
                target_met = (
                    target_lat - tol_min_lat <= center[0] <= target_lat + tol_max_lat
                ) and (
                    target_lon - tol_min_lon <= center[1] <= target_lon + tol_max_lon
                )

                yield (distance, mag, target_met)


def compute_ensemble_perturbation_variance(
    ds_ref: xr.Dataset,
    ds_ens: xr.Dataset,
    weights_latitude: xr.DataArray,
    weights_per_level: xr.DataArray,
    weights_per_variable: dict[str, float],
    scales: Optional[xr.Dataset] = None,
    locations: Optional[xr.Dataset] = None,
    score: str = "MSE",
) -> float:
    mags = []
    for ep in ds_ens["number"].values:
        mag = perturbation_magnitude(
            score=score,
            prediction=ds_ens.sel(number=ep),
            analysis=ds_ref,
            weights_latitude=weights_latitude,
            weights_per_level=weights_per_level,
            weights_per_variable=weights_per_variable,
            scales=scales,
            locations=locations,
        )
        mags.append(mag)
    return float(np.var(mags))


def _get_compute_kwargs(config) -> dict:
    scales = xr.open_dataset(config.norm_scales_path)
    print(f"Loaded normalization scales from {config.norm_scales_path}")
    locations = xr.open_dataset(config.norm_locations_path)
    print(f"Loaded normalization locations from {config.norm_locations_path}")

    per_variable_weights = (
        config.per_variable_weights
        if config.per_variable_weights
        else DEFAULT_PER_VARIABLE_WEIGHTS
    )

    # Derive latitude / level weights from the reference input
    lat_weights, level_weights = get_dim_weights(
        level_weight_map=config.level_weight_map
    )

    return dict(
        scales=scales,
        locations=locations,
        weights_latitude=lat_weights,
        weights_per_level=level_weights,
        weights_per_variable=per_variable_weights,
        score=config.score,
    )


def compute_ensemble_perturbation_std(
    ensemble_norm: EnsembleNormConfig,
    chunks: str | dict[str, int],
    compute_kwargs: dict,
) -> float:
    ds_ref = load_dataset(
        ensemble_norm.reference_path,
        chunks=chunks,
    ).isel(time=-1, drop=False)
    ds_ens = load_dataset(
        ensemble_norm.ensemble_path,
        first_target_datetime=ensemble_norm.first_target_datetime,
        chunks=chunks,
    )

    if "number" not in ds_ens.dims:
        if ENSEMBLE_DIM in ds_ens.dims:
            ds_ens = ds_ens.rename({ENSEMBLE_DIM: "number"})
        else:
            raise ValueError(
                f"Ensemble dataset must have 'number' or '{ENSEMBLE_DIM}' dimension. "
                f"Found dims: {list(ds_ens.dims)}"
            )

    variance = compute_ensemble_perturbation_variance(
        ds_ref=ds_ref,
        ds_ens=ds_ens,
        **compute_kwargs,
    )
    stddev = float(np.sqrt(variance))
    if stddev <= 0:
        raise ValueError(
            "Ensemble perturbation standard deviation is <= 0; cannot normalize magnitudes."
        )
    return stddev


def plot_perturbation_vs_distance(
    data_by_label: dict[str, list[tuple[float, float, bool]]],
    config: PertSizeVsDistanceConfig,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, points in data_by_label.items():
        if not points:
            _logger.warning(
                "No data points for label '%s'; skipping plot for this label.", label
            )
            continue

        points_sorted = sorted(points, key=lambda x: x[0])

        distances_met = [p[0] for p in points_sorted if p[2]]
        distances_notmet = [p[0] for p in points_sorted if not p[2]]
        magnitudes_met = [p[1] for p in points_sorted if p[2]]
        magnitudes_notmet = [p[1] for p in points_sorted if not p[2]]
        ax.scatter(
            distances_met,
            magnitudes_met,
            s=config.marker_size,
            alpha=config.alpha,
            label=label,
        )
        ax.scatter(
            distances_notmet,
            magnitudes_notmet,
            s=config.marker_size,
            alpha=config.alpha,
            marker="x",
            color=ax.get_lines()[-1].get_color(),  # match color to met points
        )

    ax.set_title(config.title)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    if config.grid:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config.output_file

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def load_config(config_path: str) -> PertSizeVsDistanceConfig:
    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            data = json.load(f)
    else:
        import yaml

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
    return PertSizeVsDistanceConfig.model_validate(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot normalized perturbation size vs target distance for experiment outputs."
        )
    )
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config")
    args = parser.parse_args()

    config = load_config(args.config)

    compute_kwargs = _get_compute_kwargs(config=config)

    results: dict[str, list[tuple[float, float, bool]]] = {}
    for label, home_path in config.home_paths.items():
        if config.ensemble_norm_by_label is not None:
            if label not in config.ensemble_norm_by_label:
                raise KeyError(
                    f"Missing ensemble_norm_by_label entry for label '{label}'"
                )
            ensemble_cfg = config.ensemble_norm_by_label[label]
        elif config.ensemble_norm is not None:
            ensemble_cfg = config.ensemble_norm
        else:
            raise ValueError(
                "Provide either 'ensemble_norm' or 'ensemble_norm_by_label' in config."
            )

        ensemble_std = compute_ensemble_perturbation_std(
            ensemble_norm=ensemble_cfg,
            chunks=config.chunks,
            compute_kwargs=compute_kwargs,
        )
        _logger.info(
            "Ensemble perturbation normalization std-dev for %s: %.6f",
            label,
            ensemble_std,
        )

        _logger.info("Collecting experiment outputs from %s (%s)", label, home_path)
        triples = list(
            load_experiment_outputs(
                dir=home_path,
                ref_target=config.ref_target_for_label(label),
                search_region=config.search_region,
                chunks=config.chunks,
                **compute_kwargs,
            )
        )

        normalized_triples = [(d, m / ensemble_std, b) for d, m, b in triples]
        results[label] = normalized_triples
        _logger.info("Loaded %d runs for %s", len(normalized_triples), label)

    out = plot_perturbation_vs_distance(results, config)
    _logger.info("Saved plot to %s", out)


if __name__ == "__main__":
    main()
