import logging
from typing import Optional

from pydantic import Field, model_validator

from scripts.hurricane_tracks_config import HurricaneTrackConfig


class InputDiffTrackConfig(HurricaneTrackConfig):
    """Config for input-difference-colored hurricane tracks.

    Datasets that have ``color`` set in their spec are plotted with that
    fixed colour (e.g. ground truth, base forecast).  All other datasets
    are coloured according to a GraphCast-style weighted MSE loss between
    their input and the reference input, using a ``BoundaryNorm`` discrete
    colormap whose boundaries are derived from the computed scores (or
    supplied explicitly via ``diff_boundaries``).

    Normalization statistics (mean / stddev) can be provided so that the
    MSE is computed in normalized space, matching the GraphCast training
    loss definition.
    """

    log_colorbar: bool = Field(
        False,
        description="If True, use a logarithmic scale for the colorbar (LogNorm). If False, use BoundaryNorm.",
    )
    colorbar_vmin: Optional[float] = Field(
        default=None,
        description="Minimum value for colorbar normalization (LogNorm/BoundaryNorm). If None, auto-calculated.",
    )
    colorbar_vmax: Optional[float] = Field(
        default=None,
        description="Maximum value for colorbar normalization (LogNorm/BoundaryNorm). If None, auto-calculated.",
    )
    reference_dataset: Optional[str] = Field(
        None,
        description="Dataset key whose input_path is the reference for diff computation",
    )
    plot_targets: Optional[dict] = Field(
        None,
        description=(
            "Optional config for plotting target points (e.g. landfall). "
            "Should be a dict with keys 'targets' (list of 2-list/tuple with 'lon',"
            "'lat', and 'kwargs' (dict of matplotlib scatter kwargs applied to all target points)."
        ),
    )
    search_region: Optional[tuple[float, float, float, float]] = Field(
        default=None,
        description="Optional region (min_lat, min_lon, max_lat, max_lon) to use for searching hurricane centers. If None, uses the plotting region. Allows extending the search window to avoid edge artefacts.",
    )
    highlight_inputs: bool = Field(
        False,
    )
    font_size: Optional[int] = Field(
        default=None,
        description="Base font size for all plot text. If None, auto-calculated from figure size.",
    )

    @model_validator(mode="after")
    def set_default_reference_dataset(self):
        # reference_dataset only required when diff-colored datasets exist.
        expanded_datasets = self.expand_datasets()
        diff_datasets = [s for s in expanded_datasets.values() if not s.color]
        if diff_datasets and self.reference_dataset is None:
            first_dataset = next(iter(expanded_datasets), None)
            if first_dataset is None:
                raise ValueError("No datasets available to infer reference_dataset")
            logging.warning(
                "reference_dataset not specified; defaulting to first dataset '%s' for diff calculation",
                first_dataset,
            )
            object.__setattr__(self, "reference_dataset", first_dataset)
        return self

    diff_colormap: str = Field(
        "RdYlBu_r",
        description="Matplotlib colormap name used for diff-coloured tracks",
    )
    diff_boundaries: Optional[list[float]] = Field(
        None,
        description=(
            "Explicit boundary values for BoundaryNorm. "
            "If None, boundaries are auto-derived from computed diff scores."
        ),
    )
    extend: str = Field(
        "neither",
        description=(
            "Direction to extend colormap for out-of-bounds values. "
            "One of 'neither', 'both', 'min', 'max'."
        ),
    )
    colorbar_label: str = Field(
        "Weighted MSE (input diff)",
        description="Label for the colorbar axis",
    )
    title: Optional[str] = Field(
        "Hurricane Tracks Coloured by Normalized Weighted MSE of Input Perturbation",
        description="Figure title",
    )

    # -- Normalization statistics --------------------------------------------
    norm_scales_path: Optional[str] = Field(
        None,
        description=(
            "Path to a netCDF / zarr file containing per-variable standard "
            "deviations used to normalize before computing MSE.  Variables "
            "are matched by name."
        ),
    )
    norm_locations_path: Optional[str] = Field(
        None,
        description=(
            "Path to a netCDF / zarr file containing per-variable means "
            "(locations) used to center before computing MSE."
        ),
    )

    # -- Weighting -----------------------------------------------------------
    level_weight_map: Optional[dict[int, float]] = Field(
        None,
        description=(
            "Mapping from pressure level (hPa, int) to weight.  "
            "If None, all levels are weighted uniformly."
        ),
    )
    per_variable_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-variable weights for the MSE loss.  Variables not listed "
            "(or with weight 0) are excluded from the loss.  An empty dict "
            "means all variables are weighted equally (weight 1)."
        ),
    )
