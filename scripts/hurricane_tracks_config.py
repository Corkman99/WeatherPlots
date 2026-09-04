import os
import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseTrackStyleSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    colormap: str = "coolwarm"
    icon: Optional[str] = None
    color: Optional[str] = None
    linestyle: Optional[str] = "-"
    marker: Optional[str] = "o"
    linewidth: Optional[float] = None
    lwd: Optional[float] = None
    line_alpha: float = 0.6
    marker_alpha: float = 0.6
    marker_size: Optional[float] = Field(
        None,
        description=(
            "Endpoint marker size for this dataset. For line-based endpoints,"
            " this maps to Matplotlib 'markersize'; for scatter-based endpoints,"
            " this maps to Matplotlib scatter 's'."
        ),
    )
    timeframe: Optional[list[int]] = None
    # Add more plotting kwargs as needed
    specs: dict[str, Any] = Field(default_factory=dict)


class DatasetTrackSpec(BaseTrackStyleSpec):
    path: str
    input_path: Optional[str] = None
    show_in_legend: bool = True


class EnsembleTrackSpec(BaseTrackStyleSpec):
    directory: str
    pattern: str
    input_directory: Optional[str] = None
    input_pattern: Optional[str] = None
    member_ids: Optional[list[str]] = None
    member_label_template: str = "{dataset_name} member {member_number}"

    @model_validator(mode="after")
    def validate_member_pattern(self):
        has_member_token = "{member_number}" in self.pattern or "{seed}" in self.pattern
        if not has_member_token:
            raise ValueError(
                "Ensemble pattern must contain '{member_number}' or '{seed}' token "
                "(e.g. 'run_s{member_number}.nc' or 'run_s{seed}.nc')."
            )
        if self.input_pattern and (
            "{member_number}" not in self.input_pattern
            and "{seed}" not in self.input_pattern
        ):
            raise ValueError(
                "input_pattern must contain '{member_number}' or '{seed}' token when provided."
            )
        return self


TrackDatasetSpec = DatasetTrackSpec | EnsembleTrackSpec


class HurricaneTrackConfig(BaseModel):
    output_file: str = "hurricane_tracks.png"
    region: tuple[float, float, float, float] = (14, 250, 35, 286)
    datasets: dict[str, TrackDatasetSpec]
    figsize: tuple[int, int] = (10, 12)
    legend_loc: str = "right"

    @model_validator(mode="after")
    def validate_datasets(self):
        if not self.datasets:
            raise ValueError("At least one dataset must be specified in 'datasets'.")
        for name, spec in self.datasets.items():
            if isinstance(spec, DatasetTrackSpec) and not spec.path:
                raise ValueError(f"Dataset '{name}' must have a 'path'.")
            if isinstance(spec, EnsembleTrackSpec) and not spec.directory:
                raise ValueError(f"Ensemble dataset '{name}' must have a 'directory'.")
        return self

    @staticmethod
    def _render_member_pattern(pattern: str, member_number: str) -> str:
        rendered = pattern.replace("{member_number}", str(member_number))
        return rendered.replace("{seed}", str(member_number))

    @staticmethod
    def _member_sort_key(member_number: str):
        try:
            return (0, int(member_number))
        except ValueError:
            return (1, member_number)

    @classmethod
    def _discover_ensemble_members(cls, directory: str, pattern: str) -> list[str]:
        escaped = re.escape(pattern)
        regex_pattern = escaped.replace(
            re.escape("{member_number}"), r"(?P<member_number>[^/]+)"
        )
        regex_pattern = regex_pattern.replace(
            re.escape("{seed}"), r"(?P<member_number>[^/]+)"
        )
        regex = re.compile("^" + regex_pattern + "$")

        try:
            filenames = os.listdir(directory)
        except FileNotFoundError as err:
            raise ValueError(f"Ensemble directory does not exist: {directory}") from err

        members = []
        for filename in filenames:
            match = regex.match(filename)
            if match:
                members.append(match.group("member_number"))

        members = sorted(set(members), key=cls._member_sort_key)
        if not members:
            raise ValueError(
                "No ensemble files matched pattern "
                f"'{pattern}' in directory '{directory}'."
            )
        return members

    def expand_datasets(self) -> dict[str, DatasetTrackSpec]:
        """Expand any ensemble entries into flat DatasetTrackSpec entries.

        Existing non-ensemble datasets are kept unchanged.
        """
        expanded: dict[str, DatasetTrackSpec] = {}

        for dataset_name, spec in self.datasets.items():
            if isinstance(spec, DatasetTrackSpec):
                expanded[dataset_name] = spec
                continue

            members = (
                spec.member_ids
                if spec.member_ids is not None
                else self._discover_ensemble_members(spec.directory, spec.pattern)
            )

            if not members:
                raise ValueError(f"Ensemble dataset '{dataset_name}' has no members.")

            for member_idx, member in enumerate(members):
                member_str = str(member)
                output_filename = self._render_member_pattern(spec.pattern, member_str)
                output_path = os.path.join(spec.directory, output_filename)
                if not os.path.exists(output_path):
                    raise ValueError(
                        f"Resolved ensemble file does not exist for dataset '{dataset_name}': "
                        f"{output_path}"
                    )

                input_path = None
                if spec.input_pattern:
                    input_dir = spec.input_directory or spec.directory
                    input_filename = self._render_member_pattern(
                        spec.input_pattern, member_str
                    )
                    input_path = os.path.join(input_dir, input_filename)
                    if not os.path.exists(input_path):
                        raise ValueError(
                            "Resolved ensemble input file does not exist for dataset "
                            f"'{dataset_name}': {input_path}"
                        )

                label = spec.member_label_template.format(
                    dataset_name=dataset_name,
                    member_number=member_str,
                    seed=member_str,
                )
                if label in expanded:
                    raise ValueError(
                        f"Expanded dataset label '{label}' is duplicated. "
                        "Adjust member_label_template to ensure unique names."
                    )

                expanded[label] = DatasetTrackSpec(
                    path=output_path,
                    input_path=input_path,
                    show_in_legend=(member_idx == 0),
                    colormap=spec.colormap,
                    icon=spec.icon,
                    color=spec.color,
                    linestyle=spec.linestyle,
                    marker=spec.marker,
                    line_alpha=spec.line_alpha,
                    marker_alpha=spec.marker_alpha,
                    timeframe=spec.timeframe,
                    specs=spec.specs,
                )

        return expanded
