from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class PlotFieldSpec(BaseModel):
    variable: str
    level: Optional[int] = None
    specs: dict[str, Any] = Field(default_factory=dict)


class ArrowSpec(BaseModel):
    variable: list[str]
    specs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("variable")
    @classmethod
    def validate_vector_pair(cls, value: list[str]) -> list[str]:
        if len(value) != 2:
            raise ValueError(
                "arrows.variable must contain exactly two variables: [u, v]."
            )
        return value


class HurricaneMapConfig(BaseModel):
    folder: str
    output_file: str = "ian_match_hres.png"
    experiment_config_file: str = "experiment_config.json"

    load_ground_truth: bool = True
    plot_ground_truth: bool = True
    ground_truth_path: Optional[str] = None

    plot_inputs_and_outputs: bool = False
    epochs: list[int] = Field(default_factory=lambda: [0, 1, 2, 5])

    region: Optional[tuple[float, float, float, float]] = (20, 272, 28, 284)
    columns: dict[str, int] = Field(default_factory=lambda: {"landfall": 0})

    fcontour: PlotFieldSpec
    contour: Optional[PlotFieldSpec] = None
    arrows: Optional[ArrowSpec] = None

    land_color: str = "#E3DFBF"

    figsize: tuple[int, int] = (18, 12)
    colormap_label: str = "Field"
    colormap_position: list[float] = Field(
        default_factory=lambda: [0.02, 0.15, 0.02, 0.7]
    )

    @model_validator(mode="after")
    def validate_ground_truth_path(self):
        if self.load_ground_truth and not self.ground_truth_path:
            raise ValueError(
                "ground_truth_path must be provided when load_ground_truth is true."
            )
        if not self.columns:
            raise ValueError(
                "columns must contain at least one column name to time index mapping."
            )
        if not self.epochs:
            raise ValueError("epochs must contain at least one epoch index.")
        if any(epoch < 0 for epoch in self.epochs):
            raise ValueError("epochs must contain non-negative integers.")
        return self
