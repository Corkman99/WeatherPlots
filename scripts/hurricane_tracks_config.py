from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class DatasetTrackSpec(BaseModel):
    path: str
    colormap: str = "coolwarm"
    icon: Optional[str] = None
    input_path: Optional[str] = None
    color: Optional[str] = None
    linestyle: Optional[str] = "-"
    marker: Optional[str] = "o"
    timeframe: Optional[list[int]] = None
    # Add more plotting kwargs as needed
    specs: dict[str, Any] = Field(default_factory=dict)


class HurricaneTrackConfig(BaseModel):
    output_file: str = "hurricane_tracks.png"
    region: tuple[float, float, float, float] = (14, 250, 35, 286)
    datasets: dict[str, DatasetTrackSpec]
    figsize: tuple[int, int] = (10, 12)
    legend_loc: str = "right"

    @model_validator(mode="after")
    def validate_datasets(self):
        if not self.datasets:
            raise ValueError("At least one dataset must be specified in 'datasets'.")
        for name, spec in self.datasets.items():
            if not spec.path:
                raise ValueError(f"Dataset '{name}' must have a 'path'.")
        return self
