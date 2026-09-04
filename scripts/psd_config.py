from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, FilePath


class PSDDataset(BaseModel):
    path: FilePath | str
    time_slice: Optional[Union[Tuple[Optional[int], Optional[int]], List[int]]] = None
    color: Optional[str] = None
    variable_weights: Optional[dict] = None


class PSDConfig(BaseModel):
    datasets: Dict[str, PSDDataset]
    variable_levels: Optional[List[Tuple[str, Union[int, None]]]] = None
    lmax: Optional[int] = None
    lat_band: Optional[Tuple[float, float]] = None
    output_dir: str = "./outputs/psd"
    output_path_root_name: str = "psd"
    relative: Optional[str] = None
