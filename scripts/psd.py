import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import xarray as xr

# Ensure project root is importable when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spectra
from psd_config import PSDConfig

import dataflows
from common_utils import get_max_epoch_in_dir


def load_dataset(dataset_cfg) -> xr.Dataset:
    print(f"Loading dataset from {dataset_cfg.path}...")
    if os.path.isdir(dataset_cfg.path):
        file, _ = get_max_epoch_in_dir(dataset_cfg.path)
    else:
        assert os.path.isfile(dataset_cfg.path)
        file = dataset_cfg.path

    ds = dataflows.load_dataset(file)
    if dataset_cfg.time_slice is not None:
        if isinstance(dataset_cfg.time_slice, list):
            ds = ds.isel(time=dataset_cfg.time_slice)
        else:
            ds = ds.isel(time=slice(*dataset_cfg.time_slice))
    return ds


def compute_avg_psd(
    ds: xr.Dataset,
    variable: str,
    level: Optional[int],
    lmax: Optional[int] = None,
    lat_band=None,
) -> xr.DataArray:

    da = spectra.subset_to_3D_grid(ds, variable, level)
    alm = spectra.xarray_to_alm_shtools(da, lmax=lmax)

    if "time" in alm.dims:
        psd_list = []
        for t in alm.time:
            alm_t = alm.sel(time=t)
            psd_t = spectra.alm_to_zonal_power_sectrum(
                alm_t, lmax=lmax, lat_band=lat_band
            )
            psd_list.append(psd_t)
        return xr.concat(psd_list, dim="time").mean("time")
    else:
        return spectra.alm_to_zonal_power_sectrum(alm, lmax=lmax, lat_band=lat_band)


def subset_ds(ds: xr.Dataset, variables: list[str], level: Optional[int]) -> xr.Dataset:
    if "10m_kinetic_energy" in variables:
        ds["10m_kinetic_energy"] = 0.5 * (
            ds["10m_u_component_of_wind"] ** 2 + ds["10m_v_component_of_wind"] ** 2
        )
    elif "kinetic_energy" in variables:
        ds["kinetic_energy"] = 0.5 * (
            ds["u_component_of_wind"] ** 2 + ds["v_component_of_wind"] ** 2
        )
    else:
        pass

    ds = ds[variables]

    if level is not None:
        level_dim = "level"
        assert level_dim in ds.dims
        ds = ds.sel({level_dim: level})

    return ds


def plot_psd(cfg: PSDConfig):
    if not cfg.variable_levels:
        raise ValueError(
            "PSDConfig.variable_levels must be set to a list of (variable, level) tuples"
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fallback_colors = ["k", "olive", "cyan", "purple", "orange", "blue", "red"]

    def _compute_for_dataset(args):
        label, ds_cfg = args
        ds = load_dataset(ds_cfg)
        ds = subset_ds(ds, [variable], level)
        psd = compute_avg_psd(ds, variable, level, lmax=cfg.lmax, lat_band=cfg.lat_band)
        return label, ds_cfg, psd

    for variable, level in cfg.variable_levels:
        fig, ax = plt.subplots(figsize=(8, 6))

        results = list(map(_compute_for_dataset, cfg.datasets.items()))

        if cfg.relative is not None:
            ref_result = [r for r in results if r[0] == cfg.relative]
            assert len(ref_result) == 1
            ref_psd = ref_result[0][2]
            relative_title = f" (relative to {cfg.relative})"
        else:
            ref_psd = 1
            relative_title = ""

        for i, (label, ds_cfg, psd) in enumerate(results):
            color = ds_cfg.color or fallback_colors[i % len(fallback_colors)]
            spectra.plot_spectrum(psd / ref_psd, "m", ax, label=label, color=color)

        level_str = f" at {level} hPa" if level is not None else ""

        ax.set_title(
            f"Spherical Harmonic Power Spectra {relative_title} - {variable} {level_str}"
        )
        ax.set_xlabel("Total wavenumber")
        ax.legend()
        plt.tight_layout()

        suffix = f"{variable}_{level_str}"
        out_path = output_dir / f"{cfg.output_path_root_name}_{suffix}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved PSD figure to {out_path}")


def load_config(path: str) -> PSDConfig:
    with open(path, "r") as f:
        data = json.load(f)
    return PSDConfig(**data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PSD from dict-based config")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    plot_psd(cfg)
