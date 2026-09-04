import numpy as np
import xarray as xr

_RD = 287.0
_VIRTUAL_TEMP_COEFF = 0.609133


def hydrostatic_balance_violation(ds: xr.Dataset) -> xr.Dataset:
    """Return deviation from a surface-anchored hydrostatic geopotential profile.

    This computes virtual temperature from ``temperature`` and
    ``specific_humidity`` and integrates the hypsometric relation in pressure
    coordinates. Integration proceeds from high pressure to low pressure and is
    anchored by ``geopotential_at_surface``.

    Required variables:
    - ``temperature(time, level, lat, lon)``
    - ``specific_humidity(time, level, lat, lon)``
    - ``geopotential(time, level, lat, lon)``
    """

    required = (
        "temperature",
        "specific_humidity",
        "geopotential",
        "geopotential_at_surface",
    )
    missing = [name for name in required if name not in ds.data_vars]
    if missing:
        raise ValueError(f"Dataset is missing required variables: {missing}")

    for name in ("temperature", "specific_humidity", "geopotential"):
        if "level" not in ds[name].dims:
            raise ValueError(f"Variable '{name}' must have a 'level' dimension.")

    # Sort by pressure from high to low so cumulative integration is monotonic
    # upward in the atmosphere.
    ds = ds.sortby("level", ascending=False)
    pressure_levels = ds["level"].astype(np.float64)
    assert pressure_levels.size > 1

    geopotential = ds["geopotential"]
    temperature = ds["temperature"]
    specific_humidity = ds["specific_humidity"]
    surface_geopotential = ds["geopotential_at_surface"]  # is a static variable

    virtual_temperature = temperature * (1.0 + _VIRTUAL_TEMP_COEFF * specific_humidity)

    # For adjacent pressure levels p_k (below) and p_{k+1} (above):
    # Delta Phi = R_d * T_v_bar * ln(p_k / p_{k+1})
    upper_levels = pressure_levels.isel(level=slice(1, None))
    dlogp = xr.DataArray(
        np.log(
            pressure_levels.isel(level=slice(None, -1)).values
            / pressure_levels.isel(level=slice(1, None)).values
        ),
        dims=("level",),
        coords={"level": upper_levels},
    )

    tv_below = virtual_temperature.isel(level=slice(None, -1)).assign_coords(
        level=upper_levels
    )
    tv_above = virtual_temperature.isel(level=slice(1, None))
    tv_layer_mean = 0.5 * (tv_below + tv_above)
    delta_phi = _RD * tv_layer_mean * dlogp

    hydrostatic_sorted = xr.concat(
        [
            surface_geopotential.expand_dims(
                level=[float(pressure_levels.isel(level=0).values)]
            ),
            surface_geopotential + delta_phi.cumsum(dim="level"),
        ],
        dim="level",
    ).assign_coords(level=pressure_levels)

    hydrostatic_geopotential = hydrostatic_sorted.sel(level=geopotential["level"])
    hydrostatic_balance_error = geopotential - hydrostatic_geopotential
    return xr.Dataset({"hydrostatic_balance_error": hydrostatic_balance_error})
