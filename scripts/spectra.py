"""
Functions to retrieve power density spectra from xarray datasets, per variable, level and time
Functions to return requested spectra data (for single variable/level/time) for plotting
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes


def subset_to_3D_grid(
    ds: xr.Dataset, var_name: str, level: Optional[int]
) -> xr.DataArray:
    """Subset the dataset for a given variable, level and valid time"""
    assert var_name in ds.data_vars, f"Variable {var_name} not in dataset."
    if level is not None:
        assert level in ds["level"], f"Level {level} not in dataset."
        da = ds.sel(level=level)
        assert var_name in da.data_vars, f"Variable {var_name} not a level_variable."
    else:
        da = ds

    dims = set(da.dims)
    if "time" in da.dims:
        dims -= {"time"}
    if "level" in da.dims:
        dims -= {"level"}
    assert dims == {"lat", "lon"}, f"dims: {dims} but expecting only ('lat', 'lon')"

    return da[var_name]


def alm_to_xarray(alm: np.ndarray, lmax: int) -> xr.DataArray:
    """
    Convert healpy alm array to xarray.DataArray with dimensions (l, m).

    Parameters
    ----------
    alm : np.ndarray
        healpy alm array (1D, m >= 0 only).
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    xr.DataArray
        DataArray with dims ('l', 'm'), complex-valued.
        Entries with m > l are NaN.
    """
    import healpy

    data = np.full((lmax + 1, lmax + 1), np.nan + 1j * np.nan, dtype=np.complex128)

    for idx in range(len(alm)):
        l, m = healpy.Alm.getlm(lmax, idx)
        data[l, m] = alm[idx]

    return xr.DataArray(
        data,
        dims=("l", "m"),
        coords={
            "l": np.arange(lmax + 1),
            "m": np.arange(lmax + 1),
        },
        name="alm",
    )


def xarray_to_alm_healpy(
    da: xr.DataArray,
    nside: int = 256,  # translates to 0.23 degree resolution
    lmax: int = 256,
):
    """
    Compute spherical harmonic coefficients from a lat-lon xarray DataArray using Healpy.
    Healpy assumes an equal-area regular grid on the sphere, so we interpolate the input data onto
    a HEALPix grid before computing the spherical harmonics.
    This interpolation allows for local spectral analysis by masking. See xarray_to_alm_healy_local.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with dimensions ('lat', 'lon').
    nside : int
        HEALPix nside parameter.
    lmax : int, optional
        Maximum multipole. Defaults to 3*nside - 1.

    Returns
    -------
    alm : np.ndarray
        Complex spherical harmonic coefficients.
    """

    import healpy
    from scipy.interpolate import RegularGridInterpolator

    # TODO: generalize to accept time dimension as well
    # Interpolator expects increasing coordinates
    da_sorted = da.sortby(["lat", "lon"])
    lat_sorted = da_sorted["lat"].values
    lon_sorted = np.mod(da_sorted["lon"].values, 360.0)
    values = da_sorted.values

    interp = RegularGridInterpolator(
        (lat_sorted, lon_sorted),
        values,
        bounds_error=False,
        fill_value=np.nan,
    )

    # HEALPix pixel centers
    npix = healpy.nside2npix(nside)
    theta, phi = healpy.pix2ang(nside, np.arange(npix))

    # Convert to lat/lon in degrees
    lat_hp = np.degrees(0.5 * np.pi - theta)
    lon_hp = np.degrees(phi)

    points = np.stack([lat_hp, lon_hp], axis=-1)
    hp_map = interp(points)

    # Optional: handle missing values
    mask = np.isnan(hp_map)
    if np.any(mask):
        hp_map[mask] = 0.0

    alm = healpy.map2alm(hp_map, lmax=lmax)

    return alm_to_xarray(alm, lmax)


def xarray_to_alm_shtools(da: xr.DataArray, lmax: Optional[int] = 256) -> xr.DataArray:
    """
    Compute SHTOOLS spherical harmonic coefficients from a regular lat-lon xarray DataArray
    and return as xarray.DataArray with dims (l, m) or (time, l, m).

    Parameters
    ----------
    da : xr.DataArray
        Input with dims ('lat', 'lon') or ('time', 'lat', 'lon').
        lat must be descending (north -> south).
    lmax : int, optional
        Maximum spherical harmonic degree.

    Returns
    -------
    alm_xr : xr.DataArray
        Complex spherical harmonic coefficients with dims (l, m) or (time, l, m).
        Shape: (lmax+1, lmax+1) or (time, lmax+1, lmax+1)
    """

    import pyshtools

    # Ensure lats descending and remove -90° S if present
    # to work with Driscoll and Healy (1994) grid
    if da.lat[0] < da.lat[-1]:
        da = da.sortby("lat", ascending=False)
    da = da.sel(lat=da.lat > -90)

    # Determine if time dimension exists
    if "time" in da.dims:
        ntime = da.sizes["time"]
        times = da["time"]
    else:
        ntime = 1
        da = da.expand_dims("time")
        times = da["time"]

    if lmax is None:
        lmax = da.sizes["lat"] // 2 - 1
        print(f"Setting lmax to {lmax} based on data resolution.")

    # Prepare output array
    coeffs_array = np.zeros((ntime, lmax + 1, lmax + 1), dtype=np.complex128)

    # Loop over time
    for i, time_val in enumerate(da["time"]):
        data = da.sel(time=time_val).values  # shape (nlat, nlon)
        clm = pyshtools.expand.SHExpandDH(
            data,
            sampling=2,
            lmax_calc=lmax,
        )
        # Convert to complex: a_lm = cos - i*sin
        coeffs_complex = clm[0, ...] - 1j * clm[1, ...]  # shape (l, m)
        coeffs_array[i, :, :] = coeffs_complex

    # Build xarray
    dims = ("time", "l", "m")
    coords = {"time": times, "l": np.arange(lmax + 1), "m": np.arange(lmax + 1)}

    alm_xr = xr.DataArray(coeffs_array, dims=dims, coords=coords, name="alm")
    # Remove extra time dim if input had none
    if ntime == 1:
        alm_xr = alm_xr.squeeze("time", drop=True)

    return alm_xr


def alm_to_angular_power_spectrum(alm: xr.DataArray) -> xr.DataArray:
    """
    Compute angular power spectrum C_l from an xarray alm(l, m).

    Parameters
    ----------
    alm_xr : xr.DataArray
        Complex spherical harmonic coefficients with dims ('l', 'm').
        Entries with m > l should be NaN.

    Returns
    -------
    xr.DataArray
        Power spectral density C_l with dim ('l',).
    """
    # Ensure we only use valid (l, m)
    alm_valid = alm.where(alm.m <= alm.l)

    # Power |alm|^2
    power_lm = xr.DataArray(np.abs(alm_valid) ** 2)

    # m = 0 term
    power_m0 = power_lm.sel(m=0)

    # m >= 1 terms (counted twice)
    power_mpos = power_lm.where(alm.m > 0).sum("m")

    # Full sum over m = -l ... l
    power_sum = power_m0 + 2.0 * power_mpos

    # Normalize by (2l + 1)
    cl = power_sum / (2 * alm.l + 1)

    return cl.rename("Cl")


def alm_to_zonal_power_sectrum(
    alm_xr: xr.DataArray, lmax: Optional[int] = None, lat_band=None
) -> xr.DataArray:
    """
    Compute zonal wavenumber spectrum P(m) from alm_xr(l, m).
    Optionally integrate over a lat band (lat_band = (lat1, lat2)).

    Parameters
    ----------
    alm_xr : xr.DataArray
        Complex SH coefficients, dims (l, m)
    lmax : int
        Maximum degree to include (if None, use alm_xr.l.max())
    lat_band : tuple(float,float)
        lat range to integrate over (phi1, phi2), in degrees

    Returns
    -------
    Pm : xr.DataArray
        Zonal power spectrum, dims (m,)
    """

    import pyshtools

    if lmax is None:
        lmax = int(alm_xr.l.max())

    # Create SHCoeffs object from alm
    clm = np.zeros((2, lmax + 1, lmax + 1))
    # fill in real and imag parts
    for li in range(lmax + 1):
        for mi in range(li + 1):
            c = np.real(alm_xr.sel(l=li, m=mi).values)
            s = -np.imag(alm_xr.sel(l=li, m=mi).values)  # note: SHExpandDH convention
            clm[0, li, mi] = c
            clm[1, li, mi] = s

    coeffs = pyshtools.SHCoeffs.from_array(clm)

    Pm_list = []

    for mi in alm_xr.m.values:
        # zero all other m
        clm_m = np.zeros_like(clm)
        clm_m[:, : lmax + 1, : lmax + 1] = clm  # copy
        clm_m[:, :, :] = 0  # zero everything
        for li in range(mi, lmax + 1):
            clm_m[0, li, mi] = clm[0, li, mi]
            clm_m[1, li, mi] = clm[1, li, mi]

        coeffs_m = pyshtools.SHCoeffs.from_array(clm_m)
        assert coeffs_m is not None

        # inverse transform to grid
        grid = coeffs_m.expand(grid="DH")  # 2D field

        # select lat band if specified
        if lat_band is not None:
            phi1, phi2 = lat_band
            lat_mask = (grid.lat >= phi1) & (grid.lat <= phi2)
            grid_vals = grid.data[lat_mask, :]
        else:
            grid_vals = grid.data

        # square and integrate
        Pm_val = np.mean(grid_vals**2)  # integration over lat/lon via mean
        Pm_list.append(Pm_val)

    Pm = xr.DataArray(Pm_list, dims=("m",), coords={"m": alm_xr.m})
    return Pm


def convert_m_to_physical_wavenumber(power_spectrum: xr.DataArray, lat_deg=0.0):
    """
    Convert the 'm' dimension of a PSD DataArray to physical zonal wavenumber in km^-1.

    Parameters
    ----------
    power_spectrum : xr.DataArray
        DataArray with dimension 'm' (can also have 'l' or 'time').
    lat_deg : float
        lat in degrees at which to compute the zonal wavenumber. Default = 0°.

    Returns
    -------
    psd_phys : xr.DataArray
        Same DataArray, but with new coordinate 'k_lambda' in km^-1
    """
    a = 6371.0  # Earth radius in km
    phi = np.deg2rad(lat_deg)

    m = power_spectrum["m"]
    k_lambda = m / (2 * np.pi * a * np.cos(phi))  # km^-1

    psd_phys = power_spectrum.assign_coords(k_lambda=k_lambda)
    return psd_phys


def plot_spectrum(cl: xr.DataArray, dim: str, ax: Axes, **kwargs) -> Axes:
    """
    Plot power spectrum C_l using matplotlib.

    Parameters
    ----------
    cl : xr.DataArray
        Power spectral density C_l with dim ('l',).
    """

    ax.plot(cl[dim], cl, **kwargs)
    ax.set_xlabel(dim)
    ax.set_ylabel("Power spectral density")
    ax.set_yscale("log")
    ax.set_xscale("log")

    return ax


def compare_methods():

    test_data_1 = xr.open_dataset(
        "/home/users/f/froelicm/experimentC/root_outputs/hres-f0&2022-05-06T00&aifs_v1&24.nc",
        engine="netcdf4",
    )
    test_data_1 = subset_to_3D_grid(test_data_1, "temperature", level=500)
    test_data_1 = test_data_1.isel(time=-1)

    test_data_2 = xr.open_dataset(
        "/home/users/f/froelicm/experimentC/root_outputs/hres-f0&2022-05-04T00&aifs_v1&72.nc",
        engine="netcdf4",
    )
    test_data_2 = subset_to_3D_grid(test_data_2, "temperature", level=500)
    test_data_2 = test_data_2.isel(time=-1)

    fig, ax = plt.subplots()

    das = [test_data_1, test_data_2]
    labels = ["24h forecast", "72h forecast"]
    colors = ["blue", "orange"]

    for i in range(len(das)):
        alm = xarray_to_alm_healpy(das[i], lmax=256)
        cm = alm_to_zonal_power_sectrum(alm, lmax=None, lat_band=None)
        ax = plot_spectrum(cm, "m", ax, color=colors[i], label=f"{labels[i]} (healpy)")

        alm = xarray_to_alm_shtools(das[i], lmax=256)
        cm = alm_to_zonal_power_sectrum(alm, lmax=None, lat_band=None)
        ax = plot_spectrum(
            cm,
            "m",
            ax,
            color=colors[i],
            linestyle="--",
            label=f"{labels[i]} (shtools)",
        )
    ax.legend()
    fig.savefig("test_methods_spectra.png")


def compare_level_spectra():
    test_data = xr.open_dataset(
        "/home/users/f/froelicm/experimentC/root_outputs/hres-f0&2022-05-04T00&aifs_v1&72.nc",
        engine="netcdf4",
    )
    test_data = test_data.isel(time=-1, drop=True)

    test_data["kinetic energy"] = 0.5 * (
        test_data["u_component_of_wind"] ** 2 + test_data["v_component_of_wind"] ** 2
    )
    test_data = test_data[
        ["geopotential", "kinetic energy", "specific_humidity", "temperature"]
    ]
    test_data = test_data.sel(level=[1000, 850, 500, 250])

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax_idx = {
        "geopotential": (0, 0),
        "kinetic energy": (0, 1),
        "specific_humidity": (1, 0),
        "temperature": (1, 1),
    }
    colors = {
        1000: "blue",
        850: "orange",
        500: "green",
        250: "red",
    }
    for var in test_data.data_vars:
        for level in test_data["level"].values:
            da = subset_to_3D_grid(test_data, str(var), level)
            alm = xarray_to_alm_shtools(da, lmax=None)
            cm = alm_to_zonal_power_sectrum(alm, lmax=None, lat_band=None)

            ax_plot = ax[ax_idx[str(var)]]
            ax_plot = plot_spectrum(
                cm,
                "m",
                ax_plot,
                label=f"{level} hPa",
                color=colors[level],
            )
            ax_plot.set_title(str(var))

    ax[0, 1].legend()
    fig.savefig("test_level_spectra.png")


if __name__ == "__main__":

    compare_methods()
    compare_level_spectra()
