from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy
import xarray


def spatial_softmax(
    da: xarray.DataArray, spatial_dims: tuple[str, ...] = ("lat", "lon")
) -> xarray.DataArray:
    """Numerically stable softmax over the spatial grid."""
    reduce_dims = tuple(dim for dim in spatial_dims if dim in da.dims)
    if not reduce_dims:
        return da * 0.0 + 1.0

    if len(reduce_dims) == 1:
        working = da.rename({reduce_dims[0]: "_softmax_space"})
    else:
        working = da.stack(_softmax_space=reduce_dims)

    shifted = working - working.max(dim="_softmax_space")
    exp_shifted = xarray.apply_ufunc(numpy.exp, shifted)
    weights = exp_shifted / exp_shifted.sum(dim="_softmax_space")

    if len(reduce_dims) == 1:
        return weights.rename({"_softmax_space": reduce_dims[0]})
    return weights.unstack("_softmax_space")


def smooth_noise(
    rng: numpy.random.Generator,
    shape: tuple[int, int],
    kernel: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """Create simple smoothed noise without requiring scipy."""
    resolved_kernel = kernel
    if resolved_kernel is None:
        resolved_kernel = numpy.array(
            [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=numpy.float64,
        )
    resolved_kernel = resolved_kernel / resolved_kernel.sum()
    raw = rng.standard_normal(shape)
    pad_y = resolved_kernel.shape[0] // 2
    pad_x = resolved_kernel.shape[1] // 2
    padded = numpy.pad(raw, ((pad_y, pad_y), (pad_x, pad_x)), mode="wrap")
    smoothed = numpy.zeros_like(raw)
    for iy in range(shape[0]):
        for ix in range(shape[1]):
            window = padded[
                iy : iy + resolved_kernel.shape[0], ix : ix + resolved_kernel.shape[1]
            ]
            smoothed[iy, ix] = numpy.sum(window * resolved_kernel)
    return smoothed


def build_demo_fields(
    n_lat: int = 721,
    n_lon: int = 1440,
    seed: int = 7,
) -> tuple[xarray.DataArray, xarray.DataArray]:
    """Return a dummy perturbation field and its softmax weight field."""
    lat = numpy.linspace(-90.0, 90.0, n_lat)
    lon = numpy.linspace(0.0, 360.0, n_lon, endpoint=False)
    lon_grid, lat_grid = numpy.meshgrid(lon, lat)

    rng = numpy.random.default_rng(seed)
    base_noise = smooth_noise(rng, (n_lat, n_lon))

    wave = 0.75 * numpy.sin(numpy.deg2rad(lat_grid * 1.7))
    wave += 0.55 * numpy.cos(numpy.deg2rad(lon_grid * 1.2))
    hotspot = 1.8 * numpy.exp(
        -(((lat_grid - 18.0) / 18.0) ** 2 + ((lon_grid - 215.0) / 28.0) ** 2)
    )
    noise = base_noise + wave + hotspot

    noise_field = xarray.DataArray(
        noise,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="noise",
    )
    weight_field = spatial_softmax(abs(noise_field)).rename("softmax_weight")
    return noise_field, weight_field


def build_demo_fields_from_spherical_harmonics(
    n_lat: int = 720,
    n_lon: int = 1440,
    trunc_level: int = 42,
    seed: int = 7,
) -> tuple[xarray.DataArray, xarray.DataArray]:
    """Generate a random field by expanding random SH coefficients with pyshtools.

    Args:
        n_lat: Requested latitude count. Rounded to the nearest DH-compatible size.
        n_lon: Requested longitude count. Used to infer a DH-compatible latitude count.
        trunc_level: Maximum spherical harmonic degree (higher = more detail).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (noise_field, weight_field) as xarray DataArrays.
    """
    import pyshtools

    rng = numpy.random.default_rng(seed)
    dh_n_lat = max(2 * (trunc_level + 1), int(round(max(n_lat, n_lon // 2) / 2) * 2))
    lmax = min(trunc_level, dh_n_lat // 2 - 1)

    coeffs = numpy.zeros((2, lmax + 1, lmax + 1), dtype=numpy.float64)
    degree_scale = 1.0 / numpy.sqrt(numpy.arange(lmax + 1, dtype=numpy.float64) + 1.0)

    for degree in range(lmax + 1):
        coeffs[0, degree, : degree + 1] = (
            rng.standard_normal(degree + 1) * degree_scale[degree]
        )
        if degree > 0:
            coeffs[1, degree, 1 : degree + 1] = (
                rng.standard_normal(degree) * degree_scale[degree]
            )

    field = pyshtools.expand.MakeGridDH(coeffs, sampling=2, lmax=lmax)
    field = field / numpy.std(field)

    lat = numpy.linspace(90.0, -90.0 + 180.0 / field.shape[0], field.shape[0])
    lon = numpy.linspace(0.0, 360.0, field.shape[1], endpoint=False)

    noise_field = xarray.DataArray(
        field,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="noise",
    )
    weight_field = spatial_softmax(abs(noise_field)).rename("softmax_weight")
    return noise_field, weight_field


def plot_fields(
    noise_field: xarray.DataArray,
    weight_field: xarray.DataArray,
    output_path: Path,
    show: bool = False,
) -> None:
    """Plot the perturbation field and its softmax-derived weights."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    noise_plot = axes[0].pcolormesh(
        noise_field["lon"],
        noise_field["lat"],
        noise_field,
        shading="auto",
        cmap="RdBu_r",
    )
    axes[0].set_title("Dummy kernel noise")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(noise_plot, ax=axes[0], label="Noise amplitude")

    weight_plot = axes[1].pcolormesh(
        weight_field["lon"],
        weight_field["lat"],
        weight_field,
        shading="auto",
        cmap="copper_r",
        norm=matplotlib.colors.LogNorm(
            vmin=max(1e-6, weight_field.min().item()), vmax=weight_field.max().item()
        ),
    )
    axes[1].set_title("Softmax weights on |noise|")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    fig.colorbar(weight_plot, ax=axes[1], label="Weight")

    fig.suptitle("Spatial softmax weighting used by the perturbation penalty")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_output = script_dir / "penalty_visualization.png"
    parser = argparse.ArgumentParser(
        description="Plot dummy noise and the spatial softmax weights on |noise|."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Where to save the figure (default: {default_output})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    noise_field, weight_field = build_demo_fields_from_spherical_harmonics()
    plot_fields(noise_field, weight_field, args.output, show=args.show)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
