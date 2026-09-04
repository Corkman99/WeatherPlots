from datetime import datetime
from typing import Optional

import xarray
from hurricane_maps import (
    DEFAULT_CONFIG_PATH,
    HurricaneMapConfig,
    argparse,
    cast,
    ccrs,
    create_multi_panel_figure,
    load_config,
    np,
    os,
    plot_item_to_panel_spec,
    plot_map_panel,
    plt,
    prepare_dataset,
    required_variables_from_config,
    selected_times_from_columns,
    to_iso_time_string,
    useful_hurricane_stats,
)


def create_projection(
    projection_type: str = "PlateCarree",
    region: Optional[tuple[float, float, float, float]] = None,
    satellite_height: float = 35786000.0,
) -> ccrs.Projection:
    """
    Create a cartopy projection.

    Parameters:
    - projection_type: "PlateCarree" or "NearsidePerspective"
    - region: (minlat, minlon, maxlat, maxlon) for NearsidePerspective centering
    - satellite_height: height in meters for NearsidePerspective (default: ~GEO satellite height)

    Returns:
    - cartopy projection object
    """
    if projection_type == "NearsidePerspective":
        if region is None:
            raise ValueError(
                "region must be provided for NearsidePerspective projection"
            )

        minlat, minlon, maxlat, maxlon = region
        central_lat = (minlat + maxlat) / 2
        central_lon = (minlon + maxlon) / 2

        return ccrs.NearsidePerspective(
            central_latitude=central_lat,
            central_longitude=central_lon,
            satellite_height=satellite_height,
        )
    else:
        return ccrs.PlateCarree()


def data_loading_and_preprocess(path: str) -> xarray.Dataset:
    ds = xarray.load_dataset(path, engine="cfgrib")
    # check resolution
    # rename axes
    dt = np.datetime64(datetime.fromisoformat("2020-05-21T12:00:00"))
    ftd = np.timedelta64(48, "h")
    ds = ds.rename(
        {
            "isobaricInhPa": "level",
            "z": "geopotential",
            "t": "temperature",
            "latitude": "lat",
            "longitude": "lon",
        }
    )
    ds = ds.sel(time=dt + ftd, drop=False).expand_dims("time")
    ds = ds.drop_vars(["step", "number", "valid_time"])

    ds = ds.sortby("lat")
    return ds


def zhongwei_config():
    fcontour = {
        "variable": "temperature",
        "level": 850,
        "specs": {
            "cmap": "Spectral_r",
            "levels": [x + 273.15 for x in np.arange(-10, 30, 2)],
            "extend": "max",
        },
    }
    contour = {
        "variable": "geopotential",
        "level": 850,
        "specs": {
            "levels": np.arange(13000, 16500, 200),
            "colors": "black",
            "label": True,
        },
    }
    # arrow = {
    #    "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
    #    "specs": {},
    # }
    config = HurricaneMapConfig(
        folder=".",
        output_file="zhongwei_fig.png",
        load_ground_truth=False,
        plot_ground_truth=False,
        plot_inputs_and_outputs=False,
        epochs=[0],
        pattern=None,
        region=(30, 320, 60, 50),  # EU: lat 30-60N, lon 30W-65E (as 330-65 in 0-360)
        columns={"landfall": 0},
        fcontour=fcontour,
        contour=contour,
        # arrows=arrow,
        figsize=(18, 12),
        colormap_label="2m temperature (K)",
    )
    # Use "PlateCarree" or "NearsidePerspective"
    projection_type = "PlateCarree"
    return config, projection_type


def main(config_path: str, dataset: xarray.Dataset):
    config, projection_type = zhongwei_config()

    # Derive variables and prepare a single in-memory dataset for plotting.
    variables, levels = required_variables_from_config(config)
    prepared = prepare_dataset(
        dataset,
        variables,
        levels,
        config,
        time_indices=None,
    )

    if "time" not in prepared.dims:
        prepared = prepared.expand_dims(time=[0])

    requested_global_times = selected_times_from_columns(config)
    max_requested_idx = (
        max(requested_global_times) if len(requested_global_times) > 0 else 0
    )
    if max_requested_idx >= prepared.sizes["time"]:
        raise ValueError(
            "Configured column time index exceeds available dataset time dimension. "
            f"Requested max={max_requested_idx}, available={prepared.sizes['time']}."
        )

    prepared = prepared.isel(time=requested_global_times)
    prepared = prepared.assign_coords(time=requested_global_times)

    # Print available time axis information
    num_times = prepared.sizes["time"]
    print(f"\n{'='*60}")
    print("Available time axis from loaded dataset:")
    print(f"  Total selected time steps: {num_times}")
    preview_times = [
        to_iso_time_string(t)
        for t in prepared.time.values[: min(5, len(prepared.time.values))]
    ]
    print(
        f"  Time values: {preview_times}"
        + (
            f" ... (showing first 5 of {len(prepared.time.values)})"
            if len(prepared.time.values) > 5
            else ""
        )
    )
    print(f"  Requested columns: {dict(config.columns)}")
    print(f"{'='*60}\n")

    rows = {"Loaded Dataset": prepared}
    row_titles = list(rows.keys())
    column_titles = ["" for _ in config.columns]

    selected_times = requested_global_times
    time_idx_to_local_pos = {
        time_idx: pos for pos, time_idx in enumerate(selected_times)
    }

    fcontour = plot_item_to_panel_spec(config.fcontour)
    contour = (
        plot_item_to_panel_spec(config.contour) if config.contour is not None else None
    )

    maps = []
    # Rows are epochs, columns are time steps
    for row_label, dat in rows.items():
        for col_label, time_idx in config.columns.items():
            if time_idx not in time_idx_to_local_pos:
                raise ValueError(
                    f"Configured time index {time_idx} is not available after selection."
                )
            local_t = time_idx_to_local_pos[time_idx]

            map_func = lambda ax, dat=dat, t=local_t: plot_map_panel(
                ax,
                dat.isel(time=t),
                fcontour=fcontour,
                contour=contour,
                arrows=None,
                region=None,
                title=None,
                land_color=config.land_color,
            )
            useful_hurricane_stats(dat.isel(time=local_t))
            maps.append(map_func)

    figsize = cast(tuple[int, int], tuple(config.figsize))

    # Create projection based on configuration
    projection = create_projection(
        projection_type=projection_type,
        region=config.region,
    )

    fig = create_multi_panel_figure(
        maps,
        nrows=len(row_titles),
        ncols=len(column_titles),
        figsize=figsize,
        subplot_kw={"projection": projection},
    )

    if config.output_path is not None:
        save_path = os.path.join(config.output_path, config.output_file)
    else:
        save_path = os.path.join(config.folder, config.output_file)
    plt.savefig(save_path, bbox_inches="tight")


def main_wrapper():
    parser = argparse.ArgumentParser(description="Plot hurricane maps from config.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--data-path",
        default="/home/users/f/froelicm/scratch/HRES_z_t_20200520_20200523_0012.grib",
        help="Path to GRIB dataset to load and plot.",
    )
    args = parser.parse_args()
    loaded_dataset = data_loading_and_preprocess(args.data_path)
    main(args.config, loaded_dataset)


if __name__ == "__main__":
    ds = data_loading_and_preprocess(
        "/home/users/f/froelicm/scratch/HRES_z_t_20200520_20200523_0012.grib"
    )
    main("dummy", ds)
