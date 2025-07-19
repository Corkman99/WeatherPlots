"""
SOURCE: Ben Harvey
Plots for storm ciaran paper supp mat.

To remake paper plots:
plot_all()

# To test sensitivity to storm motion vector:
calc_all_motion_vector()
plot_test_motion_vector()
"""

import datetime, os
import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import cartopy.crs as ccrs
import numpy as np

from windspharm.iris import VectorWind
from myiris import calculus, grid, constants  # this should be straight forward

ddir = "/home/users/pr902839/disks/sjclim/ciaran/data/"
fmt = "%HZ %d/%m/%Y"
savefmt = "%Y%m%d%H"

labels = {
    "era5": "ERA5",
    "an": "IFS Analysis",
    "ifs": "IFS HRES",
    "fcn": "FourCastNet",
    "fcn2": "FourCastNet v2",
    "pgw": "PanguWeather",
    "gc": "Graphcast",
}
cols = {
    "era5": "0.5",
    "an": "k",
    "ifs": "b",
    "fcn": "lightgreen",
    "fcn2": "g",
    "pgw": "yellow",
    "gc": "r",
}


def plot_all():
    # Assume same motion vector for all
    cx = 18.0
    cy = 6.0
    cxcytag = "{0:.1f}x{1:.1f}".format(cx, cy)
    for vt in [datetime.datetime(2023, 11, 1, 18), datetime.datetime(2023, 11, 2, 0)]:
        vals = {}
        for src in ["era5", "ifs", "fcn", "fcn2", "pgw", "gc"]:  # 'an'
            vals[src] = plot_gradient_winds(src=src, vt=vt, cx=cx, cy=cy)

        # Scatter wsgmax and wsgrmax against wsmax
        fig, axs = plt.subplots(2, 2, figsize=[8, 10], sharex="all", sharey="row")
        for src in ["era5", "ifs", "fcn", "fcn2", "pgw", "gc"]:  # 'an'
            # ws, wsg, wsgr, wss, wsgs, wsgrs
            # CCB Geos smooth
            axs[0, 0].plot(
                vals[src][3][0].data,
                vals[src][4][0].data,
                "o",
                c=cols[src],
                label=labels[src],
            )
            axs[0, 0].set_title("(a) CCB Geostrophic Wind", loc="left")
            # CCB Grad smooth
            axs[1, 0].plot(
                vals[src][3][0].data,
                vals[src][5][0].data,
                "o",
                c=cols[src],
                label=labels[src],
            )
            axs[1, 0].set_title("(c) CCB Gradient Wind", loc="left")
            axs[1, 0].axline([40, 40], slope=1, c="0.5", ls=":")
            # WCB Geos smooth
            axs[0, 1].plot(
                vals[src][3][1].data,
                vals[src][4][1].data,
                "o",
                c=cols[src],
                label=labels[src],
            )
            axs[0, 1].set_title("(b) WCB Geostrophic Wind", loc="left")
            # WCB Grad smooth
            axs[1, 1].plot(
                vals[src][3][1].data,
                vals[src][5][1].data,
                "o",
                c=cols[src],
                label=labels[src],
            )
            axs[1, 1].set_title("(d) WCB Gradient Wind", loc="left")
            axs[1, 1].axline([40, 40], slope=1, c="0.5", ls=":")
        axs[0, 0].set_xlim([40, 50])
        axs[0, 0].set_ylim([36, 80])
        axs[1, 0].set_ylim([30, 52])
        axs[1, 0].set_xlabel("Peak full wind [m/s]")
        axs[1, 1].set_xlabel("Peak full wind [m/s]")
        axs[0, 0].set_ylabel("Peak geostrophic wind [m/s]")
        axs[1, 0].set_ylabel("Peak gradient wind [m/s]")
        for ax in axs.flatten():
            ax.grid()
        axs[0, 0].legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(
            "gradient_winds_paper-scatter_{}_{}.png".format(
                vt.strftime(savefmt), cxcytag
            )
        )
        pdir = "/home/users/pr902839/public_html/scratch/ciaran"
        os.system("cp *paper*png " + pdir)


def plot_test_motion_vector():
    """Test range of motion vectors for era5 18Z only"""
    src = "era5"
    vt = datetime.datetime(2023, 11, 1, 18)
    # Range of plausible values from calc_all_motion_vectors
    cxcy_vals = [(0.0, 0.0), (17.7, 5.1), (21.0, 3.9), (15.9, 6.4)]
    vals = {}
    for cx, cy in cxcy_vals:
        print(cx, cy, (cx**2 + cy**2) ** 0.5)
        vals[(cx, cy)] = plot_gradient_winds(src=src, vt=vt, cx=cx, cy=cy)


def calc_all_motion_vector():
    for src in ["era5", "an", "ifs", "fcn", "fcn2", "pgw", "gc"]:
        calc_storm_motion_vector(src=src)


def load_data(src, bt):
    if src == "era5":
        # Tried Ambrogio's data but not global so VectorWind failed
        # Downlaoded my own global version
        # ddir2 = '/storage/research/sjclim/pq910803/Ciaran_ai/data'
        cl = iris.load(ddir + "/ERA5_plev_ciaran.nc")
        for cube in cl:
            if "pressure_level" in [co.name() for co in cube.coords()]:
                cube.coord("pressure_level").rename("pressure")
                cube.coord("pressure").convert_units("Pa")
    if src == "an":
        cl = iris.load(ddir + "/ben_analysis/extended_an-plev_ciaran.grib")
        for cube in cl:
            if "pressure" in [co.name() for co in cube.coords()]:
                cube.coord("pressure").convert_units("Pa")
    elif src == "ifs":
        cl = iris.load(
            [
                ddir
                + "/ben_ifs/extended_ifs-plev_ciaran_{}.grib".format(
                    bt.strftime("%Y%m%d%H")
                ),
                ddir
                + "/ben_ifs/extended_ifs-sfc-ciaran_{}.grib".format(
                    bt.strftime("%Y%m%d%H")
                ),
            ]
        )
        for cube in cl:
            if "pressure" in [co.name() for co in cube.coords()]:
                cube.coord("pressure").convert_units("Pa")
    elif src == "fcn":
        cl = iris.load(
            ddir + "/fourcastnet-ciaran-{}.grib".format(bt.strftime("%Y%m%d%H"))
        )
    elif src == "fcn2":
        cl = iris.load(
            ddir + "/fourcastnetv2-small-ciaran-{}.grib".format(bt.strftime("%Y%m%d%H"))
        )
    elif src == "pgw":
        cl = iris.load(
            ddir + "/panguweather-ciaran-{}.grib".format(bt.strftime("%Y%m%d%H"))
        )
    elif src == "gc":
        cl = iris.load(
            ddir + "/graphcast-ciaran-{}.grib".format(bt.strftime("%Y%m%d%H"))
        )
    return cl


def get_z850(cl, vt):
    vtcon = iris.Constraint(time=lambda t: t.point == vt)
    pcon = iris.Constraint(pressure=85000)
    z850 = cl.extract_cube("geopotential").extract(vtcon & pcon)
    z850 = z850 / iris.coords.AuxCoord(9.81, units="m s-2")
    z850.rename("geopotential_height")
    return z850


def get_uv850(cl, vt):
    vtcon = iris.Constraint(time=lambda t: t.point == vt)
    pcon = iris.Constraint(pressure=85000)
    try:
        uv850 = cl.extract_cube("x_wind" & pcon).extract(vtcon), cl.extract_cube(
            "y_wind" & pcon
        ).extract(vtcon)
    except:
        uv850 = cl.extract_cube("eastward_wind" & pcon).extract(vtcon), cl.extract_cube(
            "northward_wind" & pcon
        ).extract(vtcon)
    return uv850


def spheregrad(cube=None, opt=1):

    # Option 1: Use myiris.calculus
    if opt == 1:
        cubex0, cubey0 = calculus.sphere_grad(cube)  # Gradients on natural grids
        cubex = cubex0.regrid(cube, iris.analysis.Linear())  # Regrid to cube
        cubey = cubey0.regrid(cube, iris.analysis.Linear())
        cubex.convert_units(cube.units / "m")  # Fix units
        cubey.convert_units(cube.units / "m")

    # Option 2: Calculate manually (assume cube is (lats, lons) array))
    elif opt == 2:
        farr = cube.data
        lats = cube.coord("latitude").points
        lons = cube.coord("longitude").points
        a = 6.38e6
        fac = np.pi / 180
        # lat direction
        df_y = np.roll(farr, 1, axis=0) - np.roll(farr, -1, axis=0)
        dlats = np.roll(lats, 1) - np.roll(lats, -1)
        dy = a * fac * dlats[:, np.newaxis]
        dfdy = df_y / dy
        dfdy[0, :] = np.nan
        dfdy[-1, :] = np.nan
        cubey = cube.copy()
        cubey.data = dfdy
        cubey.units = cube.units / "m"
        cubey.rename(cube.name() + "_diff_in_y")
        # lon direction
        df_x = np.roll(farr, 1, axis=1) - np.roll(farr, -1, axis=1)
        dlons = np.roll(lons, 1) - np.roll(lons, -1)
        dx = a * fac * np.cos(fac * lats)[:, np.newaxis] * dlons[np.newaxis, :]
        dfdx = df_x / dx
        dfdx[:, 0] = np.nan
        dfdx[:, -1] = np.nan
        cubex = cube.copy()
        cubex.data = dfdx
        cubex.units = cube.units / "m"
        cubex.rename(cube.name() + "_diff_in_x")

    return cubex, cubey


def test_spheregrad(src="an"):
    """Testing if gradients are coming out with the right units.

    Seem to work ok (DO need convert_units to remove radians scaling)
    """
    import iris.quickplot as qplt

    # Load data and subset
    cl = load_data(src)
    box = {"longitude": [-15, 0], "latitude": [42, 55]}
    z = get_z850(cl).intersection(**box)

    zx1, zy1 = spheregrad(z, opt=1)
    zx2, zy2 = spheregrad(z, opt=2)

    fig, axs = plt.subplots(
        2, 3, figsize=[18, 6], subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax = axs[0, 0]
    cf = qplt.contourf(z, axes=ax)
    ax = axs[0, 1]
    cf = qplt.contourf(zx1, axes=ax)
    ax = axs[0, 2]
    cf = qplt.contourf(zy1, axes=ax)
    ax = axs[1, 1]
    cf = qplt.contourf(zx2, axes=ax)
    ax = axs[1, 2]
    cf = qplt.contourf(zy2, axes=ax)
    for ax in axs.flatten():
        ax.coastlines()
        ax.gridlines(xlocs=np.arange(-20, 20, 5), ylocs=np.arange(40, 70, 5))
    fig.tight_layout()


# DONE 
import xarray
from typing import List, Tuple, Optional

EARTH_RADIUS = 6.38e6
RADIANS = np.pi / 180


def calc_storm_motion_vector(
    field: xarray.DataArray, search_region: Tuple[float, float, float, float]
):
    """
    Calculate the cyclone motion vectors

    Arguments:
        - field: dataarray of geopotential with two timesteps in dimension 'time'
        - search_region: (minlat, minlon, maxlat, maxlon) of where to confine cyclone eye search
    """

    assert field.sizes["time"] == 2
    times = field["time"].values
    dt = (times[1] - times[0]) / np.timedelta64(1, "s")
    assert "level" in field.dims

    minlat, minlon, maxlat, maxlon = search_region

    # field_box = field.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))

    # compute for every pressure level (can be vectorized)
    vectors = {}
    for pl in field["level"]:
        zmin = _extract_hurricane_centers(
            field.sel(level=pl), minlat, minlon, maxlat, maxlon
        )
        dlon = zmin[1][0] - zmin[0][0]
        dlat = zmin[1][1] - zmin[0][1]
        lat0 = (zmin[1][1] + zmin[0][1]) / 2
        dx = EARTH_RADIUS * RADIANS * dlon * np.cos(RADIANS * lat0)
        dy = EARTH_RADIUS * RADIANS * dlat
        vectors[str(pl)] = (dx / dt, dy / dt)
    
    return vectors

def _extract_hurricane_centers(mslp, minlat, minlon, maxlat, maxlon):
    subregion = mslp.sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))
    min_coords = []
    for t in subregion.time:
        slice_t = subregion.sel(time=t, drop=True).drop_vars(
            [v for v in subregion.coords if v not in ("lat", "lon")],
            errors="ignore",
        )
        # Find the indices of the minimum value
        min_idx = np.unravel_index(np.argmin(slice_t.values, axis=None), slice_t.shape)
        # Get the corresponding lat/lon values
        min_lat = float(slice_t.lat.values[min_idx[0]])
        min_lon = float(slice_t.lon.values[min_idx[1]])
        min_coords.append((min_lon, min_lat))
    return min_coords


def calc_gradient_winds_simple(z):
    """Compute gradient wind speed using Helen's method."""

    # Compute dPhi/dn term
    zx, zy = spheregrad(z)
    _, _, f = grid.coriolis(z)
    ug = -constants.g / f * zy
    vg = constants.g / f * zx
    wsg = (ug**2 + vg**2) ** 0.5
    dPhidn = -constants.g * (zx**2 + zy**2) ** 0.5

    # Compute distance to low centre
    def _greatcircle_two_pts(pt0, pt1):
        """Return great circle distance between a pair of (lon, lat) points"""
        import cartopy.geodesic as cgeodesic

        geodesic = cgeodesic.Geodesic()
        dist = geodesic.inverse(pt0, pt1)[0][0]  # m
        return dist

    def _greatcircle(cube, pt0):
        """Return cube holding distances from each grid point in cube to pt0s"""
        lats = cube.coord("latitude").points
        lons = cube.coord("longitude").points
        outarr = np.zeros_like(cube.data)
        for loni, lon in enumerate(lons):
            for lati, lat in enumerate(lats):
                outarr[lati, loni] = _greatcircle_two_pts(pt0, [lon, lat])
        r = cube.copy()
        r.data = outarr
        r.rename("Distance")
        r.units = "m"
        return r

    minz = z[np.where(z.data == z.data.min())]
    pt0 = [minz.coord("longitude").points[0], minz.coord("latitude").points[0]]
    R = _greatcircle(z, pt0)

    # Compute gradient wind
    hfR = f * R / 2
    wsgr_si = -hfR + (hfR**2 - R * dPhidn) ** 0.5

    return wsgr_si, dPhidn, R, f


def calc_gradient_winds_steady_contour(z, zs, cx=18, cy=6):
    """Compute gradient wind speed using steady contour approach.

    From Brill (2014): See Table 3.

    Experimenting with smoothing data to get better results. z is used for
    geostrophic wind, zs is used for curvature.

    Experimenting with assumption of constant translation as per Holton book.
    (cx, cy) is system motion vector in m/s
    """

    # Compute geostrophic wind
    zx, zy = spheregrad(z)
    _, _, f = grid.coriolis(z)
    ug = -constants.g / f * zy
    vg = constants.g / f * zx
    wsg = (ug**2 + vg**2) ** 0.5

    # Compute geostrophic wind
    zx1, zy1 = spheregrad(zs)
    _, _, f1 = grid.coriolis(zs)
    ug1 = -constants.g / f1 * zy1
    vg1 = constants.g / f1 * zx1
    wsg1 = (ug1**2 + vg1**2) ** 0.5

    # Compute geostrophic vorticity
    zxx1, _ = spheregrad(zx1)
    _, zyy1 = spheregrad(zy1)
    zeta1 = constants.g / f1 * (zxx1 + zyy1)

    # Compute dV/deta term
    wsgx1, wsgy1 = spheregrad(wsg1)
    ve_term1 = (-vg1 * wsgx1 + ug1 * wsgy1) / wsg1

    # Compute curvature and regrid back to z grid
    k1 = (zeta1 + ve_term1) / wsg1
    k = k1.regrid(z, iris.analysis.Linear())

    # Try correction due to storm motion
    # (estimated from calc_storm_motion_vector)
    Cx = iris.cube.Cube(cx, units="m s-1")
    Cy = iris.cube.Cube(cy, units="m s-1")
    C_cos_gamma = (Cx * ug + Cy * vg) / wsg
    fp = f - k * C_cos_gamma

    # Compute gradient wind
    wsgr_cs = (-fp + (fp**2 + 4 * k * f * wsg) ** 0.5) * 0.5 * k ** (-1)

    return (
        wsg,
        wsgr_cs,
        ug,
        vg,
        f,
        k,  # Full grid terms
        ve_term1,
        zeta1,
        wsg1,
        f1,
        k1,
    )  # Reduced grid terms


def calc_gradient_winds_nonsteady_natural(z, u, v):
    """Compute gradient wind speed using natural nonsteady approach.

    From Brill (2014): See Table 3.
    """

    # Compute geostrophic wind
    zx, zy = spheregrad(z)
    _, _, f = grid.coriolis(z)
    ug = -constants.g / f * zy
    vg = constants.g / f * zx

    # Compute windspeeds
    ws = (u**2 + v**2) ** 0.5
    wsg = (ug**2 + vg**2) ** 0.5

    # Compute curvature. Is noisy so use reduced grid?
    skip = 1
    z1 = z[::skip, ::skip]
    u1 = u[::skip, ::skip]
    v1 = v[::skip, ::skip]
    ws1 = ws[::skip, ::skip]
    _, _, f1 = grid.coriolis(z1)
    zx1, zy1 = spheregrad(z1)

    # Compute dphi/deta term
    pe_term1 = constants.g * (-v1 * zx1 + u1 * zy1) / ws1

    # Compute curvature
    k1 = -(f1 * ws1 + pe_term1) / ws1**2
    k = k1.regrid(z, iris.analysis.Linear())

    # Compute gradient wind
    wsgr_nn = (-f + (f**2 + 4 * f * k * wsg) ** 0.5) * 0.5 * k ** (-1)

    return (
        ws,
        wsg,
        wsgr_nn,
        ug,
        vg,
        f,
        k,  # Full grid terms
        pe_term1,
        f1,
        ws1,
        k1,
    )  # Reduced grid terms


def plot_gradient_winds(
    src="an",
    vt=datetime.datetime(2023, 11, 1, 18),
    bt=datetime.datetime(2023, 11, 1, 0),
    paper_only=True,
    cx=18.0,
    cy=6.0,
):
    """Compute all versions of gradient wind and produce plots.

    (cx, cy) are passed to contour_steady routine.
    """

    datetag = "Valid: {}".format(vt.strftime(fmt))
    cxcytag = "{0:.1f}x{1:.1f}".format(cx, cy)
    if src != "an":
        datetag += "\nBase: {}".format(bt.strftime(fmt))
    else:
        datetag += "\n "

    wind_kws1 = {"levels": np.linspace(25, 55, 13), "cmap": "plasma_r", "extend": "max"}
    wind_kws2 = {
        "levels": np.linspace(45, 105, 13),
        "cmap": "viridis_r",
        "extend": "max",
    }
    wind_kws3 = {"levels": np.linspace(-12, 12, 9), "cmap": "bwr", "extend": "both"}
    wind_kws4 = {"levels": np.linspace(-1.2, 1.2, 9), "cmap": "bwr", "extend": "both"}
    ws_kws = {"levels": [35, 40, 45], "colors": "0.5"}

    # Load data and subset
    cl = load_data(src, bt)
    box = {"longitude": [-15, 0], "latitude": [42, 55]}
    z = get_z850(cl, vt)
    u, v = get_uv850(cl, vt)
    if src == "fcn":
        # Some missing data near poles
        z.data[np.where(z.data.mask)] = 1e3
        u.data[np.where(u.data.mask)] = 0
        v.data[np.where(v.data.mask)] = 0
    zvw = VectorWind(z, z)
    zs = zvw.truncate(z, truncation=106)
    us = zvw.truncate(u, truncation=106)
    vs = zvw.truncate(v, truncation=106)
    z = z.intersection(**box)
    u = u.intersection(**box)
    v = v.intersection(**box)
    ws = (u**2 + v**2) ** 0.5
    zs = zs.intersection(**box)
    us = us.intersection(**box)
    vs = vs.intersection(**box)
    wss = (us**2 + vs**2) ** 0.5

    # Calc gradient wind SI on original z
    wsgr_si, dPhidn, R, fsi = calc_gradient_winds_simple(z)
    # Calc gradient wind SI on smooth z
    wsgr_sis, dPhidns, Rs, fsis = calc_gradient_winds_simple(zs)
    # Calc gradient wind CS on original z
    wsg, wsgr_cs, ug, vg, f, k, ve_term1, zeta1, wsg1, f1, k1 = (
        calc_gradient_winds_steady_contour(z, zs, cx=cx, cy=cy)
    )
    # Calc gradient wind CS on smooth z
    wsgs, wsgr_css, ugs, vgs, fs, ks, ve_term1s, zeta1s, wsg1s, f1s, k1s = (
        calc_gradient_winds_steady_contour(zs, zs, cx=cx, cy=cy)
    )
    # Calc gradient wind NN on original z, u and v
    wsnn, wsgnn, wsgr_nn, ugnn, vgnn, fnn, knn, pe_term1nn, f1nn, ws1nn, k1nn = (
        calc_gradient_winds_nonsteady_natural(z, u, v)
    )
    # Calc gradient wind NN on smooth z, u and v
    (
        wsnns,
        wsgnns,
        wsgr_nns,
        ugnns,
        vgnns,
        fnns,
        knns,
        pe_term1nns,
        f1nns,
        ws1nns,
        k1nns,
    ) = calc_gradient_winds_nonsteady_natural(zs, us, vs)

    # Plot multi-panel for single model - ALL WIND COMBINATIONS
    wsgr_si_mask = wsgr_si.copy(data=np.isnan(wsgr_si.data))
    wsgr_sis_mask = wsgr_sis.copy(data=np.isnan(wsgr_sis.data))
    wsgr_cs_mask = wsgr_cs.copy(data=np.isnan(wsgr_cs.data))
    wsgr_css_mask = wsgr_css.copy(data=np.isnan(wsgr_css.data))
    wsgr_nn_mask = wsgr_nn.copy(data=np.isnan(wsgr_nn.data))
    wsgr_nns_mask = wsgr_nns.copy(data=np.isnan(wsgr_nns.data))
    fig, axs = plt.subplots(
        4, 6, figsize=[22, 9], subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax = axs[0, 0]
    cf = iplt.contourf(z, levels=np.linspace(1e3, 1.5e3, 21), cmap="viridis", axes=ax)
    ax.set_title("Height", loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[0, 1]
    cf = iplt.contourf(ws, axes=ax, **wind_kws1)
    ax.set_title("Full Speed", loc="left")
    plt.colorbar(cf, ax=ax, ticks=cf.levels[::2])
    ax = axs[0, 2]
    cf = iplt.contourf(wsg, axes=ax, **wind_kws2)
    ax.set_title("Geostrophic Speed", loc="left")
    plt.colorbar(cf, ax=ax, ticks=cf.levels[::2])
    ax = axs[0, 4]
    cf = iplt.contourf(wsgr_cs, axes=ax, **wind_kws1)
    iplt.contourf(wsgr_cs_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("Gradient Speed (CS)", loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[0, 5]
    cf = iplt.contourf(wsgr_nn, axes=ax, **wind_kws1)
    iplt.contourf(wsgr_nn_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("Gradient Speed (NN)", loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[0, 3]
    cf = iplt.contourf(wsgr_si, axes=ax, **wind_kws1)
    iplt.contourf(wsgr_si_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("Gradient Speed (SI)", loc="left")
    plt.colorbar(cf, ax=ax)

    ax = axs[1, 0]
    ax.set_visible(False)
    ax = axs[1, 1]
    ax.set_visible(False)
    ax = axs[1, 2]
    ax.set_visible(False)
    ax = axs[1, 4]
    cf = iplt.contourf(ws - wsgr_cs, axes=ax, **wind_kws3)
    iplt.contourf(wsgr_cs_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("Full - Gradient (CS)", loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[1, 5]
    cf = iplt.contourf(ws - wsgr_nn, axes=ax, **wind_kws4)
    iplt.contourf(wsgr_nn_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("Full - Gradient (NN)", loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[1, 3]
    cf = iplt.contourf(ws - wsgr_si, axes=ax, **wind_kws3)
    iplt.contourf(wsgr_si_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("Full - Gradient (SI)", loc="left")
    plt.colorbar(cf, ax=ax)

    ax = axs[2, 0]
    cf = iplt.contourf(zs, levels=np.linspace(1e3, 1.5e3, 21), cmap="viridis", axes=ax)
    ax.set_title("Height T106", loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[2, 1]
    cf = iplt.contourf(wss, axes=ax, **wind_kws1)
    ax.set_title("Full Speed T106", loc="left")
    plt.colorbar(cf, ax=ax, ticks=cf.levels[::2])
    ax = axs[2, 2]
    cf = iplt.contourf(wsgs, axes=ax, **wind_kws2)
    ax.set_title("Geostrophic Speed T106", loc="left")
    plt.colorbar(cf, ax=ax, ticks=cf.levels[::2])
    ax = axs[2, 4]
    cf = iplt.contourf(wsgr_css, axes=ax, **wind_kws1)
    iplt.contourf(wsgr_css_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("Gradient Speed (CS) T106", loc="left")
    plt.colorbar(cf, ax=ax, ticks=cf.levels[::2])
    ax = axs[2, 5]
    cf = iplt.contourf(wsgr_nns, axes=ax, **wind_kws1)
    iplt.contourf(wsgr_nns_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("Gradient Speed (NN) T106", loc="left")
    plt.colorbar(cf, ax=ax, ticks=cf.levels[::2])
    ax = axs[2, 3]
    cf = iplt.contourf(wsgr_sis, axes=ax, **wind_kws1)
    iplt.contourf(wsgr_sis_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("Gradient Speed (SI) T106", loc="left")
    plt.colorbar(cf, ax=ax, ticks=cf.levels[::2])

    ax = axs[3, 0]
    ax.set_visible(False)
    ax = axs[3, 1]
    cf = iplt.contourf(ws - wss, axes=ax, **wind_kws3)
    ax.set_title("Full - Full T106", loc="left")
    plt.colorbar(cf, ax=ax, ticks=cf.levels[::2])
    ax = axs[3, 2]
    ax.set_visible(False)
    ax = axs[3, 4]
    cf = iplt.contourf(wss - wsgr_css, axes=ax, **wind_kws3)
    iplt.contourf(wsgr_css_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("Full T106 - Gradient (CS) T106", loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[3, 5]
    cf = iplt.contourf(wss - wsgr_nns, axes=ax, **wind_kws4)
    iplt.contourf(wsgr_nns_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("Full T106 - Gradient (NN) T106", loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[3, 3]
    cf = iplt.contourf(wss - wsgr_sis, axes=ax, **wind_kws3)
    iplt.contourf(wsgr_sis_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("Full T106 - Gradient (SI) T106", loc="left")
    plt.colorbar(cf, ax=ax)

    for ax in axs.flatten():
        ax.coastlines()
        ax.gridlines(xlocs=np.arange(-20, 20, 5), ylocs=np.arange(40, 70, 5))
    fig.suptitle(datetag, x=0.98, y=0.99, va="top", ha="right")
    fig.tight_layout()
    fig.text(0.02, 0.99, labels[src], va="top", ha="left", size="large")
    fig.savefig(
        "gradient_winds_all-maps_{}_{}_{}.png".format(
            src, vt.strftime(savefmt), cxcytag
        )
    )
    fig.show()

    # Plot multi-panel for single model - SUBSET FOR PAPER
    # Select version to use
    # # natural non-steady
    # wsgr = wsgr_nn
    # wsgrs = wsgr_nns
    # wsgr_mask = wsgr_nn_mask
    # wsgrs_mask = wsgr_nns_mask
    # tag = ' (NN)'
    # fntag = '_nn'
    # wind_kws = wind_kws4
    # contour steady
    wsgr = wsgr_cs
    wsgrs = wsgr_css
    wsgr_mask = wsgr_cs_mask
    wsgrs_mask = wsgr_css_mask
    tag = ""
    fntag = "_cs"
    wind_kws = wind_kws3

    fig, axs = plt.subplots(
        2, 4, figsize=[14, 5], subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax = axs[0, 0]
    cf = iplt.contourf(ws, axes=ax, **wind_kws1)
    cc = iplt.contour(z, levels=np.linspace(1e3, 1.5e3, 11), colors="0.5", axes=ax)
    plt.clabel(cc)
    ax.set_title("(a) Full Wind", loc="left")
    plt.colorbar(cf, ax=ax, ticks=cf.levels[::2])
    ax = axs[0, 1]
    cf = iplt.contourf(wsg, axes=ax, **wind_kws2)
    iplt.contour(ws, axes=ax, **ws_kws)
    ax.set_title("(b) Geostrophic Wind", loc="left")
    plt.colorbar(cf, ax=ax, ticks=cf.levels[::2])
    ax = axs[0, 2]
    cf = iplt.contourf(wsgr, axes=ax, **wind_kws1)
    iplt.contour(ws, axes=ax, **ws_kws)
    iplt.contourf(wsgr_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("(c) Gradient Wind" + tag, loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[0, 3]
    cf = iplt.contourf(ws - wsgr, axes=ax, **wind_kws)
    iplt.contour(ws, axes=ax, **ws_kws)
    iplt.contourf(wsgr_cs_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("(d) Full - Gradient" + tag, loc="left")
    plt.colorbar(cf, ax=ax)

    ax = axs[1, 0]
    cf = iplt.contourf(wss, axes=ax, **wind_kws1)
    cc = iplt.contour(zs, levels=np.linspace(1e3, 1.5e3, 11), colors="0.5", axes=ax)
    plt.clabel(cc)
    ax.set_title("(e) Full Wind T106", loc="left")
    plt.colorbar(cf, ax=ax, ticks=cf.levels[::2])
    ax = axs[1, 1]
    cf = iplt.contourf(wsgs, axes=ax, **wind_kws2)
    iplt.contour(wss, axes=ax, **ws_kws)
    ax.set_title("(f) Geostrophic Wind T106", loc="left")
    plt.colorbar(cf, ax=ax, ticks=cf.levels[::2])
    ax = axs[1, 2]
    cf = iplt.contourf(wsgrs, axes=ax, **wind_kws1)
    iplt.contour(wss, axes=ax, **ws_kws)
    iplt.contourf(wsgrs_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("(g) Gradient Wind" + tag + " T106", loc="left")
    plt.colorbar(cf, ax=ax, ticks=cf.levels[::2])
    ax = axs[1, 3]
    cf = iplt.contourf(wss - wsgrs, axes=ax, **wind_kws)
    iplt.contour(wss, axes=ax, **ws_kws)
    iplt.contourf(wsgrs_mask, levels=[1e-5, 10], colors=["0.75"], axes=ax)
    ax.set_title("(h) Full T106 - Gradient" + tag + " T106", loc="left")
    plt.colorbar(cf, ax=ax)

    for ax in axs.flatten():
        ax.coastlines()
        ax.gridlines(xlocs=np.arange(-20, 20, 5), ylocs=np.arange(40, 70, 5))
    # fig.suptitle(datetag, x=0.98, y=0.99, va='top', ha='right')
    fig.tight_layout()
    # fig.text(0.02, 0.99, labels[src], va='top', ha='left', size='large')

    ### CCB and WCB at 18Z 1 Nov
    ccbbox = {"longitude": [-13, -8], "latitude": [46, 50]}
    wcbbox = {"longitude": [-8, -4], "latitude": [44, 46]}

    def _draw_box(ax, lons, lats, t):
        ax.plot(
            [lons[0], lons[1], lons[1], lons[0], lons[0]],
            [lats[0], lats[0], lats[1], lats[1], lats[0]],
            color="k",
            linestyle=":",
        )
        ax.text(
            lons[0] + 0.02 * (lons[1] - lons[0]),
            lats[0] + 0.02 * (lats[1] - lats[0]),
            t,
        )

    for ax in axs[1, 0:3].flatten():
        _draw_box(ax, ccbbox["longitude"], ccbbox["latitude"], "C")
        _draw_box(ax, wcbbox["longitude"], wcbbox["latitude"], "W")

    fig.savefig(
        "gradient_winds_paper-maps_{}_{}_{}_{}.png".format(
            src, fntag, vt.strftime(savefmt), cxcytag
        )
    )
    fig.show()

    if paper_only:
        # Compute max wind speeds
        def get_max(cube):
            c = cube.intersection(**ccbbox)
            c.data[np.where(np.isnan(c.data))] = 0.0
            ccbmax = c.collapsed(["longitude", "latitude"], iris.analysis.MAX)
            c = cube.intersection(**wcbbox)
            c.data[np.where(np.isnan(c.data))] = 0.0
            wcbmax = c.collapsed(["longitude", "latitude"], iris.analysis.MAX)
            return [ccbmax, wcbmax]

        out = []
        for cube in ws, wsg, wsgr, wss, wsgs, wsgrs:
            out.append(get_max(cube))
        return out

    print("Stopping before component plots - need to think about these...")
    return

    # Plot multi-panel for single model - Gradient wind CS terms for smooth case
    fig, axs = plt.subplots(
        3, 4, figsize=[15, 10], subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax = axs[0, 0]
    cf = iplt.contourf(wsg1s, levels=np.linspace(35, 95, 9), cmap="bwr", axes=ax)
    ax.set_title("WSG1 [{}]".format(wsg1.units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[0, 1]
    cf = iplt.contourf(zeta1s, levels=np.linspace(-1e-3, 1e-3, 21), cmap="bwr", axes=ax)
    ax.set_title("Zeta1 [{}]".format(zeta1.units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[0, 2]
    cf = iplt.contourf(
        ve_term1s, levels=np.linspace(-1e-3, 1e-3, 21), cmap="bwr", axes=ax
    )
    ax.set_title("VE_term1 [{}]".format(ve_term1.units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[0, 3]
    cf = iplt.contourf(k1s, levels=np.linspace(-1e-5, 1e-5, 21), cmap="bwr", axes=ax)
    ax.set_title("k1 [{}]".format(k1.units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[1, 0]
    cf = iplt.contourf(fs**2, levels=np.linspace(-5e-7, 5e-7, 21), cmap="bwr", axes=ax)
    ax.set_title("f^2 [{}]".format((f**2).units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[1, 1]
    cf = iplt.contourf(
        4 * fs * ks * wsgs, levels=np.linspace(-5e-7, 5e-7, 21), cmap="bwr", axes=ax
    )
    ax.set_title("4fKVg [{}]".format((f * k * wsg).units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[1, 2]
    cf = iplt.contourf(-fs, levels=np.linspace(-1e-3, 1e-3, 21), cmap="bwr", axes=ax)
    ax.set_title("-f [{}]".format((f).units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[1, 3]
    cf = iplt.contourf(
        (fs**2 + 4 * fs * ks * wsgs) ** 0.5,
        levels=np.linspace(-1e-3, 1e-3, 21),
        cmap="bwr",
        axes=ax,
    )
    ax.set_title("(f^2+4fKVg)^(1/2) [{}]".format(f.units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[2, 0]
    cf = iplt.contourf(
        -fs + (fs**2 + 4 * fs * ks * wsgs) ** 0.5,
        levels=np.linspace(-1e-3, 1e-3, 21),
        cmap="bwr",
        axes=ax,
    )
    ax.set_title("-f+(f^2+4fKVg)^(1/2) [{}]".format(f.units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[2, 1]
    cf = iplt.contourf(
        (-fs + (fs**2 + 4 * fs * ks * wsgs) ** 0.5) * 0.5 * ks ** (-1),
        levels=np.linspace(-50, 50, 21),
        cmap="bwr",
        axes=ax,
    )
    ax.set_title("-f+(f^2+4fKVg)^(1/2)/2K [{}]".format((f / k).units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[2, 2]
    cf = iplt.contourf(wsgr_css, levels=np.linspace(-50, 50, 21), cmap="bwr", axes=ax)
    ax.set_title("WSGR_cs [{}]".format((f / k).units), loc="left")
    plt.colorbar(cf, ax=ax)

    for ax in axs.flatten():
        ax.coastlines()
    fig.suptitle(datetag, x=0.98, y=0.99, va="top", ha="right")
    fig.tight_layout()
    fig.savefig(
        "gradient_winds_cs-maps-terms_{}_{}.png".format(src, vt.strftime(savefmt))
    )
    fig.show()

    # Plot multi-panel for single model - Gradient wind NN terms for smooth case
    fig, axs = plt.subplots(
        3, 4, figsize=[15, 10], subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax = axs[0, 0]
    cf = iplt.contourf(
        f1nns * ws1nns, levels=np.linspace(-1e-2, 1e-2, 21), cmap="bwr", axes=ax
    )
    ax.set_title("fV [{}]".format((f1nns * ws1nns).units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[0, 1]
    cf = iplt.contourf(
        pe_term1nns, levels=np.linspace(-1e-2, 1e-2, 21), cmap="bwr", axes=ax
    )
    ax.set_title("dphi/deta [{}]".format(pe_term1nns.units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[0, 2]
    cf = iplt.contourf(k1nns, levels=np.linspace(-1e-4, 1e-4, 21), cmap="bwr", axes=ax)
    ax.set_title("Curvature [{}]".format(k1nns.units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[0, 3]
    cf = iplt.contourf(
        k1nns ** (-1), levels=np.linspace(-5e6, 5e6, 21), cmap="bwr", axes=ax
    )
    ax.set_title("1 / Curvature [{}]".format((k1nns ** (-1)).units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[1, 0]
    cf = iplt.contourf(
        fnns**2, levels=np.linspace(-1e-6, 1e-6, 21), cmap="bwr", axes=ax
    )
    ax.set_title("f^2 [{}]".format((fnns**2).units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[1, 1]
    cf = iplt.contourf(
        4 * fnns * knns * wsgnns,
        levels=np.linspace(-1e-6, 1e-6, 21),
        cmap="bwr",
        axes=ax,
    )
    ax.set_title("4fKVg [{}]".format((fnns * knns * wsgnns).units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[1, 2]
    cf = iplt.contourf(-fnns, levels=np.linspace(-1e-3, 1e-3, 21), cmap="bwr", axes=ax)
    ax.set_title("-f [{}]".format((fnns).units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[1, 3]
    cf = iplt.contourf(
        (fnns**2 + 4 * fnns * knns * wsgnns) ** 0.5,
        levels=np.linspace(-1e-3, 1e-3, 21),
        cmap="bwr",
        axes=ax,
    )
    ax.set_title("(f^2+4fKVg)^(1/2) [{}]".format(fnns.units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[2, 0]
    cf = iplt.contourf(
        -fnns + (fnns**2 + 4 * fnns * knns * wsgnns) ** 0.5,
        levels=np.linspace(-1e-3, 1e-3, 21),
        cmap="bwr",
        axes=ax,
    )
    ax.set_title("-f+(f^2+4fKVg)^(1/2) [{}]".format(fnns.units), loc="left")
    plt.colorbar(cf, ax=ax)
    ax = axs[2, 1]
    cf = iplt.contourf(
        (-fnns + (fnns**2 + 4 * fnns * knns * wsgnns) ** 0.5) * 0.5 * knns ** (-1),
        levels=np.linspace(-50, 50, 21),
        cmap="bwr",
        axes=ax,
    )
    ax.set_title(
        "(-f+(f^2+4fKVg)^(1/2))/2K [{}]".format((fnns / knns).units), loc="left"
    )
    plt.colorbar(cf, ax=ax)

    for ax in axs.flatten():
        ax.coastlines()
    fig.suptitle(datetag, x=0.98, y=0.99, va="top", ha="right")
    fig.tight_layout()
    fig.savefig(
        "gradient_winds_nn-maps-terms_{}_{}.png".format(src, vt.strftime(savefmt))
    )
    fig.show()
