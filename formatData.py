import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path

from scipy.spatial import ConvexHull, QhullError
from matplotlib.patches import Polygon as MplPolygon

# Convert to JD time
def unix_to_jd(unix_time: float) -> float:
    return unix_time / 86400.0 + 2440587.5

def cleanData(df):
    # Drop rows with flag != 0
    print('Dropping rows with flag != 0')
    print('Len df Before: {0}'.format(len(df)))
    df = df[df['flags'] == 0].reset_index(drop=True)
    print('Len df After: {0}'.format(len(df)))

    # Drop nan rows
    print('Dropping nan-rows')
    print('Len df Before: {0}'.format(len(df)))
    df.dropna(axis='rows')
    print('Len df After: {0}'.format(len(df)))

    print('{0} rows in data'.format(len(df)))
    print('{0} columns in data: {1}'.format(len(df.columns), list(df.columns)))

    df['unix_time'] = df['t'] + 315964800
    df['JD'] = unix_to_jd(df['unix_time'])
    # detrad - Mean detector radius (distance from detector center) for events within the aperture.
    return(df)


def make_overlap_offsets(coord_bin_arcsec, step_arcsec=None, mode="full"):
    """
    For coord_bin_arcsec=180 and step_arcsec=90:

    mode="full" gives:
        (0,0), (0,90), (90,0), (90,90)

    mode="diagonal" gives:
        (0,0), (90,90)
    """
    if step_arcsec is None:
        step_arcsec = coord_bin_arcsec / 2

    offsets_1d = np.arange(0, coord_bin_arcsec, step_arcsec)

    if mode == "full":
        offsets = [(dra, ddec) for dra in offsets_1d for ddec in offsets_1d]
    elif mode == "diagonal":
        offsets = [(off, off) for off in offsets_1d]
    else:
        raise ValueError("mode must be 'full' or 'diagonal'")

    return offsets


def make_shifted_edges(vmin, vmax, bin_deg, offset_deg, origin=0.0):
    """
    Make edges with fixed bin size, shifted by offset_deg.

    Ensures the edges cover the full data range.
    """
    start = origin + offset_deg + np.floor((vmin - origin - offset_deg) / bin_deg) * bin_deg

    # make sure first edge is before vmin
    while start > vmin:
        start -= bin_deg

    edges = np.arange(start, vmax + bin_deg, bin_deg)

    return edges


def binData_overlap(
    dfNow,
    col_time="t",
    col_ra="ra",
    col_dec="dec",
    t_bin=10,
    coord_bin_arcsec=180,
    grid_step_arcsec=None,
    offsets_arcsec=None,
    offset_mode="full",
    ra_range=None,
    dec_range=None,
    time_range=None,
    prnt_Binning=False,
):
    coord_bin_deg = coord_bin_arcsec / 3600

    if grid_step_arcsec is None:
        grid_step_arcsec = coord_bin_arcsec / 2

    if offsets_arcsec is None:
        offsets_arcsec = make_overlap_offsets(
            coord_bin_arcsec=coord_bin_arcsec,
            step_arcsec=grid_step_arcsec,
            mode=offset_mode,
        )

    t = dfNow[col_time].to_numpy()
    ra = dfNow[col_ra].to_numpy()
    dec = dfNow[col_dec].to_numpy()

    if time_range is None:
        t_min, t_max = np.nanmin(t), np.nanmax(t)
    else:
        t_min, t_max = time_range

    if ra_range is None:
        ra_min, ra_max = np.nanmin(ra), np.nanmax(ra)
    else:
        ra_min, ra_max = ra_range

    if dec_range is None:
        dec_min, dec_max = np.nanmin(dec), np.nanmax(dec)
    else:
        dec_min, dec_max = dec_range

    t_edges = np.arange(t_min, t_max + t_bin, t_bin)

    samples = dfNow[[col_time, col_ra, col_dec]].to_numpy()

    results = {}

    for dra_arcsec, ddec_arcsec in offsets_arcsec:
        dra_deg = dra_arcsec / 3600
        ddec_deg = ddec_arcsec / 3600

        ra_edges = make_shifted_edges(
            ra_min,
            ra_max,
            coord_bin_deg,
            dra_deg,
            origin=0.0,
        )

        dec_edges = make_shifted_edges(
            dec_min,
            dec_max,
            coord_bin_deg,
            ddec_deg,
            origin=0.0,
        )

        cube, edges = np.histogramdd(
            samples,
            bins=(t_edges, ra_edges, dec_edges),
        )

        key = f"raoff_{dra_arcsec:g}_decoff_{ddec_arcsec:g}"

        results[key] = {
            "cube": cube,
            "edges": edges,
            "offset_arcsec": (dra_arcsec, ddec_arcsec),
            "coord_bin_arcsec": coord_bin_arcsec,
            "grid_step_arcsec": grid_step_arcsec,
        }

    if prnt_Binning:
        print("Number of grids:", len(results))
        for key, out in results.items():
            t_edges, ra_edges, dec_edges = out["edges"]
            print()
            print(key)
            print("offset_arcsec:", out["offset_arcsec"])
            print("first RA edges:", ra_edges[:5])
            print("first Dec edges:", dec_edges[:5])

    return results


def plot_overlap_grid_lines_debug(
    df,
    overlap_results,
    col_ra="ra",
    col_dec="dec",
    ra_center=None,
    dec_center=None,
    half_width_arcsec=600,
    max_points=10000,
    show_points=True,
):
    """
    Plot shifted grids with different colors and styles.
    This makes the overlap visually obvious.
    """

    if ra_center is None:
        ra_center = 0.5 * (df[col_ra].min() + df[col_ra].max())

    if dec_center is None:
        dec_center = 0.5 * (df[col_dec].min() + df[col_dec].max())

    half_width_deg = half_width_arcsec / 3600

    ra_min = ra_center - half_width_deg
    ra_max = ra_center + half_width_deg
    dec_min = dec_center - half_width_deg
    dec_max = dec_center + half_width_deg

    df_zoom = df[
        (df[col_ra] >= ra_min) &
        (df[col_ra] <= ra_max) &
        (df[col_dec] >= dec_min) &
        (df[col_dec] <= dec_max)
    ]

    if len(df_zoom) > max_points:
        df_zoom = df_zoom.sample(max_points, random_state=0)

    fig, ax = plt.subplots(figsize=(4, 4))

    if show_points:
        ax.scatter(
            df_zoom[col_ra],
            df_zoom[col_dec],
            s=2,
            alpha=0.15,
            color="black",
            label="photons",
        )

    colors = plt.cm.tab10(np.linspace(0, 1, len(overlap_results)))
    linestyles = ["-", "--", ":", "-."]

    for k, ((grid_key, out), color) in enumerate(zip(overlap_results.items(), colors)):
        t_edges, ra_edges, dec_edges = out["edges"]

        ra_edges_zoom = ra_edges[
            (ra_edges >= ra_min) &
            (ra_edges <= ra_max)
        ]

        dec_edges_zoom = dec_edges[
            (dec_edges >= dec_min) &
            (dec_edges <= dec_max)
        ]

        ls = linestyles[k % len(linestyles)]

        for j, ra_edge in enumerate(ra_edges_zoom):
            ax.axvline(
                ra_edge,
                color=color,
                ls=ls,
                lw=1.5,
                alpha=0.9,
                label=f"{grid_key}, offset={out['offset_arcsec']}" if j == 0 else None,
            )

        for dec_edge in dec_edges_zoom:
            ax.axhline(
                dec_edge,
                color=color,
                ls=ls,
                lw=1.5,
                alpha=0.9,
            )

    ax.set_xlim(ra_min, ra_max)
    ax.set_ylim(dec_min, dec_max)

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(
        f"Shifted overlapping grids\n"
        f"center=({ra_center:.6f}, {dec_center:.6f}), "
        f"half width={half_width_arcsec} arcsec"
    )

    ax.legend(fontsize=8, markerscale=5)
    ax.set_aspect("equal", adjustable="box")

    plt.show()

    return


def allData_to_df(allData_final):
    allData_final = np.array(allData_final)

    df_final = pd.DataFrame(
        allData_final,
        columns=['ra', 'dec', 'time', 'signal']
    )
    return(df_final)


def cut_df_by_unordered_convex_region(
    df,
    polygon_points,
    cols=("ra", "dec"),
    tol=1e-10,
    return_hull=False,
):
    pts = np.asarray(polygon_points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = np.unique(pts, axis=0)

    if len(pts) < 3:
        raise ValueError(f"Need at least 3 unique points for cut in columns {cols}.")

    mins = pts.min(axis=0)
    spans = pts.max(axis=0) - mins

    if np.any(spans == 0):
        raise ValueError(
            f"Degenerate cut for columns {cols}. "
            "The polygon has zero width in at least one coordinate."
        )

    # Normalize polygon and dataframe coordinates.
    # This is important for RA-time cuts where time is ~8e8.
    pts_scaled = (pts - mins) / spans
    xy = df[list(cols)].to_numpy(dtype=float)
    xy_scaled = (xy - mins) / spans

    try:
        hull = ConvexHull(pts_scaled)
    except QhullError:
        raise ValueError(f"Could not build convex hull for cut in columns {cols}.")

    # ConvexHull equations are of the form:
    # normal . x + offset <= 0 for points inside the hull.
    equations = hull.equations
    A = equations[:, :-1]
    b = equations[:, -1]

    finite_mask = np.isfinite(xy_scaled).all(axis=1)
    inside_mask = np.full(len(df), False)

    inside_mask[finite_mask] = np.all(
        xy_scaled[finite_mask] @ A.T + b <= tol,
        axis=1,
    )

    df_cut = df.loc[inside_mask].copy().reset_index(drop=True)

    ordered_hull_original = pts[hull.vertices]

    if return_hull:
        return df_cut, ordered_hull_original

    return df_cut


def cut_for_one_satellite(
    df_inp,
    signalSelected,
    radec_cut=None,
    ratime_cut=None,
    dectime_cut=None,
    timeMin_cut=None,
):
    df_inp_i = df_inp.copy()

    if radec_cut is not None:
        df_inp_i = cut_df_by_unordered_convex_region(
            df_inp_i,
            radec_cut,
            cols=("ra", "dec"),
        )

    if ratime_cut is not None:
        df_inp_i = cut_df_by_unordered_convex_region(
            df_inp_i,
            ratime_cut,
            cols=("ra", "time"),
        )

    if dectime_cut is not None:
        df_inp_i = cut_df_by_unordered_convex_region(
            df_inp_i,
            dectime_cut,
            cols=("dec", "time"),
        )

    if timeMin_cut is not None:
        df_inp_i = df_inp_i[df_inp_i["time"] >= timeMin_cut].reset_index(drop=True)

    df_inp_i = df_inp_i.sort_values("time").reset_index(drop=True)

    return df_inp_i