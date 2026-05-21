import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

import pandas as pd

from scipy.spatial import ConvexHull, QhullError
from matplotlib.patches import Polygon as MplPolygon

def plotCandidates_A(resultsCandidates, overlap_results):
    xvals = [resCand[0] for resCand in resultsCandidates[list(resultsCandidates.keys())[0]]]
    yvals = [resCand[1] for resCand in resultsCandidates[list(resultsCandidates.keys())[0]]]
    tvals = [np.mean(resCand[2]) for resCand in resultsCandidates[list(resultsCandidates.keys())[0]]]
    all_t = np.concatenate([np.asarray(tvals)])
    # Get global time range for this plot, so all candidates share the same color scale
    norm = mpl.colors.Normalize(vmin=np.nanmin(all_t), vmax=np.nanmax(all_t))

    allData_final = []

    fig, ax = plt.subplots()
    for ky, val in resultsCandidates.items():
        out = overlap_results[ky]
        t_edges, ra_edges, dec_edges = out["edges"]
        t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
        ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
        dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])

        for resCand in val:
            x_i, y_i, t_i, sig_i = resCand
            ra_i = ra_centers[x_i]
            dec_i = dec_centers[y_i]
            
            sc = ax.scatter(
                ra_i,
                dec_i,
                c=np.median(t_i),              # color by time
                cmap="viridis",
                norm=norm,
                s=60
            )
            allData_final.append([ra_i, dec_i, np.mean(t_i), np.mean(sig_i)])

    cbar = fig.colorbar(sc, ax=ax)

    ax.set_xlabel("RA[id]")
    ax.set_ylabel("Dec[id]")
    ax.set_title(ky)

    fig.tight_layout()
    plt.show()
    return(allData_final)

# def plt_RegionCut(df_final, df_final_cand0, signalSelected, radec_cut, ratime_cut, dectime_cut, timeMin_cut, showFulLCands=True):
#     fig, axs = plt.subplots(1, 3, figsize=(14, 5))
#     ax = axs[0]

#     if radec_cut is not None:
#         for xin in radec_cut:
#             ax.axvline(xin[0], color='purple', ls='dashed')
#             ax.axhline(xin[1], color='purple', ls='dashed')
#             ax.scatter(xin[0], xin[1], color='purple', marker='x')
#     if showFulLCands:
#         ax.scatter(df_final['ra'], df_final['dec'], color='black')
#     ax.scatter(df_final_cand0['ra'], df_final_cand0['dec'], alpha=0.5, color='orange')
#     ax.set_xlabel('RA [deg]')
#     ax.set_ylabel('Dec [deg]')
#     ax = axs[1]
#     if ratime_cut is not None:
#         for xin in ratime_cut:
#             ax.axvline(xin[1], color='purple', ls='dashed')
#             ax.axhline(xin[0], color='purple', ls='dashed')
#             ax.scatter(xin[1], xin[0], color='purple', marker='x')
#     if showFulLCands:
#         ax.scatter(df_final['time'], df_final['ra'], color='black')
#     ax.scatter(df_final_cand0['time'], df_final_cand0['ra'], alpha=0.5, color='orange')
#     ax.set_xlabel('Time')
#     ax.set_ylabel('RA [deg]')
#     ax = axs[2]
#     if dectime_cut is not None:
#         for xin in dectime_cut:
#             ax.axvline(xin[1], color='purple', ls='dashed')
#             ax.axhline(xin[0], color='purple', ls='dashed')
#             ax.scatter(xin[1], xin[0], color='purple', marker='x')
#     if showFulLCands:
#         ax.scatter(df_final['time'], df_final['dec'], color='black')
#     ax.scatter(df_final_cand0['time'], df_final_cand0['dec'], alpha=0.5, color='orange')
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Dec [deg]')
#     fig.tight_layout()
#     plt.show()
#     return


def _plot_cut_region(
    ax,
    cut_points,
    point_cols,
    plot_xcol,
    plot_ycol,
    facecolor="purple",
    edgecolor="purple",
    alpha=0.18,
    label="cut region",
):
    """
    Draw a filled convex hull region.

    cut_points are in the coordinate order `point_cols`.
    The plot axes are defined by `plot_xcol`, `plot_ycol`.
    """

    if cut_points is None:
        return

    hull_points = _get_ordered_convex_hull(cut_points)

    hull_df = pd.DataFrame(hull_points, columns=point_cols)

    polygon_xy = hull_df[[plot_xcol, plot_ycol]].to_numpy()

    patch = MplPolygon(
        polygon_xy,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=2,
        zorder=2,
        label=label,
    )

    ax.add_patch(patch)

    ax.plot(
        polygon_xy[:, 0],
        polygon_xy[:, 1],
        color=edgecolor,
        linewidth=2,
        zorder=3,
    )

    ax.scatter(
        polygon_xy[:, 0],
        polygon_xy[:, 1],
        color=edgecolor,
        marker="x",
        s=70,
        linewidths=2,
        zorder=4,
    )


def _shade_time_min(ax, timeMin_cut):
    if timeMin_cut is None:
        return

    xmin, xmax = ax.get_xlim()

    ax.axvspan(
        xmin,
        timeMin_cut,
        color="red",
        alpha=0.08,
        zorder=0,
        label="excluded by timeMin",
    )

    ax.axvline(
        timeMin_cut,
        color="red",
        linestyle="--",
        linewidth=1.5,
        zorder=3,
    )

    ax.set_xlim(xmin, xmax)

def _get_ordered_convex_hull(points):
    """
    Takes unordered 2D points and returns the convex hull vertices
    ordered around the boundary.
    """
    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = np.unique(pts, axis=0)

    if len(pts) < 3:
        raise ValueError("Need at least 3 unique finite points to define an area.")

    try:
        hull = ConvexHull(pts)
    except QhullError:
        raise ValueError("Polygon points appear to be collinear or degenerate.")

    return pts[hull.vertices]



def plt_RegionCut(
    df_final,
    df_final_cand0,
    signalSelected,
    radec_cut=None,
    ratime_cut=None,
    dectime_cut=None,
    timeMin_cut=None,
    showFulLCands=True,
):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    # -------------------------
    # RA-Dec panel
    # -------------------------
    ax = axs[0]

    _plot_cut_region(
        ax,
        radec_cut,
        point_cols=("ra", "dec"),
        plot_xcol="ra",
        plot_ycol="dec",
        label="RA-Dec cut",
    )

    if showFulLCands:
        ax.scatter(
            df_final["ra"],
            df_final["dec"],
            color="black",
            s=8,
            alpha=0.18,
            rasterized=True,
            label="all candidates",
            zorder=1,
        )

    ax.scatter(
        df_final_cand0["ra"],
        df_final_cand0["dec"],
        color="orange",
        edgecolor="none",
        s=18,
        alpha=0.75,
        label="kept candidates",
        zorder=5,
    )

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(f"{signalSelected}: RA-Dec")
    ax.grid(alpha=0.25)

    # -------------------------
    # Time-RA panel
    # -------------------------
    ax = axs[1]

    _plot_cut_region(
        ax,
        ratime_cut,
        point_cols=("ra", "time"),
        plot_xcol="time",
        plot_ycol="ra",
        label="RA-time cut",
    )

    if showFulLCands:
        ax.scatter(
            df_final["time"],
            df_final["ra"],
            color="black",
            s=8,
            alpha=0.18,
            rasterized=True,
            label="all candidates",
            zorder=1,
        )

    ax.scatter(
        df_final_cand0["time"],
        df_final_cand0["ra"],
        color="orange",
        edgecolor="none",
        s=18,
        alpha=0.75,
        label="kept candidates",
        zorder=5,
    )

    _shade_time_min(ax, timeMin_cut)

    ax.set_xlabel("Time")
    ax.set_ylabel("RA [deg]")
    ax.set_title(f"{signalSelected}: Time-RA")
    ax.grid(alpha=0.25)

    # -------------------------
    # Time-Dec panel
    # -------------------------
    ax = axs[2]

    _plot_cut_region(
        ax,
        dectime_cut,
        point_cols=("dec", "time"),
        plot_xcol="time",
        plot_ycol="dec",
        label="Dec-time cut",
    )

    if showFulLCands:
        ax.scatter(
            df_final["time"],
            df_final["dec"],
            color="black",
            s=8,
            alpha=0.18,
            rasterized=True,
            label="all candidates",
            zorder=1,
        )

    ax.scatter(
        df_final_cand0["time"],
        df_final_cand0["dec"],
        color="orange",
        edgecolor="none",
        s=18,
        alpha=0.75,
        label="kept candidates",
        zorder=5,
    )

    _shade_time_min(ax, timeMin_cut)

    ax.set_xlabel("Time")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(f"{signalSelected}: Time-Dec")
    ax.grid(alpha=0.25)

    for ax in axs:
        ax.legend(frameon=True, fontsize=8)

    plt.show()

    return fig, axs