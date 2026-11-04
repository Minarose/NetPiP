"""Figure 5: high-PiP (critical) nodes of the brain language network on the cortical surface.

The critical set is the Ward cluster of linear PiP profiles with the highest mean
participation (k chosen by silhouette). Cerebellar nodes lie below the cortical
surface and are not drawn. All three views share one scale.

Run:  python reproduce/fig5_language_map.py [--dpi 1200]
"""
import numpy as np
from matplotlib.lines import Line2D

import _brain
import _paper as P
from _paper import netpip

PIP_RED, OTHER = "#df242b", "#9aa0a8"


def draw_view(ax, view, hemis, xyz, critical, keep):
    lo, hi = _brain.draw_cortex(ax, view, hemis=hemis)
    pts, _ = _brain.project_points(xyz, view)
    idx = np.flatnonzero(keep)
    other, crit = idx[~critical[idx]], idx[critical[idx]]
    ax.scatter(pts[other, 0], pts[other, 1], s=16, color=OTHER, edgecolor="white", linewidth=0.5, zorder=5)
    ax.scatter(pts[crit, 0], pts[crit, 1], s=46, color=PIP_RED, edgecolor="white", linewidth=0.7, zorder=6)
    pad = 6
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    return hi - lo + 2 * pad


def main():
    args = P.parser(__doc__).parse_args()
    plt = P.setup_matplotlib()
    rows = P.nodes("language")
    xyz = np.array([[float(r["mni_x"]), float(r["mni_y"]), float(r["mni_z"])] for r in rows])
    cluster = netpip.critical_cluster(P.surface("language", "linear")[1].T)
    critical = np.zeros(len(rows), bool)
    critical[cluster["critical_nodes"]] = True
    cortical = np.array([not r["aal_label"].startswith("Cerebelum") for r in rows])
    names = [f"{i + 1} {rows[i]['aal_label']}" for i in cluster["critical_nodes"]]
    print(f"  critical nodes: {', '.join(names)}")

    panels = [("a", "Left hemisphere", "left", ("left",), cortical & (xyz[:, 0] <= 0)),
              ("b", "Right hemisphere", "right", ("right",), cortical & (xyz[:, 0] >= 0)),
              ("c", "Top view", "top", ("left", "right"), cortical)]
    width, side, gap, title_h, legend_h = 183, 2, 5, 9, 10
    fig = plt.figure(figsize=(width * P.MM, 100 * P.MM))
    axes, extents = [], []
    for _, _, view, hemis, keep in panels:
        ax = fig.add_axes([0, 0, 1, 1])
        extents.append(draw_view(ax, view, hemis, xyz, critical, keep))
        axes.append(ax)
    extents = np.array(extents)
    scale = (width - 2 * side - gap * (len(panels) - 1)) / extents[:, 0].sum()  # mm per MNI unit
    brain_h = extents[:, 1].max() * scale
    height = title_h + brain_h + legend_h
    fig.set_size_inches(width * P.MM, height * P.MM)
    x = side
    for ax, (w, h), (letter, title, *_) in zip(axes, extents * scale, panels):
        ax.set_position([x / width, (legend_h + (brain_h - h) / 2) / height, w / width, h / height])
        top = (legend_h + brain_h + 1.5) / height
        fig.text((x + 1) / width, top, letter, fontsize=12, fontweight="bold", color=P.INK, va="bottom")
        fig.text((x + w / 2) / width, top, title, fontsize=9, color=P.INK, va="bottom", ha="center")
        x += w + gap
    handles = [Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=PIP_RED, markeredgecolor="white",
                      label=f"High PiP nodes ({critical.sum()})"),
               Line2D([], [], marker="o", ls="", markersize=5, markerfacecolor=OTHER, markeredgecolor="white",
                      label="Other brain language network nodes")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               fontsize=9, handletextpad=0.4, columnspacing=2.0)
    P.save(fig, "Fig5_language_critical_nodes", args.dpi)


if __name__ == "__main__":
    main()
