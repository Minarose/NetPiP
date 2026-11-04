"""Supplementary Figure 3 and Table 1: LCC and SLCC trajectories of PiP- and centrality-ordered attacks.

Curves are means over the saved tie-breaking orders; dots mark each method's
percolation point. Table 1 (LCC area and removals to 10% LCC) is written as CSV.

Run:  python reproduce/supp_fig3_lcc.py
"""
import csv

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import _paper as P
from _paper import netpip

DRAW_ORDER = ["Degree", "Betweenness", "PageRank", "Linear PiP"]  # PiP on top


def main():
    args = P.parser(__doc__).parse_args()
    plt = P.setup_matplotlib()
    fig = plt.figure(figsize=(183 * P.MM, 215 * P.MM))
    outer = fig.add_gridspec(2, 1, left=0.10, right=0.985, top=0.90, bottom=0.05, hspace=0.24)
    grids = [outer[k].subgridspec(2, 3, hspace=0.12, wspace=0.34) for k in range(2)]
    table = []
    for i, (key, cfg) in enumerate(P.NETWORKS.items()):
        A = P.adjacency(key)
        n = len(A)
        csr = netpip.io.to_csr(A)
        orders = P.saved_orders(key)
        g = grids[i // 3]
        ax_l = fig.add_subplot(g[0, i % 3])
        ax_s = fig.add_subplot(g[1, i % 3], sharex=ax_l)
        slcc_max = 0
        q = np.arange(n + 1) / n
        for method in DRAW_ORDER:
            traces = [netpip.component_trace(csr, o) for o in orders[method]]
            lcc = np.mean([t[0] for t in traces], axis=0) / n
            slcc = np.mean([t[1] for t in traces], axis=0) / n
            stats = netpip.evaluate_attacks(csr, orders[method])
            table.append(dict(network=key, method=P.LABELS[method], nodes=n,
                              lcc_area=round(stats["lcc_area"], 3),
                              removals_to_10pct=stats["removals_to_10pct"],
                              removals_to_10pct_percent=round(100 * stats["removals_to_10pct"] / n, 1)))
            top = method == "Linear PiP"
            lw = 2.2 if top else 1.6
            ax_l.plot(q, lcc, color=P.COLORS[method], lw=lw, zorder=3 if top else 2)
            ax_s.plot(q, slcc, color=P.COLORS[method], lw=lw, zorder=3 if top else 2)
            qp = stats["peak_removals"] / n
            ax_s.scatter([qp], [np.interp(qp, q, slcc)], s=30, color=P.COLORS[method], edgecolor="white",
                         linewidth=0.7, zorder=5)
            slcc_max = max(slcc_max, slcc.max())
        ax_l.set_ylim(0, 1)
        ax_l.set_yticks([0, 0.5, 1])
        ax_l.set_yticklabels(["0", "0.5", "1"])
        ax_s.set_ylim(0, slcc_max * 1.2)
        ax_s.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_s.set_xlim(0, 1)
        ax_s.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax_s.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        plt.setp(ax_l.get_xticklabels(), visible=False)
        for ax in (ax_l, ax_s):
            ax.tick_params(labelsize=8.5, length=3, pad=2)
            ax.spines[["top", "right"]].set_visible(False)
        ax_l.set_title(cfg["title"], fontsize=9, pad=5)
        ax_l.text(-0.24, 1.12, "abcdef"[i], transform=ax_l.transAxes, fontsize=12, fontweight="bold",
                  color=P.INK, va="bottom")
        if i % 3 == 0:
            ax_l.set_ylabel("LCC\n(fraction of nodes)", fontsize=9, labelpad=3)
            ax_s.set_ylabel("SLCC\n(fraction of nodes)", fontsize=9, labelpad=3)
        ax_s.set_xlabel("Fraction of nodes removed", fontsize=9, labelpad=3)
    handles = [Line2D([], [], color=P.COLORS[m], lw=2.2, label=P.LABELS[m]) for m in P.METHODS]
    handles.append(Line2D([], [], color=P.INK, marker="o", ls="", ms=5.5, markeredgecolor="white",
                          label="Percolation point"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.995), ncol=5, frameon=False,
               fontsize=9.5, handlelength=2.2, columnspacing=1.5)
    P.save(fig, "SuppFig3_LCC_SLCC_curves", args.dpi)
    table.sort(key=lambda r: (list(P.NETWORKS).index(r["network"]), [P.LABELS[m] for m in P.METHODS].index(r["method"])))
    with (P.OUT / "SuppTable1_LCC_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print("  wrote reproduce/output/SuppTable1_LCC_metrics.csv")


if __name__ == "__main__":
    main()
