"""Figure 4: nodes removed at the percolation point for PiP- vs centrality-ordered attacks.

Bars use the saved attack orders (500 tie-breaking orders for degree, betweenness
and PageRank); error bars are ±1 s.d. across tie orders, drawn where s.d. > 0.

Run:  python reproduce/fig4_attacks.py [--dpi 600]
"""
from matplotlib.ticker import MaxNLocator

import _paper as P
from _paper import netpip


def main():
    args = P.parser(__doc__).parse_args()
    plt = P.setup_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(183 * P.MM, 118 * P.MM))
    fig.subplots_adjust(left=0.085, right=0.968, top=0.93, bottom=0.12, wspace=0.42, hspace=0.62)
    x = range(len(P.METHODS))
    for i, (ax, (key, cfg)) in enumerate(zip(axes.flat, P.NETWORKS.items())):
        A = P.adjacency(key)
        n = len(A)
        orders = P.saved_orders(key)
        stats = {m: netpip.evaluate_attacks(netpip.io.to_csr(A), orders[m]) for m in P.METHODS}
        means = [100 * stats[m]["peak_removals"] / n for m in P.METHODS]
        sds = [100 * stats[m]["peak_removals_sd"] / n for m in P.METHODS]
        ax.bar(x, means, width=0.72, color=[P.COLORS[m] for m in P.METHODS], zorder=2)
        for xi, mean, sd in zip(x, means, sds):
            if sd > 0:
                ax.errorbar(xi, mean, yerr=sd, color=P.INK, lw=1.0, capsize=3, capthick=1.0, zorder=3)
        top = max(m + s for m, s in zip(means, sds))
        for xi, mean, sd in zip(x, means, sds):
            ax.text(xi, mean + sd + top * 0.025, f"{mean:.1f}", ha="center", va="bottom", fontsize=7.5, color=P.INK)
        ax.set_ylim(0, top * 1.22)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.set_xticks(list(x))
        ax.set_xticklabels([P.LABELS[m] for m in P.METHODS], rotation=35, ha="right", rotation_mode="anchor",
                           fontsize=8.5)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_xlim(-0.6, len(P.METHODS) - 0.4)
        ax.tick_params(axis="y", labelsize=9, length=3, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(cfg["title"], fontsize=9, color=P.INK, pad=6)
        ax.text(-0.30, 1.13, "abcdef"[i], transform=ax.transAxes, fontsize=12, fontweight="bold",
                color=P.INK, va="bottom", ha="left")
        if i % 3 == 0:
            ax.set_ylabel("Nodes removed at\npercolation point (%)", fontsize=9.5, labelpad=4)
    P.save(fig, "Fig4_PiP_vs_centrality", args.dpi)


if __name__ == "__main__":
    main()
