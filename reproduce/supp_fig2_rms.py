"""Supplementary Figure 2: RMS change in the linear PiP surface between successive checkpoints.

RMS values are recomputed from the saved checkpoint counts, except for the
airport network, whose checkpoint matrices are too large for git (its values
are read from results/airport/rms.csv). Dashed line: threshold 0.005.

Run:  python reproduce/supp_fig2_rms.py
"""
import csv

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

import _paper as P

THRESHOLD = 0.005
STYLE = {"hospital": ("#cb2432", "o"), "airport": ("#ed8428", "^"), "yeast": ("#641f36", "s"),
         "language": ("#567c91", "X"), "watts_strogatz": ("#bd930b", "v"), "erdos_renyi": ("#252525", "D")}
TICKS = {1e7: "10M", 2.5e7: "25M", 5e7: "50M", 1e8: "100M", 2.5e8: "250M", 5e8: "500M", 1e9: "1B"}


def main():
    args = P.parser(__doc__).parse_args()
    plt = P.setup_matplotlib()
    fig, ax = plt.subplots(figsize=(183 * P.MM, 115 * P.MM))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.75, bottom=0.12)
    handles, table = [], []
    for key, cfg in P.NETWORKS.items():
        rows = [r for r in P.rms_rows(key) if r["attacks"] >= 10_000_000]
        table += [dict(network=key, **r) for r in rows]
        x = np.array([r["attacks"] for r in rows], float)
        y = np.array([r["rms"] for r in rows])
        color, marker = STYLE[key]
        every = np.unique(np.linspace(0, len(x) - 1, min(12, len(x)), dtype=int))
        ax.plot(x, y, color=color, lw=2.6, marker=marker, ms=7, markevery=every, markeredgecolor="white",
                markeredgewidth=0.8, zorder=3, clip_on=False)
        handles.append(Line2D([], [], color=color, lw=2.6, marker=marker, ms=7, markeredgecolor="white",
                              markeredgewidth=0.8, label=cfg["title"]))
    ax.axhline(THRESHOLD, color="#6b7079", lw=1.6, ls=(0, (6, 4)), zorder=1)
    handles.append(Line2D([], [], color="#6b7079", lw=1.6, ls=(0, (6, 4)), label="RMS = 0.005"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e7, 1e9)
    ax.xaxis.set_major_locator(FixedLocator(list(TICKS)))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: TICKS.get(v, "")))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="major", labelsize=10, length=4, pad=3)
    ax.set_xlabel("Sampled attacks", fontsize=11, labelpad=5)
    ax.set_ylabel("RMS change in PiP surface", fontsize=11, labelpad=5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.995), ncol=2, frameon=False,
               fontsize=10, handlelength=2.6, columnspacing=1.6, labelspacing=0.7)
    P.save(fig, "SuppFig2_RMS_convergence", args.dpi)
    with (P.OUT / "SuppFig2_RMS_values.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


if __name__ == "__main__":
    main()
