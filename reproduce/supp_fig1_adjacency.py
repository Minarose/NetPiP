"""Supplementary Figure 1: binary adjacency matrices of the six networks.

The airport matrix (3,214 nodes) is drawn with each pixel covering a 4 x 4 block,
black if the block contains any edge.

Run:  python reproduce/supp_fig1_adjacency.py
"""
import numpy as np

import _paper as P


def block_max(A, b):
    n = A.shape[0]
    m = -(-n // b) * b
    Q = np.zeros((m, m), bool)
    Q[:n, :n] = A
    return Q.reshape(m // b, b, m // b, b).any(axis=(1, 3))


def main():
    args = P.parser(__doc__).parse_args()
    plt = P.setup_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(183 * P.MM, 128 * P.MM))
    fig.subplots_adjust(left=0.10, right=0.955, top=0.91, bottom=0.07, wspace=0.42, hspace=0.42)
    for i, (ax, (key, cfg)) in enumerate(zip(axes.flat, P.NETWORKS.items())):
        A = P.adjacency(key).astype(bool)
        n, e = len(A), int(A.sum() // 2)
        shown = block_max(A, 4) if key == "airport" else A
        ax.imshow(shown, cmap="Greys", vmin=0, vmax=1, interpolation="nearest", origin="lower",
                  extent=(0.5, n + 0.5, 0.5, n + 0.5))
        ax.set_xticks([1, n])
        ax.set_yticks([1, n])
        ax.set_xticklabels(["1", f"{n:,}"])
        ax.set_yticklabels(["1", f"{n:,}"])
        ax.tick_params(labelsize=9, length=3, pad=2)
        ax.set_xlabel("Node", fontsize=9.5, labelpad=1)
        ax.set_ylabel("Node", fontsize=9.5, labelpad=1)
        ax.set_title(f"{cfg['title']}\n{n:,} nodes, {e:,} edges", fontsize=9.5, pad=6, linespacing=1.3)
        ax.text(-0.26, 1.13, "abcdef"[i], transform=ax.transAxes, fontsize=12, fontweight="bold",
                color=P.INK, va="bottom")
    P.save(fig, "SuppFig1_adjacency_matrices", args.dpi)


if __name__ == "__main__":
    main()
