"""Figure 3: exponential-weighted PiP surfaces of the six networks (3D, one colour scale per panel).

Run:  python reproduce/fig3_surfaces.py [--dpi 1200]
"""
import numpy as np

import _paper as P


def display_surface(key):
    """Retained steps as a contiguous span; unobserved steps inside it (hospital step 2) shown at zero."""
    steps, W = P.surface(key, "exponential")
    display = np.zeros((int(steps[-1] - steps[0] + 1), W.shape[1]))
    display[steps - steps[0]] = W
    return np.arange(steps[0], steps[-1] + 1), display


def mesh(steps, display, limit=None):
    step_idx, node_idx = np.arange(len(steps)), np.arange(display.shape[1])
    if limit is not None:  # airport: reduced mesh that keeps the peak and the 15 highest-mean nodes
        peak_step, peak_node = np.unravel_index(np.argmax(display), display.shape)
        step_idx = np.unique(np.r_[np.linspace(0, len(steps) - 1, min(len(steps), limit), dtype=int), peak_step])
        node_idx = np.unique(np.r_[np.linspace(0, display.shape[1] - 1, min(display.shape[1], limit), dtype=int),
                                   np.argsort(-display.mean(axis=0))[:15], peak_node])
    X, Y = np.meshgrid(steps[step_idx], node_idx + 1)
    return X, Y, display[np.ix_(step_idx, node_idx)].T


def main():
    args = P.parser(__doc__).parse_args()
    plt = P.setup_matplotlib()
    import matplotlib
    from matplotlib import cm, colors
    from matplotlib.ticker import MaxNLocator

    fig = plt.figure(figsize=(183 * P.MM, 128 * P.MM))
    w, h = 0.235, 0.40
    lefts, bottoms = [0.035, 0.355, 0.675], [0.50, 0.03]
    cmap = matplotlib.colormaps["gist_heat_r"]
    for i, (key, cfg) in enumerate(P.NETWORKS.items()):
        steps, display = display_surface(key)
        X, Y, Z = mesh(steps, display, limit=160 if key == "airport" else None)
        vmax = float(display.max())
        ax = fig.add_axes([lefts[i % 3], bottoms[i // 3], w, h], projection="3d")
        # Colour each patch by the highest of its four corners so spikes reach the top colour.
        corner_max = np.maximum.reduce([Z[:-1, :-1], Z[1:, :-1], Z[:-1, 1:], Z[1:, 1:]])
        patch = np.pad(corner_max, ((0, 1), (0, 1)), mode="edge")
        norm = colors.Normalize(vmin=0, vmax=vmax)
        surf = ax.plot_surface(X, Y, Z, facecolors=cmap(norm(patch)), linewidth=0, antialiased=False,
                               rstride=1, cstride=1, alpha=0.9, shade=False)
        surf.set_rasterized(True)
        ax.view_init(elev=20, azim=130)
        ax.set_xlim(steps[0], steps[-1])
        ax.set_ylim(display.shape[1], 1)
        ax.invert_xaxis()
        ax.set_zticks([])
        ax.zaxis.line.set_color((1, 1, 1, 0))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1, 1, 1, 0))
            axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.tick_params(axis="x", labelsize=7, pad=-2)
        ax.tick_params(axis="y", labelsize=7, pad=-2)
        ax.set_xlabel("Removal step", fontsize=8, labelpad=-4)
        ax.set_ylabel("Node index", fontsize=8, labelpad=-4)
        ax.set_box_aspect((4, 4, 3), zoom=1.12)

        cax = fig.add_axes([lefts[i % 3] + w + 0.035, bottoms[i // 3] + 0.12, 0.009, h * 0.55])
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cbar.set_ticks([0, vmax / 2, vmax])
        cbar.set_ticklabels(["0", f"{vmax / 2:.2f}", f"{vmax:.2f}"])
        cbar.ax.tick_params(labelsize=7, length=2, pad=1.5)
        cbar.outline.set_linewidth(0.5)
        cbar.ax.set_title("PiP", fontsize=8, pad=4, color=P.INK)
        fig.text(lefts[i % 3] - 0.03, bottoms[i // 3] + h - 0.005, "abcdef"[i], fontsize=12,
                 fontweight="bold", color=P.INK, va="bottom")
        fig.text(lefts[i % 3] + w / 2 + 0.02, bottoms[i // 3] + h - 0.005, cfg["title"], fontsize=9,
                 color=P.INK, va="bottom", ha="center")
    P.save(fig, "Fig3_PiP_surfaces", args.dpi)


if __name__ == "__main__":
    main()
