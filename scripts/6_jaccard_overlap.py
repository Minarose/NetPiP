#!/usr/bin/env python3
"""
Average-graph percolation-point + top-n node overlap/visualization.

Given:
  - an average binary adjacency MAT with `psi_adj` (66×66)
  - an average PiP convergence MAT with `node_P` (steps×66)
  - MNI coordinates for the 66 AAL nodes (one "x y z" triplet per line)

This script:
  1) Computes percolation point for four strategies on the single AVG graph:
       Degree, Betweenness, PageRank, PiP (tilt τ=S/6 + negative->0 clip order)
  2) Saves a clean bar plot of percolation point (one value per strategy).
  3) Plots the top-n nodes from each strategy on a nilearn brain (one image per strategy).
  4) Computes Jaccard similarity between PiP top-n and each metric top-n
     (optionally across a sweep of n).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.colors import to_hex

try:
    from nilearn import plotting  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "nilearn is required for brain plots. Install it via the top-level "
        "requirements (`pip install -r requirements.txt`) or `pip install nilearn`."
    ) from exc

try:
    import networkx as nx  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("networkx is required (added to requirements.txt).") from exc

try:
    from matplotlib_venn import venn2  # type: ignore
except Exception:  # pragma: no cover
    venn2 = None

# Reuse canonical PiP MAT loader + ranking logic (handles v7.3 MAT via h5py)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers.pip_plot_utils import (  # noqa: E402
    crop_longest_non_nan_block,
    get_tilt_peak_order_amplitude,
    load_pip_any,
)


@dataclass(frozen=True)
class StrategyResult:
    name: str
    order0: np.ndarray  # 0-based node order, length N, first removed first
    perc_point: int  # 1-based removal index where 2nd-largest component peaks


def load_binary_adj(mat_path: Path) -> np.ndarray:
    d = sio.loadmat(mat_path)
    if "psi_adj" not in d:
        raise KeyError(f"`psi_adj` not found in {mat_path}")
    adj = np.array(d["psi_adj"], dtype=float)
    adj = (adj > 0).astype(float)
    adj = np.triu(adj, 1)
    adj = adj + adj.T
    np.fill_diagonal(adj, 0.0)
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square; got {adj.shape}")
    return adj


def load_node_p(conv_path: Path) -> np.ndarray:
    # NOTE: many ConvHW.mat files are MATLAB v7.3 (HDF5) which scipy.io.loadmat
    # cannot read. helpers.pip_plot_utils.load_pip_any handles both classic + v7.3.
    P = load_pip_any(str(conv_path), varname="node_P")
    P = crop_longest_non_nan_block(P)
    if P.size == 0:
        raise ValueError(f"Empty node_P after crop in {conv_path}")
    return P


def percolation_point(adj: np.ndarray, order0: np.ndarray) -> int:
    """
    Mirror MATLAB run_attack_once():
      - remove nodes one by one in attack order
      - track size of 2nd-largest component after each removal
      - return first index (1..N) maximizing that 2nd-largest size
      - if it never exceeds 1, return N
    """
    n = adj.shape[0]
    if order0.shape[0] != n:
        raise ValueError(f"order length {order0.shape[0]} != n {n}")

    A = adj.copy()
    second = np.zeros(n, dtype=int)
    for i in range(n):
        # compute connected component sizes
        n_comp, labels = connected_components(csr_matrix(A), directed=False, connection="weak")
        if n_comp <= 1:
            second[i] = 0
        else:
            sizes = np.bincount(labels, minlength=n_comp)
            sizes.sort()
            second[i] = int(sizes[-2])

        node = int(order0[i])
        A[node, :] = 0.0
        A[:, node] = 0.0

    mx = int(second.max())
    if mx > 1:
        return int(np.where(second == mx)[0][0] + 1)  # 1-based
    return int(n)


def break_ties_randomly_desc(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Sort descending with random tie-breaking.
    Returns 0-based node order, first removed first.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    n = v.size
    # stable-ish: for each unique value descending, shuffle indices within that value
    order: list[int] = []
    for val in np.unique(v)[::-1]:
        idx = np.where(v == val)[0]
        if idx.size > 1:
            rng.shuffle(idx)
        order.extend(idx.tolist())
    if len(order) != n:
        raise RuntimeError("Tie-break ordering failed.")
    return np.array(order, dtype=int)


def pip_order_from_node_p(P_raw: np.ndarray, clip_negative: bool = True) -> np.ndarray:
    order0, _, _ = get_tilt_peak_order_amplitude(
        P_raw, tau_factor=1.0 / 6.0, clip_negative=clip_negative
    )
    return order0.astype(int)


def metric_orders_from_adj(
    adj: np.ndarray, n_iter: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Degree, betweenness, pagerank values computed once; random tie-break across n_iter
    is handled by sampling orders. We return the *mean percolation point* across
    those random orders, but for top-n visualization we also return one canonical
    order (using the RNG once).
    """
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    G = nx.from_numpy_array(adj)

    deg = np.array([d for _, d in G.degree(weight=None)], dtype=float)
    bc = np.array(list(nx.betweenness_centrality(G, normalized=False).values()), dtype=float)
    pr = np.array(list(nx.pagerank(G, alpha=0.85).values()), dtype=float)

    # produce one representative order per metric (for brain plot / Jaccard)
    deg_order0 = break_ties_randomly_desc(deg, rng)
    bc_order0 = break_ties_randomly_desc(bc, rng)
    pr_order0 = break_ties_randomly_desc(pr, rng)

    # compute mean percolation point with resampled tie-breaks for fairness
    def mean_pp(vals: np.ndarray) -> float:
        pps = np.zeros(n_iter, dtype=float)
        for i in range(n_iter):
            ord0 = break_ties_randomly_desc(vals, rng)
            pps[i] = percolation_point(adj, ord0)
        return float(pps.mean())

    # We keep the representative orders above; percolation points are computed outside.
    return deg_order0, bc_order0, pr_order0


def load_coords(coords_path: Path, n_nodes: int) -> np.ndarray:
    coords = np.loadtxt(coords_path, dtype=float)
    coords = np.atleast_2d(coords)
    if coords.shape != (n_nodes, 3):
        raise ValueError(f"coords must be shape ({n_nodes},3); got {coords.shape}")
    return coords


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    A = set(a)
    B = set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def save_barplot(out_path: Path, results: list[StrategyResult]) -> None:
    # Match paper aesthetics (Seaborn + gist_heat palette)
    cmap = plt.cm.gist_heat
    palette = {
        "Betweenness": cmap(0.9),
        "Degree": cmap(0.8),
        "PageRank": cmap(0.7),
        "PiP": cmap(0.485),
    }

    df = pd.DataFrame(
        {
            "Group": [r.name for r in results if r.name in palette],
            "Value": [int(r.perc_point) for r in results if r.name in palette],
        }
    )
    df = df.sort_values("Value", ascending=True).reset_index(drop=True)
    order = df["Group"].tolist()

    sns.set_theme(context="paper", style="white")
    plt.figure(figsize=(10, 7), dpi=160)
    ax = sns.barplot(
        x="Group",
        y="Value",
        hue="Group",
        data=df,
        order=order,
        palette=palette,
        errorbar=None,
        saturation=0.9,
        legend=False,
    )

    max_v = float(df["Value"].max())
    label_offset = max(max_v * 0.02, 0.15)
    for i, v in enumerate(df["Value"].tolist()):
        ax.text(
            i,
            v + label_offset,
            str(int(v)),
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="semibold",
        )

    ax.set_ylim(0, max_v * 1.15 + 0.5)
    ax.set_xlabel("Attack Strategy", fontsize=18)
    ax.set_ylabel("Percolation Point (# nodes)", fontsize=18)
    ax.set_title("Percolation Point by Attack Strategy", fontsize=20)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(False)
    sns.despine(ax=ax, top=True, right=True)
    ax.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_jaccard_barplot(out_path: Path, jac: dict[str, float]) -> None:
    """Bar plot for Jaccard(PiP, metric) with same aesthetics."""
    cmap = plt.cm.gist_heat
    palette = {
        "Betweenness": cmap(0.9),
        "Degree": cmap(0.8),
        "PageRank": cmap(0.7),
    }
    order = ["Betweenness", "Degree", "PageRank"]
    df = pd.DataFrame({"Group": list(jac.keys()), "Value": list(jac.values())})

    sns.set_theme(context="paper", style="whitegrid")
    plt.figure(figsize=(10, 7), dpi=160)
    ax = sns.barplot(
        x="Group",
        y="Value",
        hue="Group",
        data=df,
        order=order,
        palette=palette,
        errorbar=None,
        saturation=0.9,
        legend=False,
    )
    ax.set_xlabel("Attack Strategy", fontsize=18)
    ax.set_ylabel("Jaccard Similarity", fontsize=18)
    ax.set_title("Node-set overlap with PiP (Jaccard)", fontsize=20)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def paper_palette() -> dict[str, object]:
    """Shared palette (gist_heat) used across paper figures."""
    cmap = plt.cm.gist_heat
    return {
        "Betweenness": to_hex(cmap(0.9)),
        "Degree": to_hex(cmap(0.8)),
        "PageRank": to_hex(cmap(0.7)),
        "PiP": to_hex(cmap(0.485)),
    }


def save_brain_plot(
    out_path: Path,
    coords: np.ndarray,
    top_nodes0: np.ndarray,
    title: str,
    node_color,
    node_size: int = 85,
) -> None:
    # Use plot_connectome so we can force a single solid node color (paper palette).
    n = coords.shape[0]
    adj0 = np.zeros((n, n), dtype=float)  # no edges; nodes only
    node_colors = ["#FFFFFF"] * n
    node_sizes = np.zeros(n, dtype=float)
    for i in top_nodes0:
        node_colors[int(i)] = node_color
        node_sizes[int(i)] = float(node_size)
    disp = plotting.plot_connectome(
        adj0,
        coords,
        node_color=node_colors,
        node_size=node_sizes,
        display_mode="lzry",
        title=title,
        colorbar=False,
    )
    disp.savefig(out_path)
    disp.close()

def save_overlap_brain_plot(
    out_path: Path,
    coords: np.ndarray,
    pip_nodes0: Iterable[int],
    metric_nodes0: Iterable[int],
    title: str,
    pip_color,
    metric_color,
    node_size: int = 90,
) -> None:
    """
    Plot overlap between PiP and a metric.
      - PiP-only: red
      - metric-only: blue
      - shared: black
    """
    pip_set = set(int(x) for x in pip_nodes0)
    met_set = set(int(x) for x in metric_nodes0)
    shared = pip_set & met_set
    pip_only = pip_set - shared
    met_only = met_set - shared

    # Use plot_connectome for per-node colors + per-node sizes (cleaner than colormap hacks).
    n = coords.shape[0]
    adj0 = np.zeros((n, n), dtype=float)  # no edges; nodes only

    node_colors = ["#FFFFFF"] * n
    node_sizes = np.zeros(n, dtype=float)
    for i in pip_only:
        node_colors[i] = pip_color
        node_sizes[i] = float(node_size)
    for i in shared:
        node_colors[i] = "#000000"
        node_sizes[i] = float(node_size)
    for i in met_only:
        node_colors[i] = metric_color
        node_sizes[i] = float(node_size)

    disp = plotting.plot_connectome(
        adj0,
        coords,
        node_color=node_colors,
        node_size=node_sizes,
        display_mode="lzry",
        title=title,
        colorbar=False,
    )
    disp.savefig(out_path)
    disp.close()

def save_pip_order_gradient_plot(
    out_path: Path,
    coords: np.ndarray,
    pip_order0: np.ndarray,
    title: str = "PiP order (earlier = darker)",
    node_size: int = 60,
    cmap=plt.cm.Reds,
) -> None:
    """Plot all nodes colored by PiP order rank (earlier = darker)."""
    n = coords.shape[0]
    if pip_order0.shape[0] != n:
        raise ValueError(f"pip_order0 length {pip_order0.shape[0]} != n {n}")
    # rank[i] = position in order (0..n-1)
    rank = np.empty(n, dtype=int)
    rank[pip_order0] = np.arange(n, dtype=int)
    # intensity: 1 for earliest, 0 for latest
    if n > 1:
        vals = 1.0 - (rank.astype(float) / float(n - 1))
    else:
        vals = np.ones(n, dtype=float)

    # Map scalar intensities to explicit colors (hex) since nilearn 0.11.x
    # plot_connectome doesn't accept node_cmap/node_vmin/node_vmax.
    node_colors = [to_hex(cmap(v)) for v in vals.tolist()]
    adj0 = np.zeros((n, n), dtype=float)
    disp = plotting.plot_connectome(
        adj0,
        coords,
        node_color=node_colors,
        node_size=np.full(n, float(node_size)),
        display_mode="lzry",
        title=title,
        colorbar=False,
    )
    disp.savefig(out_path)
    disp.close()

def write_brainnet_node(
    out_path: Path,
    coords: np.ndarray,
    pip_nodes0: Iterable[int],
    metric_nodes0: Iterable[int],
    metric_name: str,
    node_size: float = 4.0,
) -> None:
    """
    BrainNet Viewer .node file for overlap visualization.

    NOTE: BrainNet has multiple .node conventions depending on version/build.
    The BrainNet build used here interprets columns as:
      x y z module size label
    (i.e., module/group is column 4, node size is column 5).

    We encode group membership via the `module` code (1-based):
      - PiP-only: 1
      - shared: 2
      - metric-only: 3

    We write ONLY the selected nodes (not all 66).
    """
    pip_set = set(int(x) for x in pip_nodes0)
    met_set = set(int(x) for x in metric_nodes0)
    shared = pip_set & met_set
    pip_only = pip_set - shared
    met_only = met_set - shared

    with open(out_path, "w", encoding="utf-8") as f:
        for idx0 in sorted(pip_only):
            x, y, z = coords[idx0]
            # x y z module size label
            f.write(f"{x:.6f}\t{y:.6f}\t{z:.6f}\t1\t{node_size:.3f}\tPiP_only\n")
        for idx0 in sorted(shared):
            x, y, z = coords[idx0]
            f.write(f"{x:.6f}\t{y:.6f}\t{z:.6f}\t2\t{node_size:.3f}\tShared\n")
        for idx0 in sorted(met_only):
            x, y, z = coords[idx0]
            f.write(f"{x:.6f}\t{y:.6f}\t{z:.6f}\t3\t{node_size:.3f}\t{metric_name}_only\n")

def save_venn(
    out_path: Path,
    a: Iterable[int],
    b: Iterable[int],
    label_a: str,
    label_b: str,
    title: str,
    *,
    a_color: str = "#d32f2f",
    b_color: str = "#ffde59",
    intersection_color: str = "#FF8C00",
) -> None:
    if venn2 is None:
        raise RuntimeError(
            "matplotlib-venn is required for Venn diagrams. "
            "Install via `pip install -r requirements.txt`."
        )
    A = set(a)
    B = set(b)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=160)
    v = venn2([A, B], set_labels=(label_a, label_b), ax=ax)
    if v.get_patch_by_id("10") is not None:
        v.get_patch_by_id("10").set_color(a_color)
        v.get_patch_by_id("10").set_alpha(0.55)
    if v.get_patch_by_id("01") is not None:
        v.get_patch_by_id("01").set_color(b_color)
        v.get_patch_by_id("01").set_alpha(0.55)
    if v.get_patch_by_id("11") is not None:
        v.get_patch_by_id("11").set_color(intersection_color)
        v.get_patch_by_id("11").set_alpha(0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--avg-adj-mat",
        default=str(repo_root / "data" / "PSI_broadband_MEG_mats" / "avg" / "AVG_broadband_psi_adj_giant75_nonexcluded.mat"),
        help="Average adjacency MAT with `psi_adj` (binary).",
    )
    p.add_argument(
        "--avg-conv-mat",
        default=str(repo_root / "results" / "pip_convergence" / "avg_giant75" / "AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW.mat"),
        help="Average PiP MAT with `node_P`.",
    )
    p.add_argument(
        "--coords",
        default=str(repo_root / "data" / "MNI_66_coords.txt"),
        help="MNI coords text (66 lines, 3 cols).",
    )
    p.add_argument("--out-dir", default=str(repo_root / "results" / "graph_theory_overlap"), help="Output directory.")
    p.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Top-n nodes to plot and compare. If omitted and --perc-csv is provided, "
        "n is derived from MATLAB percolation points (see --top-n-from).",
    )
    p.add_argument("--n-iter", type=int, default=500, help="Tie-break resamples for BCT-like metrics.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for tie-breaks.")
    p.add_argument(
        "--perc-csv",
        default=None,
        help="Optional CSV exported by MATLAB script (avg_percolation_points_matlab.csv). "
        "If set, use those percolation points for the bar plot instead of recomputing.",
    )
    p.add_argument(
        "--top-n-from",
        choices=["pip", "metric"],
        default="pip",
        help="When deriving n from MATLAB percolation points: "
        "`pip` uses n = PiP percolation point for ALL strategies (recommended for Jaccard). "
        "`metric` uses each strategy's own percolation point for its brain plot; "
        "Jaccard still uses PiP's n unless you also pass --jaccard-n-from metric.",
    )
    p.add_argument(
        "--jaccard-n-from",
        choices=["pip", "metric"],
        default="pip",
        help="Which n to use for Jaccard comparisons when --perc-csv is provided. "
        "`pip` uses n = PiP percolation point. "
        "`metric` uses each metric's percolation point (compared to PiP truncated to that n).",
    )
    p.add_argument(
        "--jaccard-sweep",
        action="store_true",
        help="Also compute and save Jaccard(PiP, metric) for n=1..N.",
    )
    args = p.parse_args()

    avg_adj_mat = Path(args.avg_adj_mat)
    avg_conv_mat = Path(args.avg_conv_mat)
    coords_path = Path(args.coords)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adj = load_binary_adj(avg_adj_mat)
    P = load_node_p(avg_conv_mat)

    n_nodes = adj.shape[0]
    coords = load_coords(coords_path, n_nodes)

    # Strategy orders (0-based; first removed first)
    rng = np.random.default_rng(args.seed)

    # PiP: tilt τ=S/6 and clip negative to 0 (matches surfaces/label export defaults)
    pip_order0 = pip_order_from_node_p(P, clip_negative=True)
    # Degree / BC / PR: use networkx values, then randomized tie-break
    G = nx.from_numpy_array(adj)
    deg_vals = np.array([d for _, d in G.degree(weight=None)], dtype=float)
    bc_vals = np.array(list(nx.betweenness_centrality(G, normalized=False).values()), dtype=float)
    pr_vals = np.array(list(nx.pagerank(G, alpha=0.85).values()), dtype=float)

    deg_order0 = break_ties_randomly_desc(deg_vals, rng)
    bc_order0 = break_ties_randomly_desc(bc_vals, rng)
    pr_order0 = break_ties_randomly_desc(pr_vals, rng)

    # Percolation points (mean over tie-break resamples for degree/bc/pr; single for PiP)
    def mean_pp(vals: np.ndarray) -> int:
        pps = np.zeros(int(args.n_iter), dtype=float)
        for i in range(int(args.n_iter)):
            ord0 = break_ties_randomly_desc(vals, rng)
            pps[i] = percolation_point(adj, ord0)
        return int(np.round(pps.mean()))

    matlab_perc: dict[str, int] | None = None
    if args.perc_csv:
        rows = (Path(args.perc_csv)).read_text(encoding="utf-8").strip().splitlines()
        # expect header: metric,percolation_point
        matlab_perc = {}
        for line in rows[1:]:
            if not line.strip():
                continue
            m, v = line.split(",", 1)
            matlab_perc[m.strip()] = int(round(float(v)))
        missing = [k for k in ["Degree", "Betweenness", "PageRank", "PiP"] if k not in matlab_perc]
        if missing:
            raise ValueError(f"--perc-csv missing metrics: {missing}")

    perc_deg = matlab_perc["Degree"] if matlab_perc else mean_pp(deg_vals)
    perc_bc = matlab_perc["Betweenness"] if matlab_perc else mean_pp(bc_vals)
    perc_pr = matlab_perc["PageRank"] if matlab_perc else mean_pp(pr_vals)
    perc_pip = matlab_perc["PiP"] if matlab_perc else percolation_point(adj, pip_order0)

    results = [
        StrategyResult("Degree", deg_order0, perc_deg),
        StrategyResult("Betweenness", bc_order0, perc_bc),
        StrategyResult("PageRank", pr_order0, perc_pr),
        StrategyResult("PiP", pip_order0, perc_pip),
    ]

    # Save bar plot (percolation points from MATLAB if --perc-csv provided)
    save_barplot(out_dir / "avg_percolation_point_bar.png", results)

    # Node-set comparison + brain plots: PURE PiP-order vs metric orders.
    # Use metric-specific n taken from the MATLAB percolation-point CSV (or --top-n fallback).
    results_by_name = {r.name: r for r in results}
    if matlab_perc is not None and args.top_n is None:
        n_for_plot = {
            "Degree": min(int(matlab_perc["Degree"]), n_nodes),
            "Betweenness": min(int(matlab_perc["Betweenness"]), n_nodes),
            "PageRank": min(int(matlab_perc["PageRank"]), n_nodes),
            "PiP": min(int(matlab_perc["PiP"]), n_nodes),
        }
    else:
        n0 = int(args.top_n) if args.top_n is not None else 10
        n0 = min(max(n0, 1), n_nodes)
        n_for_plot = {k: n0 for k in ["Degree", "Betweenness", "PageRank", "PiP"]}

    top = {name: results_by_name[name].order0[: n_for_plot[name]] for name in n_for_plot.keys()}

    # Jaccard between sets of DIFFERENT sizes is well-defined.
    # We compare each metric's top-n_metric against PiP's top-n_pip.
    jac = {
        "Degree": jaccard(top["PiP"].tolist(), top["Degree"].tolist()),
        "Betweenness": jaccard(top["PiP"].tolist(), top["Betweenness"].tolist()),
        "PageRank": jaccard(top["PiP"].tolist(), top["PageRank"].tolist()),
    }

    save_jaccard_barplot(out_dir / "avg_jaccard_bar.png", jac)

    # Venn diagrams (PiP vs each metric)
    try:
        save_venn(
            out_dir / "venn_pip_vs_degree.png",
            top["PiP"].tolist(),
            top["Degree"].tolist(),
            label_a=f"PiP (n={len(top['PiP'])})",
            label_b=f"Degree (n={len(top['Degree'])})",
            title=f"PiP vs Degree  (J={jac['Degree']:.3f})",
        )
        save_venn(
            out_dir / "venn_pip_vs_betweenness.png",
            top["PiP"].tolist(),
            top["Betweenness"].tolist(),
            label_a=f"PiP (n={len(top['PiP'])})",
            label_b=f"Betweenness (n={len(top['Betweenness'])})",
            title=f"PiP vs Betweenness  (J={jac['Betweenness']:.3f})",
        )
        save_venn(
            out_dir / "venn_pip_vs_pagerank.png",
            top["PiP"].tolist(),
            top["PageRank"].tolist(),
            label_a=f"PiP (n={len(top['PiP'])})",
            label_b=f"PageRank (n={len(top['PageRank'])})",
            title=f"PiP vs PageRank  (J={jac['PageRank']:.3f})",
        )
    except RuntimeError:
        # matplotlib-venn not installed: skip silently (Jaccard text file still written)
        pass

    (out_dir / "jaccard_topn.txt").write_text(
        "\n".join(
            [
                "reference_sets=PiP_order vs metric_order",
                "n_used_for_plots: " + ", ".join([f"{k}={len(v)}" for k, v in top.items()]),
                f"Jaccard(PiP_top_{len(top['PiP'])}, Degree_top_{len(top['Degree'])})={jac['Degree']:.6f}",
                f"Jaccard(PiP_top_{len(top['PiP'])}, Betweenness_top_{len(top['Betweenness'])})={jac['Betweenness']:.6f}",
                f"Jaccard(PiP_top_{len(top['PiP'])}, PageRank_top_{len(top['PageRank'])})={jac['PageRank']:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Brain plots (metric-specific n)
    pal = paper_palette()
    for name in ["PiP", "Degree", "Betweenness", "PageRank"]:
        n_here = len(top[name])
        save_brain_plot(
            out_dir / f"brain_top{n_here:02d}_{name}.png",
            coords=coords,
            top_nodes0=top[name],
            title=f"{name}: top-{n_here} nodes (average graph)",
            node_color=pal[name] if name == "PiP" else "#3a89dc",
        )

    # Overlap plots: PiP (red) vs metric (blue), shared (black)
    save_overlap_brain_plot(
        out_dir / "brain_overlap_pip_vs_degree.png",
        coords=coords,
        pip_nodes0=top["PiP"],
        metric_nodes0=top["Degree"],
        title="PiP vs Degree (red=PiP, blue=Degree, black=shared)",
        pip_color=pal["PiP"],
        metric_color="#3a89dc",
    )
    write_brainnet_node(
        out_dir / "brainnet_overlap_pip_vs_degree.node",
        coords=coords,
        pip_nodes0=top["PiP"],
        metric_nodes0=top["Degree"],
        metric_name="Degree",
    )
    save_overlap_brain_plot(
        out_dir / "brain_overlap_pip_vs_betweenness.png",
        coords=coords,
        pip_nodes0=top["PiP"],
        metric_nodes0=top["Betweenness"],
        title="PiP vs Betweenness (red=PiP, blue=Betweenness, black=shared)",
        pip_color=pal["PiP"],
        metric_color="#3a89dc",
    )
    write_brainnet_node(
        out_dir / "brainnet_overlap_pip_vs_betweenness.node",
        coords=coords,
        pip_nodes0=top["PiP"],
        metric_nodes0=top["Betweenness"],
        metric_name="Betweenness",
    )
    save_overlap_brain_plot(
        out_dir / "brain_overlap_pip_vs_pagerank.png",
        coords=coords,
        pip_nodes0=top["PiP"],
        metric_nodes0=top["PageRank"],
        title="PiP vs PageRank (red=PiP, blue=PageRank, black=shared)",
        pip_color=pal["PiP"],
        metric_color="#3a89dc",
    )
    write_brainnet_node(
        out_dir / "brainnet_overlap_pip_vs_pagerank.node",
        coords=coords,
        pip_nodes0=top["PiP"],
        metric_nodes0=top["PageRank"],
        metric_name="PageRank",
    )

    # PiP order gradient plot (all nodes colored by PiP rank)
    save_pip_order_gradient_plot(
        out_dir / "brain_pip_order_gradient.png",
        coords=coords,
        pip_order0=pip_order0,
        title="PiP order on average graph (earlier = darker red)",
    )

    # Optional Jaccard sweep over n=1..N
    if args.jaccard_sweep:
        ns = np.arange(1, n_nodes + 1)
        jac_deg = np.zeros_like(ns, dtype=float)
        jac_bc = np.zeros_like(ns, dtype=float)
        jac_pr = np.zeros_like(ns, dtype=float)
        for i, n in enumerate(ns):
            pip_i = set(pip_order0[:n].tolist())
            jac_deg[i] = jaccard(pip_i, deg_order0[:n].tolist())
            jac_bc[i] = jaccard(pip_i, bc_order0[:n].tolist())
            jac_pr[i] = jaccard(pip_i, pr_order0[:n].tolist())

        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
        ax.plot(ns, jac_deg, label="PiP vs Degree", lw=2)
        ax.plot(ns, jac_bc, label="PiP vs Betweenness", lw=2)
        ax.plot(ns, jac_pr, label="PiP vs PageRank", lw=2)
        ax.set_xlabel("Top-n")
        ax.set_ylabel("Jaccard similarity")
        ax.set_title("Node-set overlap vs top-n (average graph)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "jaccard_sweep.png", bbox_inches="tight")
        plt.close(fig)

        np.savetxt(
            out_dir / "jaccard_sweep.csv",
            np.column_stack([ns, jac_deg, jac_bc, jac_pr]),
            delimiter=",",
            header="top_n,jaccard_pip_degree,jaccard_pip_betweenness,jaccard_pip_pagerank",
            comments="",
        )

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()

