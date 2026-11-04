"""Run the full PiP workflow on your own network and save tables and figures.

Input: a binary, undirected adjacency matrix (.npy, .npz, .csv, .txt, .mat)
or a two-column edge list (--edge-list). For weighted matrices, pass
--threshold (keep weights above it) or --density (keep the strongest edges).

Examples
--------
python examples/analyze_network.py examples/example_data/karate_adjacency.csv
python examples/analyze_network.py my_edges.csv --edge-list --n-nodes 120 --attacks 1e7 --seed 7
python examples/analyze_network.py examples/example_data/weighted_correlation_matrix.csv --density 0.08

Outputs in --out (default: pip_output/<input name>/):
  pip_result.npz          counts, hits and checkpoints (reload with netpip.PiPResult.load)
  node_scores.csv         linear and exponential PiP score and rank of every node
  rms_convergence.csv     RMS change between checkpoints (paper threshold 0.005)
  attack_comparison.csv   removals to the percolation point for PiP vs centralities
  pip_surface.png         exponential-weighted PiP surface (heat map)
  attack_curves.png       LCC and SLCC during each targeted attack
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import netpip


def load(args):
    path = Path(args.matrix)
    if args.edge_list:
        return netpip.load_edge_list(path, n_nodes=args.n_nodes, one_based=not args.zero_based)
    if args.threshold is None and args.density is None:
        return netpip.load_adjacency(path)
    W = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None) if path.suffix in (".csv", ".txt") \
        else np.load(path)
    W = np.abs(W) if args.absolute else W
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, -np.inf)
    if args.density is not None:  # keep the strongest edges up to the requested density
        iu = np.triu_indices(len(W), 1)
        k = int(round(args.density * len(iu[0])))
        threshold = np.sort(W[iu])[::-1][k - 1] if k > 0 else np.inf
        print(f"Density {args.density:.3f}: keeping weights >= {threshold:.4g}")
        A = (W >= threshold).astype(np.uint8)
        np.fill_diagonal(A, 0)
        return netpip.validate_adjacency(np.maximum(A, A.T))
    return netpip.binarize(W, args.threshold)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("matrix")
    p.add_argument("--edge-list", action="store_true", help="input is a two-column edge list")
    p.add_argument("--n-nodes", type=int, help="number of nodes (edge lists; keeps isolated nodes)")
    p.add_argument("--zero-based", action="store_true", help="edge-list node labels start at 0")
    p.add_argument("--threshold", type=float, help="binarize a weighted matrix: keep weights > threshold")
    p.add_argument("--density", type=float, help="binarize a weighted matrix: keep this fraction of strongest edges")
    p.add_argument("--absolute", action="store_true", help="threshold |weights|")
    p.add_argument("--attacks", type=float, default=1e6, help="number of random attacks (default 1e6)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    A = load(args)
    n, e = len(A), int(A.sum() // 2)
    out = args.out or Path("pip_output") / Path(args.matrix).stem
    out.mkdir(parents=True, exist_ok=True)
    print(f"Network: {n} nodes, {e} edges, density {2 * e / (n * (n - 1)):.4f}")

    budget = int(args.attacks)
    marks = [int(m) for m in np.unique(np.round(np.geomspace(1e4, budget, 8), -3)) if 1e3 <= m < budget]
    result = netpip.run_pip(A, budget, seed=args.seed, checkpoints=marks, n_threads=args.threads, progress=True)
    result.save(out / "pip_result.npz")

    lin, exp = result.scores("linear"), result.scores("exponential")
    rank = np.empty(n, int)
    rank[result.ranking("linear")] = np.arange(1, n + 1)
    with (out / "node_scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "rank", "pip_linear", "pip_exponential"])
        for i in np.argsort(rank):
            w.writerow([i + 1, rank[i], f"{lin[i]:.6g}", f"{exp[i]:.6g}"])

    rms = result.rms_convergence()
    with (out / "rms_convergence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rms[0]))
        w.writeheader()
        w.writerows(rms)
    stable = netpip.first_stable_checkpoint(rms, 0.005)
    print("RMS change <= 0.005 first at:", f"{stable:,} attacks" if stable else "not reached; sample more")

    orders = {"PiP": netpip.attack_orders(lin)}
    orders.update({k: netpip.attack_orders(s) for k, s in netpip.centrality_scores(A).items()})
    csr = netpip.to_csr(A)
    rows = []
    for name, o in orders.items():
        s = netpip.evaluate_attacks(csr, o)
        rows.append(dict(method=name, removals_to_percolation_point=s["peak_removals"],
                         sd_over_tie_orders=s["peak_removals_sd"], lcc_area=s["lcc_area"],
                         removals_to_10pct_lcc=s["removals_to_10pct"], tie_orders=s["n_orders"]))
        print(f"  {name:12s} percolation point after {s['peak_removals']} removals")
    with (out / "attack_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return
    steps, W = result.surface("exponential")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(W.T, aspect="auto", origin="lower", cmap="gist_heat_r",
                   extent=(steps[0] - 0.5, steps[-1] + 0.5, 0.5, n + 0.5))
    fig.colorbar(im, ax=ax, label="PiP (exponential weighting)")
    ax.set(xlabel="Removal step", ylabel="Node", title="PiP surface")
    fig.tight_layout()
    fig.savefig(out / "pip_surface.png", dpi=200)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for name, o in orders.items():
        lcc, slcc = netpip.component_trace(csr, o[0])
        q = np.arange(n + 1) / n
        axes[0].plot(q, lcc / n, label=name)
        axes[1].plot(q, slcc / n, label=name)
    axes[0].set(xlabel="Fraction removed", ylabel="LCC / N")
    axes[1].set(xlabel="Fraction removed", ylabel="SLCC / N")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "attack_curves.png", dpi=200)
    print(f"Saved results to {out}/")


if __name__ == "__main__":
    main()
