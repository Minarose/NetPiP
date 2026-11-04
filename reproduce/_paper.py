"""Shared settings for reproducing the paper: networks, saved results and plotting style."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # use the repository's netpip without installing

import netpip  # noqa: E402

DATA = ROOT / "data" / "networks"
RESULTS = ROOT / "results" / "paper_1B"
OUT = ROOT / "reproduce" / "output"

# Figure 2 order. rule/basis follow the Methods: the hospital network keeps all
# observed steps (step 2 never occurs) and weights by the actual step k with N in place of S.
NETWORKS = {
    "hospital": dict(title="Hospital contact network", rule="all_observed", basis="absolute"),
    "airport": dict(title="Global airport network", rule="longest_block", basis="retained"),
    "yeast": dict(title="Yeast DNA-repair protein network", rule="longest_block", basis="retained"),
    "language": dict(title="Brain language network", rule="longest_block", basis="retained"),
    "watts_strogatz": dict(title="Watts–Strogatz network", rule="longest_block", basis="retained"),
    "erdos_renyi": dict(title="Erdős–Rényi random graph", rule="longest_block", basis="retained"),
}
METHODS = ["Linear PiP", "Degree", "Betweenness", "PageRank"]
COLORS = {"Linear PiP": "#df242b", "Degree": "#b84b08", "Betweenness": "#ed8a24", "PageRank": "#fac47b"}
LABELS = {"Linear PiP": "PiP", "Degree": "Degree", "Betweenness": "Betweenness", "PageRank": "PageRank"}
INK = "#262626"
MM = 1 / 25.4


def nodes(key):
    with (DATA / key / "nodes.csv").open() as f:
        return list(csv.DictReader(f))


def adjacency(key):
    return netpip.load_edge_list(DATA / key / "edges.csv", n_nodes=len(nodes(key)))


def run_info(key):
    return json.loads((RESULTS / key / "run.json").read_text())


def result(key) -> netpip.PiPResult:
    """Saved 1-billion-attack counts, with milestone snapshots where shipped."""
    res = netpip.PiPResult.load(RESULTS / key / "counts.npz")
    snaps = {}
    for path in sorted((RESULTS / key / "snapshots").glob("*.npz")):
        with np.load(path) as z:
            snaps[int(path.stem)] = (z["counts"], z["hits"])
    res.snapshots = dict(sorted(snaps.items()))
    return res


def surface(key, kind="linear"):
    cfg = NETWORKS[key]
    return result(key).surface(kind, rule=cfg["rule"], basis=cfg["basis"])


def saved_orders(key):
    """The exact 0-based attack orders behind Fig. 4 (tie orders for centralities)."""
    with np.load(RESULTS / key / "attack_orders.npz") as z:
        return {m: z[m.replace(" ", "_")] - 1 for m in METHODS}


def rms_rows(key):
    """RMS change between successive checkpoints (paper definition)."""
    if key == "airport":  # snapshots too large for git; values computed from them in the paper
        with (RESULTS / key / "rms.csv").open() as f:
            return [dict(attacks=int(r["attacks"]), previous_attacks=int(r["previous_attacks"]),
                         rms=float(r["rms"]), compared_rows=int(r["compared_rows"])) for r in csv.DictReader(f)]
    cfg = NETWORKS[key]
    return result(key).rms_convergence(rule=cfg["rule"], basis=cfg["basis"])


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    family = "Arial" if "Arial" in names else "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
                         "axes.labelcolor": INK})
    return plt


def save(fig, name, dpi):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", dpi=dpi)
    fig.savefig(OUT / f"{name}.png", dpi=dpi, facecolor="white")
    print(f"  wrote reproduce/output/{name}.pdf and .png")


def parser(doc):
    p = argparse.ArgumentParser(description=doc)
    p.add_argument("--dpi", type=int, default=300, help="raster resolution (paper figures used 600-1200)")
    return p
