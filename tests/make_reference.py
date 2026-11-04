"""Write Python reference outputs used by matlab/tests/test_against_python.m.

MATLAB and NumPy use different random number generators, so the MATLAB test
feeds both toolboxes the same fixed removal orders and the same saved counts,
and checks that every downstream quantity agrees.

Run:  python tests/make_reference.py
"""
import json
from pathlib import Path

import numpy as np

import netpip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "reference"


def main():
    OUT.mkdir(exist_ok=True)
    A = netpip.load_edge_list(ROOT / "examples/example_data/karate_edges.csv", n_nodes=34)
    orders = netpip.random_orders(np.random.default_rng(2024), 34, 2000)
    events = netpip._kernels.batch_event_steps(orders, *netpip.to_csr(A))
    counts = np.zeros(34, np.uint64)
    hits = np.zeros((34, 34), np.uint64)
    netpip._kernels.accumulate(orders, events, counts, hits)
    np.savetxt(OUT / "karate_orders.csv", orders + 1, fmt="%d", delimiter=",")
    np.savetxt(OUT / "karate_events.csv", events, fmt="%d", delimiter=",")
    np.savetxt(OUT / "karate_counts.csv", counts, fmt="%d", delimiter=",")
    np.savetxt(OUT / "karate_hits.csv", hits, fmt="%d", delimiter=",")
    trace_lcc, trace_slcc = netpip.component_trace(A, orders[0])

    with np.load(ROOT / "results/paper_1B/language/counts.npz") as z:
        lc, lh = z["counts"], z["hits"]
    np.savetxt(OUT / "language_counts.csv", lc, fmt="%d", delimiter=",")
    np.savetxt(OUT / "language_hits.csv", lh, fmt="%d", delimiter=",")
    snaps = {}
    for p in sorted((ROOT / "results/paper_1B/language/snapshots").glob("*.npz")):
        with np.load(p) as z:
            snaps[int(p.stem)] = (z["counts"], z["hits"])
            np.savetxt(OUT / f"language_snapshot_{p.stem}_counts.csv", z["counts"], fmt="%d", delimiter=",")
            np.savetxt(OUT / f"language_snapshot_{p.stem}_hits.csv", z["hits"], fmt="%d", delimiter=",")
    LA = netpip.load_edge_list(ROOT / "data/networks/language/edges.csv", n_nodes=66)
    lin = netpip.pip_scores(lc, lh, "linear")
    order = netpip.attack_orders(lin)[0]
    steps, W = netpip.weighted_surface(lc, lh, "linear")
    cl = netpip.critical_cluster(W.T)
    rms = netpip.rms_convergence(snaps)
    hosp_w = netpip.step_weights([1, 3, 4, 69], 75, "exponential", "absolute")

    ref = dict(
        karate=dict(trace_order1_lcc=trace_lcc.tolist(), trace_order1_slcc=trace_slcc.tolist()),
        language=dict(retained_steps=steps.tolist(), linear_scores=lin.tolist(),
                      exponential_scores=netpip.pip_scores(lc, lh, "exponential").tolist(),
                      pip_order=(order + 1).tolist(),
                      pip_attack=netpip.attack_outcome(LA, order),
                      critical_nodes=(cl["critical_nodes"] + 1).tolist(), k=cl["k"], silhouette=cl["silhouette"],
                      rms=[dict(attacks=r["attacks"], rms=r["rms"]) for r in rms]),
        hospital_exponential_absolute_weights=dict(steps=[1, 3, 4, 69], n=75, weights=hosp_w.tolist()),
    )
    (OUT / "expected.json").write_text(json.dumps(ref, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
