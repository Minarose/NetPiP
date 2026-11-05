"""Recompute the paper's key numbers from the saved 1-billion-attack counts and check them.

Checks (and prints) for each network:
  * peak nodes of the exponential PiP surface (Results, Fig. 3)
  * first checkpoint with RMS change <= 0.005 (Supplementary Fig. 2)
  * removals to the percolation point for PiP-, degree-, betweenness- and
    PageRank-ordered attacks, and PiP's reduction vs the best centrality (Fig. 4)
  * LCC area and removals to 10% LCC (Supplementary Table 1)
  * language network critical cluster (Fig. 5)
Exits with an error if any value differs from results/reference_*.csv.

Run:  python reproduce/key_numbers.py            (a few seconds)
      python reproduce/key_numbers.py --fresh-centralities   also recompute centrality orders from scratch
"""
import csv
import hashlib
import sys

import numpy as np

import _paper as P
from _paper import netpip

PAPER_KEYS = {"hospital": "Hospital", "airport": "Airport", "yeast": "Yeast", "language": "Language",
              "watts_strogatz": "WS200", "erdos_renyi": "ER60"}
EXPECTED_PEAK_NODES = {"hospital": [1, 17, 20], "airport": [1862], "yeast": [153, 187], "language": [37, 49],
                       "watts_strogatz": [106, 117, 164], "erdos_renyi": [32]}
EXPECTED_CRITICAL_LANGUAGE = [21, 22, 28, 34, 35]


def peak_nodes(key):
    """Nodes whose highest exponential-weighted participation is at least half the surface maximum."""
    _, W = P.surface(key, "exponential")
    peak = W.max(axis=0)
    return sorted((np.flatnonzero(peak >= 0.5 * peak.max()) + 1).tolist())


def main():
    args = P.parser(__doc__)
    args.add_argument("--fresh-centralities", action="store_true",
                      help="also recompute degree/betweenness/PageRank and new tie orders (needs bctpy)")
    args = args.parse_args()
    ref = {(r["network"], r["method"]): r for r in csv.DictReader((P.RESULTS / "reference_attack_metrics.csv").open())}
    ref_rms = list(csv.DictReader((P.RESULTS / "reference_rms.csv").open()))
    problems, rows = [], []

    def check(label, got, want, tol=1e-9):
        ok = (got == want) if not isinstance(want, float) else abs(got - want) <= tol * max(1, abs(want))
        if not ok:
            problems.append(f"{label}: got {got}, paper {want}")

    print(f"{'network':16s} {'peak nodes':18s} {'RMS<=0.005':>11s} {'PiP':>6s} {'Degree':>15s} {'Betw.':>7s} "
          f"{'PageRank':>8s} {'reduction':>9s}")
    for key, cfg in P.NETWORKS.items():
        A = P.adjacency(key)
        res = P.result(key)
        n = len(A)
        info = P.run_info(key)
        check(f"{key} adjacency", hashlib.sha256(A.tobytes()).hexdigest(), info["adjacency_sha256"])
        check(f"{key} attacks", res.n_attacks, 1_000_000_000)

        peaks = peak_nodes(key)
        check(f"{key} peak nodes", peaks, EXPECTED_PEAK_NODES[key])

        rms = P.rms_rows(key)
        first = netpip.first_stable_checkpoint(rms, 0.005, start=10_000_000)
        for row in rms:  # compare every plotted value with the paper's table
            match = [r for r in ref_rms if r["network"] == PAPER_KEYS[key] and int(r["attacks"]) == row["attacks"]]
            if match:
                check(f"{key} RMS at {row['attacks']}", row["rms"], float(match[0]["rms"]), tol=1e-9)

        # PiP order recomputed from the scores must equal the saved order.
        scores = res.scores("linear", rule=cfg["rule"], basis=cfg["basis"])
        fresh_pip = netpip.attack_orders(scores, seed=860024)
        saved = P.saved_orders(key)
        check(f"{key} PiP order", fresh_pip.tolist(), saved["Linear PiP"].tolist())

        stats = {}
        for method in P.METHODS:
            s = netpip.evaluate_attacks(netpip.io.to_csr(A), saved[method])
            stats[method] = s
            r = ref[(PAPER_KEYS[key], method)]
            check(f"{key} {method} peak", s["peak_removals"], float(r["peak_removals"]))
            check(f"{key} {method} LCC area", s["lcc_area"], float(r["lcc_area"]), tol=1e-9)
            check(f"{key} {method} removals to 10%", s["removals_to_10pct"], float(r["removals_to_10pct"]))
        best = min(stats[m]["peak_removals"] for m in P.METHODS[1:])
        reduction = 100 * (1 - stats["Linear PiP"]["peak_removals"] / best)
        rows.append(dict(network=key, nodes=n, peak_nodes=" ".join(map(str, peaks)),
                         first_rms_below_0_005=first, reduction_percent=round(reduction, 1),
                         **{f"{m}_removals": stats[m]["peak_removals"] for m in P.METHODS},
                         **{f"{m}_lcc_area": stats[m]["lcc_area"] for m in P.METHODS},
                         **{f"{m}_removals_to_10pct": stats[m]["removals_to_10pct"] for m in P.METHODS}))
        d = stats["Degree"]
        print(f"{key:16s} {' '.join(map(str, peaks)):18s} {first/1e6:>9.0f} M {stats['Linear PiP']['peak_removals']:>6.0f} "
              f"{d['peak_removals']:>7.2f} ± {d['peak_removals_sd']:<5.2f} {stats['Betweenness']['peak_removals']:>7.0f} "
              f"{stats['PageRank']['peak_removals']:>8.0f} {reduction:>8.1f}%")

        if args.fresh_centralities:
            for method, sc in netpip.centrality_scores(A).items():
                s = netpip.evaluate_attacks(netpip.io.to_csr(A), netpip.attack_orders(sc, seed=860024))
                print(f"    recomputed {method:12s} {s['peak_removals']:.2f} ± {s['peak_removals_sd']:.2f} "
                      f"(saved orders: {stats[method]['peak_removals']:.2f})")

    lang = P.surface("language", "linear")[1]
    cl = netpip.critical_cluster(lang.T)
    crit = (cl["critical_nodes"] + 1).tolist()
    check("language critical cluster", crit, EXPECTED_CRITICAL_LANGUAGE)
    check("language k", cl["k"], 3)
    print(f"\nLanguage clustering: k = {cl['k']} (mean silhouette {cl['silhouette']:.3f}); "
          f"critical nodes {crit}")
    lo = min(r["reduction_percent"] for r in rows)
    hi = max(r["reduction_percent"] for r in rows)
    print(f"PiP-ordered attacks needed {lo}–{hi}% fewer removals than the best centrality.")

    P.OUT.mkdir(parents=True, exist_ok=True)
    with (P.OUT / "key_numbers.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("  wrote reproduce/output/key_numbers.csv")
    if problems:
        print("\nMISMATCHES WITH THE PAPER:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("\nAll values match the paper.")


if __name__ == "__main__":
    main()
