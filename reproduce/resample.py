"""Rerun PiP sampling from scratch with the paper's seeds.

Because the random stream is fixed by the seed, a rerun reproduces the paper's
counts exactly. When the chosen budget equals a saved checkpoint (1M, 10M, 100M,
... or the full 1B), the new counts are compared with the saved ones.

Examples
--------
python reproduce/resample.py                         # all six networks, 1 million attacks each (~1 min)
python reproduce/resample.py --network language --attacks 25_000_000
python reproduce/resample.py --full                  # 1 billion attacks per network (hours; airport ~1 day)

Results are written to reproduce/output/resampled/<network>.npz (netpip.PiPResult format).
"""
import argparse
import time

import numpy as np

import _paper as P
from _paper import netpip


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", choices=list(P.NETWORKS), nargs="*", default=list(P.NETWORKS))
    p.add_argument("--attacks", type=lambda s: int(float(s.replace("_", ""))), default=1_000_000)
    p.add_argument("--full", action="store_true", help="use the paper's budget of 1 billion attacks")
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args()
    budget = 1_000_000_000 if args.full else args.attacks
    out = P.OUT / "resampled"
    out.mkdir(parents=True, exist_ok=True)
    for key in args.network:
        info = P.run_info(key)
        A = P.adjacency(key)
        marks = [m for m in (1e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8, 5e8, 7.5e8) if m < budget]
        print(f"{key}: {budget:,} attacks, seed {info['seed']}", flush=True)
        t = time.perf_counter()
        res = netpip.run_pip(A, budget, seed=info["seed"], batch_size=info["batch"], checkpoints=marks,
                             n_threads=args.threads, progress=budget >= 1e8)
        print(f"  {time.perf_counter() - t:.1f} s", flush=True)
        res.save(out / f"{key}.npz")
        saved = {}
        if budget == 1_000_000_000:
            ref = P.result(key)
            saved[budget] = (ref.counts, ref.hits)
        snap = P.RESULTS / key / "snapshots" / f"{budget}.npz"
        if snap.exists():
            with np.load(snap) as z:
                saved[budget] = (z["counts"], z["hits"])
        if key == "yeast" and budget == 1_000_000:
            saved.pop(budget, None)  # the paper's 1M yeast snapshot came from a separate pilot run
        if budget in saved:
            same = np.array_equal(res.counts, saved[budget][0]) and np.array_equal(res.hits, saved[budget][1])
            print(f"  identical to the paper's saved counts at {budget:,} attacks: {same}")
            if not same:
                raise SystemExit("mismatch with saved counts")
        top = res.ranking("linear", rule=P.NETWORKS[key]["rule"], basis=P.NETWORKS[key]["basis"])[:5] + 1
        print(f"  top-5 nodes by linear PiP: {top.tolist()}")


if __name__ == "__main__":
    main()
