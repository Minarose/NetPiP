"""Reproduce every number and figure of the paper from the saved 1-billion-attack counts.

Run from the repository root:
    python reproduce/run_all.py              # under a minute, outputs in reproduce/output/
    python reproduce/run_all.py --dpi 1200   # print-resolution figures

To rerun the sampling itself, see reproduce/resample.py.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ("Key numbers (checked against the paper)", "key_numbers.py", False),
    ("Figure 3: PiP surfaces", "fig3_surfaces.py", True),
    ("Figure 4: PiP vs centrality attacks", "fig4_attacks.py", True),
    ("Figure 5: language network critical nodes", "fig5_language_map.py", True),
    ("Supplementary Figure 1: adjacency matrices", "supp_fig1_adjacency.py", True),
    ("Supplementary Figure 2: RMS convergence", "supp_fig2_rms.py", True),
    ("Supplementary Figure 3 and Table 1: LCC/SLCC", "supp_fig3_lcc.py", True),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    start = time.perf_counter()
    for title, script, takes_dpi in STEPS:
        print(f"\n== {title}", flush=True)
        cmd = [sys.executable, str(HERE / script)] + (["--dpi", str(args.dpi)] if takes_dpi else [])
        subprocess.run(cmd, check=True, cwd=HERE)
    print(f"\nDone in {time.perf_counter() - start:.0f} s. Outputs: reproduce/output/")


if __name__ == "__main__":
    main()
