# Changelog

## 1.0.0 (2026)

New release accompanying "Participation in Percolation (PiP): a probabilistic
metric for identifying critical nodes in complex networks".

* **Method.** Fixed-budget sampling with checkpoints and an RMS convergence
  check (threshold 0.005) replaces the Wilson-interval plateau stopping rule.
  Linear step weighting for scores, rankings and clustering; exponential
  weighting for visualisation. The analysis keeps the longest contiguous block
  of observed steps (or all observed steps, optionally).
* **Python toolbox (`netpip`).** Exact union–find percolation kernel compiled
  with Numba (pure-Python fallback); `run_pip`, `PiPResult`, `rms_convergence`,
  `attack_orders`/`evaluate_attacks` for comparisons with degree, betweenness and
  PageRank, `critical_cluster`, and loaders for matrices, edge lists and
  weighted data. Package moved to `src/netpip` with a top-level `pyproject.toml`.
* **MATLAB toolbox** rewritten to the same definitions, with a test against the
  Python reference outputs.
* **Paper data and results.** Six networks (hospital contact, global airport,
  yeast DNA-repair protein, brain language, Watts–Strogatz, Erdős–Rényi) with a
  script that rebuilds them from their sources, and the saved 1-billion-attack
  counts, checkpoints and attack orders.
* **Reproduction.** `reproduce/run_all.py` regenerates Figs 3–5, Supplementary
  Figs 1–3 and Table 1 and checks every key number; `reproduce/resample.py`
  replays the sampling with the paper's seeds.
* Removed the scripts, results and figures of the earlier language-only
  analysis (`scripts/`, `results/`, `figures/`); they remain in the history of
  the `main` branch. The MEG input matrices moved to `data/raw/language_meg/`.

## 0.1.0 (2025)

First release: PiP applied to the MEG expressive language network, with
Wilson-interval convergence and tilted-peak ranking.
