# Saved results: 1 billion attacks per network

These files let every figure and number in the paper be regenerated in minutes
(`python reproduce/run_all.py`) without rerunning the sampling.

Per network (`hospital`, `airport`, `yeast`, `language`, `watts_strogatz`, `erdos_renyi`):

| File | Contents |
|---|---|
| `counts.npz` | `counts` (N) and `hits` (N × N) after 1,000,000,000 attacks; load with `netpip.PiPResult.load()` |
| `snapshots/<attacks>.npz` | The same at each checkpoint used for the RMS analysis (Supplementary Fig. 2) |
| `attack_orders.npz` | 1-based removal orders behind Fig. 4: `Linear_PiP` (1 order) and `Degree`, `Betweenness`, `PageRank` (500 tie-breaking orders each, scores rounded to 12 decimals) |
| `run.json` | Seed, batch size, threads, run time and the SHA-256 fingerprint of the adjacency matrix |

`counts[k-1]` is the number of attacks whose percolation point was at removal
step k; `hits[k-1, i]` is how many of those had removed node i by step k.

Notes

* **Airport.** The counts are the 1-billion checkpoint of a 3-billion-attack run
  with the same seed. Its checkpoint matrices (≈ 70 MB) are too large for git, so
  `airport/rms.csv` holds the RMS values computed from them; `resample.py` can
  regenerate them.
* **Yeast.** The 1-million checkpoint comes from a separate pilot run and serves
  only as the first RMS checkpoint, as in the paper.
* **Centrality attack orders.** Degree, betweenness and PageRank orders were
  computed with the Brain Connectivity Toolbox and random tie-breaking; storing
  them makes Fig. 4 exact. `reproduce/key_numbers.py --fresh-centralities`
  recomputes them from scratch (bctpy); tie-breaking can then shift the degree
  means slightly.
* `reference_attack_metrics.csv` and `reference_rms.csv` are the values reported
  in the paper; `reproduce/key_numbers.py` checks its results against them.
