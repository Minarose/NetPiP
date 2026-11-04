# netpip for MATLAB

The same PiP definitions as the Python toolbox, as a `+netpip` package.

```matlab
addpath('path/to/NetPiP/matlab');
A = double(myAdjacency);                     % binary, symmetric, zero diagonal
result = netpip.run_pip(A, 1e6, 'Seed', 1, 'Checkpoints', [1e4 1e5]);
scores = netpip.pip_scores(result.counts, result.hits, 'linear');
[~, ranking] = sort(scores, 'descend');      % most critical nodes first
```

| Function | Purpose |
|---|---|
| `run_pip` | Sample random attacks; returns `counts`, `hits` and checkpoint snapshots (`'UseParfor', true` for parallel batches) |
| `accumulate_orders` | PiP counts for given removal orders |
| `percolation_event`, `slcc_trace` | Percolation point of one random attack |
| `retained_steps`, `step_weights`, `weighted_surface`, `pip_scores` | PiP surface and node scores |
| `rms_convergence` | RMS change between checkpoints (paper threshold 0.005) |
| `component_trace`, `percolation_point`, `attack_orders`, `evaluate_attacks` | Targeted attacks and their outcome |
| `critical_cluster` | Ward clustering of PiP profiles (needs the Statistics and Machine Learning Toolbox) |
| `validate_adjacency`, `neighbor_lists` | Input checks and helpers |

Requirements: MATLAB R2019b or newer; Statistics and Machine Learning Toolbox
for `critical_cluster` only; Parallel Computing Toolbox only for `'UseParfor'`.

## Differences from the Python toolbox

* **Random numbers.** MATLAB's generator differs from NumPy's, so a MATLAB run
  is statistically equivalent to, but not bit-identical with, the Python runs in
  the paper. To reproduce the paper's counts exactly, use Python
  (`reproduce/resample.py`) or load the saved counts in `results/paper_1B/`.
* **Speed.** The MATLAB sampler is plain MATLAB code and much slower than the
  Numba-compiled Python sampler. For budgets of 10^8 attacks or more, use Python
  or `'UseParfor', true` on many cores.

## Testing

```matlab
cd matlab
run tests/test_against_python.m
```

The test feeds both toolboxes the same fixed removal orders and the paper's
saved language-network counts (reference outputs in `../tests/reference/`,
written by `python tests/make_reference.py`) and checks events, counts,
component traces, scores, attack outcomes, clustering and RMS values.

> **Status:** the MATLAB code was written to match the Python reference and its
> core kernel logic was checked line by line, but it has not yet been run in
> MATLAB. Please run `tests/test_against_python.m` once and report any failure.

`examples/quickstart.m` runs PiP on the karate club network.
