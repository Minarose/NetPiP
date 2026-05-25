# netpip — MATLAB toolbox

A clean MATLAB API for **Participation in Percolation (PiP)**, mirroring the Python `netpip` package. This is the engine used to produce the manuscript results; the canonical Slurm/HPC driver lives in `scripts/2_pip_converge.m`, while this folder packages the same algorithm as a reusable, parameterized `+netpip` namespace you can drop into any MATLAB project.

## Requirements

- **MATLAB R2020b or newer** (tested on R2024b)
- **Graph and Network Algorithms** (`graph`, `conncomp` — base MATLAB)
- **Statistics and Machine Learning Toolbox** (only for `prctile`)
- **Parallel Computing Toolbox** *(optional)* — needed only if you pass `'UseParfor', true`
- **Brain Connectivity Toolbox** *(optional)* — only required for the benchmarking scripts in `../scripts/compare_pip_bct_percolation_*.m`

## Install

There is no `mltbx` install — this is a folder-based package. From your MATLAB session:

```matlab
addpath('/path/to/NetPiP/matlab');
% optionally make permanent:
% savepath
```

You can then call any function as `netpip.<name>`.

## Quickstart

```matlab
addpath('matlab');

A = double(your_binary_symmetric_adjacency);     % N x N
netpip.validate_adjacency(A, 'MinGiantFraction', 0.75);

res = netpip.run_pip(A, ...
    'MaxAttacks', 1e6, 'ChunkSize', 1e4, 'Seed', 42, ...
    'EnforceHW95', false, 'Verbose', true);

[order, peak_step, peak_amp] = netpip.tilted_peak_rank(res.node_P);
pp  = netpip.percolation_point(A, order);
hub = order(1:pp);                              % PiP hub set
```

A full runnable demo lives in [`examples/quickstart.m`](examples/quickstart.m).

## Input contract

`netpip.validate_adjacency` performs **read-only** checks on `A`. The matrix is never modified. `A` must already be:

- 2D, square, `N >= 2`
- finite (no `NaN` / `Inf`)
- binary (entries `{0, 1}`)
- symmetric (`A == A.'`)
- zero-diagonal (`diag(A) == 0`)
- sparse (density `< 1`, density `> 0`)
- with a largest connected component of at least `MinGiantFraction * N` nodes (default 0.5)

If any check fails, an `error('netpip:validation:<reason>', ...)` is raised.

## Public API

| Function | Purpose |
|---|---|
| `netpip.validate_adjacency(A, ...)` | Read-only input checks; returns descriptive `report` struct |
| `netpip.run_pip(A, ...)` | Monte Carlo PiP convergence engine |
| `netpip.wilson_half_width(part_counts, counts_per_step)` | Per-cell Wilson 95% half-widths |
| `netpip.plateau_reached(attacks_hist, hw95_hist, ...)` | Convergence plateau diagnostic |
| `netpip.tilted_peak_rank(node_P, ...)` | Time-tilted (`τ = S/6`) node ranking |
| `netpip.percolation_point(A, attack_order)` | 1-based step where 2nd component first peaks |

## How `node_P` is defined

```
counts_per_step(p)   = # attacks whose percolation step (1-based) equals p
part_counts(p, i)    = # attacks where (percolation step == p) AND
                       (node i was among the first p nodes removed)

node_P(p, i)         = part_counts(p, i) / counts_per_step(p)
                     = P(node i was removed by step p | percolation step == p)
```

This matches the `node_P` saved by `scripts/2_pip_converge.m` exactly (the underlying loop is the same; the wrapper just makes it parameterized and Slurm-free).

## Reproducing the paper results

The single-matrix MATLAB driver `scripts/5_graph_theory_avg.m` and the per-subject driver `scripts/5_graph_theory_per_subject.m` compute the BCT-based Degree / Betweenness / PageRank benchmarks reported in the manuscript. The HPC convergence wrapper `scripts/2_pip_converge.m` is the Slurm-aware production driver; `netpip.run_pip` is the same algorithm exposed as a reusable, interactive function.

## License

MIT. See `../LICENSE` at the repository root.
