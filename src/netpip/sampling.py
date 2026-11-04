"""Monte Carlo sampling of random attacks: the core of PiP."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _kernels
from .io import to_csr, validate_adjacency
from .surface import pip_scores, retained_steps, weighted_surface

__all__ = ["PiPResult", "run_pip", "random_orders"]


def random_orders(rng: np.random.Generator, n_nodes: int, n_orders: int) -> np.ndarray:
    """Uniformly random removal orders, drawn exactly as in the paper.

    The stream does not depend on how attacks are split into batches, so a run
    with the same seed reproduces the same attacks for any batch size.
    """
    return rng.permuted(np.tile(np.arange(n_nodes), (n_orders, 1)), axis=1)


@dataclass
class PiPResult:
    """Sampled PiP counts for one network.

    counts[k-1]   attacks whose percolation event occurred at removal step k
    hits[k-1, i]  of those, attacks in which node i had been removed by step k
    snapshots     {n_attacks: (counts, hits)} saved at checkpoints during sampling
    """
    counts: np.ndarray
    hits: np.ndarray
    n_attacks: int
    seed: int | None = None
    snapshots: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.counts)

    def surface(self, kind: str | None = "linear", rule: str = "longest_block", basis: str = "retained"):
        """(steps, W): retained 1-based steps and the weighted PiP surface (steps x nodes)."""
        return weighted_surface(self.counts, self.hits, kind, rule, basis)

    def scores(self, kind: str = "linear", rule: str = "longest_block", basis: str = "retained"):
        """Node PiP scores (mean weighted participation); rank nodes by descending score."""
        return pip_scores(self.counts, self.hits, kind, rule, basis)

    def ranking(self, kind: str = "linear", **kwargs) -> np.ndarray:
        """0-based node indices from most to least critical (ties keep node order)."""
        return np.argsort(-np.round(self.scores(kind, **kwargs), 12), kind="stable")

    def checkpoints(self) -> dict:
        """All saved checkpoints plus the final counts: {n_attacks: (counts, hits)}."""
        out = dict(self.snapshots)
        out[self.n_attacks] = (self.counts, self.hits)
        return dict(sorted(out.items()))

    def rms_convergence(self, **kwargs):
        """RMS change between successive checkpoints; see :func:`netpip.rms_convergence`."""
        from .convergence import rms_convergence

        return rms_convergence(self.checkpoints(), **kwargs)

    def retained_steps(self, rule: str = "longest_block"):
        return retained_steps(self.counts, rule)

    def save(self, path) -> None:
        """Save counts, hits, snapshots and run information to one .npz file."""
        arrays = dict(counts=self.counts, hits=self.hits)
        for attacks, (c, h) in self.snapshots.items():
            arrays[f"snapshot_{attacks}_counts"] = c
            arrays[f"snapshot_{attacks}_hits"] = h
        info = dict(self.info, n_attacks=self.n_attacks, seed=self.seed)
        np.savez_compressed(path, info=json.dumps(info), **arrays)

    @classmethod
    def load(cls, path) -> "PiPResult":
        """Load a result saved with :meth:`save`, or a plain npz with `counts` and `hits`."""
        with np.load(path) as z:
            counts, hits = z["counts"].astype(np.uint64), z["hits"].astype(np.uint64)
            info = json.loads(str(z["info"])) if "info" in z.files else {}
            snaps = {}
            for name in z.files:
                if name.startswith("snapshot_") and name.endswith("_counts"):
                    attacks = int(name.split("_")[1])
                    snaps[attacks] = (z[name], z[f"snapshot_{attacks}_hits"])
        n_attacks = int(info.pop("n_attacks", counts.sum()))
        seed = info.pop("seed", None)
        return cls(counts, hits, n_attacks, seed, dict(sorted(snaps.items())), info)


def run_pip(adjacency, n_attacks: int = 1_000_000, *, seed: int | None = None,
            batch_size: int = 10_000, checkpoints=None, n_threads: int | None = None,
            progress: bool = False) -> PiPResult:
    """Sample `n_attacks` uniformly random attacks and record percolation participation.

    Parameters
    ----------
    adjacency : (N, N) binary symmetric array with zero diagonal.
    n_attacks : number of random removal orders to sample (the paper used 1e9).
    seed : seed for numpy.random.default_rng; use the same seed to reproduce a run.
    batch_size : orders drawn per batch (memory vs. overhead only; results do not change).
    checkpoints : attack totals at which to keep a copy of (counts, hits), e.g.
        [1e6, 1e7, 1e8]. Use these with :func:`netpip.rms_convergence`.
    n_threads : Numba threads (default: all cores).
    progress : print progress at each checkpoint / every 10 batches.
    """
    A = validate_adjacency(adjacency)
    n_attacks = int(n_attacks)
    if n_attacks < 1:
        raise ValueError("n_attacks must be positive")
    n = len(A)
    indptr, indices = to_csr(A)
    if n_threads is not None and _kernels.HAVE_NUMBA:
        import numba

        numba.set_num_threads(int(n_threads))
    rng = np.random.default_rng(seed)
    counts = np.zeros(n, dtype=np.uint64)
    hits = np.zeros((n, n), dtype=np.uint64)
    marks = sorted({int(c) for c in (checkpoints or []) if 0 < int(c) <= n_attacks})
    snapshots = {}
    done, batches, start = 0, 0, time.perf_counter()
    while done < n_attacks:
        next_mark = next((m for m in marks if m > done), n_attacks)
        b = min(int(batch_size), next_mark - done)
        orders = random_orders(rng, n, b)
        steps = _kernels.batch_event_steps(orders, indptr, indices)
        _kernels.accumulate(orders, steps, counts, hits)
        done += b
        batches += 1
        if done in marks:
            snapshots[done] = (counts.copy(), hits.copy())
        if progress and (done in marks or done == n_attacks or batches % 10 == 0):
            rate = done / max(time.perf_counter() - start, 1e-9)
            print(f"  {done:,}/{n_attacks:,} attacks ({rate:,.0f}/s)", flush=True)
    info = dict(adjacency_sha256=hashlib.sha256(A.tobytes()).hexdigest(), batch_size=int(batch_size),
                numba=_kernels.HAVE_NUMBA, elapsed_seconds=time.perf_counter() - start,
                threads=n_threads or os.cpu_count())
    return PiPResult(counts, hits, n_attacks, seed, snapshots, info)
