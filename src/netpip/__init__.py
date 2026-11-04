"""Participation in Percolation (PiP).

A probabilistic metric for identifying the nodes that drive network fragmentation.
PiP samples random removal orders, finds where each attack reaches its
percolation point (first maximum of the second-largest connected component),
and estimates how often each node has been removed by then.

Quick start
-----------
>>> import netpip
>>> A = netpip.load_adjacency("my_network.npy")           # binary, symmetric
>>> result = netpip.run_pip(A, n_attacks=1_000_000, seed=1)
>>> scores = result.scores()                               # linear-weighted PiP
>>> top = result.ranking()[:10]                            # most critical nodes
"""
from ._version import __version__
from .attacks import (attack_orders, attack_outcome, centrality_scores, component_trace,
                      evaluate_attacks, percolation_point)
from .clustering import critical_cluster, silhouette_score
from .convergence import first_stable_checkpoint, rms_convergence
from .io import binarize, load_adjacency, load_edge_list, to_csr, validate_adjacency
from .sampling import PiPResult, random_orders, run_pip
from .surface import (observed_steps, pip_probability, pip_scores, retained_steps, step_weights,
                      weighted_surface)

__all__ = [
    "__version__", "run_pip", "PiPResult", "random_orders",
    "load_adjacency", "load_edge_list", "validate_adjacency", "binarize", "to_csr",
    "observed_steps", "retained_steps", "step_weights", "pip_probability", "weighted_surface",
    "pip_scores", "rms_convergence", "first_stable_checkpoint",
    "component_trace", "percolation_point", "attack_outcome", "attack_orders", "evaluate_attacks",
    "centrality_scores", "critical_cluster", "silhouette_score",
]
