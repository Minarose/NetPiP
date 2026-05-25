"""netpip - Participation in Percolation (PiP) for network hub identification.

Public API (stable):

    Validation
    ----------
    validate_adjacency
    AdjacencyReport
    AdjacencyValidationError

    Core engine
    -----------
    run_pip
    PiPResult

    Convergence
    -----------
    wilson_half_width
    plateau_reached

    Ranking
    -------
    tilted_peak_rank
    pip_top_n_at_percolation_point
    percolation_point

    Hub clustering
    --------------
    ward_silhouette_cluster
    pip_hub_cluster

    Benchmarking against classical centralities
    -------------------------------------------
    degree_attack_order
    betweenness_attack_order
    pagerank_attack_order
    metric_top_n_at_percolation_point
    jaccard
"""

from __future__ import annotations

from netpip._version import __version__
from netpip.validation import (
    AdjacencyReport,
    AdjacencyValidationError,
    validate_adjacency,
)
from netpip.convergence import plateau_reached, wilson_half_width
from netpip.core import PiPResult, run_pip
from netpip.ranking import (
    percolation_point,
    pip_top_n_at_percolation_point,
    tilted_peak_rank,
)
from netpip.clustering import pip_hub_cluster, ward_silhouette_cluster
from netpip.metrics import (
    betweenness_attack_order,
    degree_attack_order,
    jaccard,
    metric_top_n_at_percolation_point,
    pagerank_attack_order,
)

__all__ = [
    "__version__",
    # validation
    "AdjacencyReport",
    "AdjacencyValidationError",
    "validate_adjacency",
    # convergence
    "plateau_reached",
    "wilson_half_width",
    # core
    "PiPResult",
    "run_pip",
    # ranking
    "percolation_point",
    "pip_top_n_at_percolation_point",
    "tilted_peak_rank",
    # clustering
    "pip_hub_cluster",
    "ward_silhouette_cluster",
    # metrics
    "betweenness_attack_order",
    "degree_attack_order",
    "jaccard",
    "metric_top_n_at_percolation_point",
    "pagerank_attack_order",
]
