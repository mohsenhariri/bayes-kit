"""
Rank Centrality: ranking from pairwise comparisons.

This module implements the Markov-chain estimator of Negahban, Oh, and Shah
(2017).

Notation
--------

Let :math:`R \\in \\{0,1\\}^{L \\times M \\times N}`. For each pair
:math:`(i,j)`, let :math:`W_{ij}` be decisive wins of :math:`i` over :math:`j`.
Define the pairwise empirical probabilities :math:`\\widehat{P}_{i\\succ j}`
from decisive outcomes (optionally tie adjusted).

Rank Centrality builds a row-stochastic transition matrix :math:`P` with
off-diagonal mass proportional to :math:`\\widehat{P}_{j\\succ i}` and ranks by
the stationary distribution :math:`\\pi`:

.. math::
    \\pi^\\top P = \\pi^\\top, \\qquad \\sum_i \\pi_i = 1.
"""

import numpy as np

from scorio.utils import rank_scores

from ._base import build_pairwise_counts, build_pairwise_wins, validate_input
from ._types import RankMethod, RankResult


def _is_connected_undirected(adj: np.ndarray) -> bool:
    """Check connectivity of an undirected graph given a boolean adjacency matrix."""
    n = adj.shape[0]
    if n == 0:
        return True

    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True

    while stack:
        i = stack.pop()
        neighbors = np.flatnonzero(adj[i] & ~seen)
        if neighbors.size:
            seen[neighbors] = True
            stack.extend(int(j) for j in neighbors)

    return bool(np.all(seen))


def _stationary_distribution_power(
    P: np.ndarray, max_iter: int = 10_000, tol: float = 1e-12
) -> np.ndarray:
    """Compute stationary distribution of a row-stochastic P via power iteration."""
    n = P.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    pi = np.ones(n, dtype=float) / n
    for _ in range(int(max_iter)):
        pi_new = P.T @ pi
        s = pi_new.sum()
        if s <= 0:
            return np.ones(n, dtype=float) / n
        pi_new = pi_new / s
        if np.linalg.norm(pi_new - pi, 1) < tol:
            return pi_new
        pi = pi_new

    return pi


def rank_centrality(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
    tie_handling: str = "half",
    smoothing: float = 0.0,
    teleport: float = 0.0,
    max_iter: int = 10_000,
    tol: float = 1e-12,
) -> RankResult:
    """
    Rank models with Rank Centrality.

    Method context:
        Build a row-stochastic random-walk matrix over models where transition
        probabilities prefer moving from a model to models that beat it. The
        stationary distribution of this chain is the score vector.

    Args:
        R: Binary tensor of shape (L, M, N) (or (L, M) which is treated as N=1).
        method: Ranking method passed to `scorio.utils.rank_scores`.
        return_scores: If True, return (ranking, scores) where scores are the
            stationary distribution.
        tie_handling:
            - "ignore": only decisive comparisons (i correct, j incorrect)
            - "half": treat ties (both same) as 0.5 win for each side
        smoothing: Nonnegative pseudocount added to every directed win count.
            Use this to avoid disconnected graphs when `tie_handling="ignore"`.
        teleport: Teleportation probability in [0, 1). When > 0, makes the
            Markov chain ergodic even if the comparison graph is disconnected.
        max_iter: Max iterations for the power method.
        tol: Convergence tolerance (L1 difference) for the power method.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns stationary-distribution scores
        (shape ``(L,)``).

    Formula:
        Let ``d_max`` be the maximum degree of the undirected comparison graph.
        For ``i != j``,

        .. math::
            P_{ij} = \\frac{1}{d_{\\max}}\\,\\widehat{P}_{j\\succ i},
            \\quad
            P_{ii} = 1 - \\sum_{j\\neq i} P_{ij}

        where ``P_hat`` is computed from pairwise tied-split outcomes.

    References:
        Negahban, S., Oh, S., & Shah, D. (2017).
        Rank Centrality: Ranking from Pairwise Comparisons.
        Operations Research.
    """
    R = validate_input(R)
    L = R.shape[0]

    tie_handling = str(tie_handling)
    if tie_handling not in {"ignore", "half"}:
        raise ValueError('tie_handling must be "ignore" or "half"')

    smoothing = float(smoothing)
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be >= 0")

    teleport = float(teleport)
    if not np.isfinite(teleport) or not (0.0 <= teleport < 1.0):
        raise ValueError("teleport must be in [0, 1)")

    if not isinstance(max_iter, (int, np.integer)):
        raise TypeError(f"max_iter must be an integer, got {type(max_iter).__name__}")
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be a positive finite scalar, got {tol}")

    if tie_handling == "ignore":
        wins = build_pairwise_wins(R)
    else:
        wins, ties = build_pairwise_counts(R)
        wins = wins + 0.5 * ties

    # Apply optional pseudocount smoothing.
    wins_s = wins + smoothing
    denom = wins_s + wins_s.T  # total (possibly smoothed) comparisons per pair

    eye = np.eye(L, dtype=bool)
    adj = (denom > 0) & ~eye
    deg = adj.sum(axis=1)
    d_max = int(deg.max()) if deg.size else 0

    if d_max == 0:
        scores = np.ones(L, dtype=float) / L
        ranking = rank_scores(scores)[method]
        return (ranking, scores) if return_scores else ranking

    if (
        teleport == 0.0
        and smoothing == 0.0
        and tie_handling == "ignore"
        and not _is_connected_undirected(adj)
    ):
        # With decisive-only comparisons and no regularization, the graph may
        # be disconnected; Rank Centrality is not identifiable across components.
        raise ValueError(
            "Rank Centrality requires a connected comparison graph; "
            "use teleport>0, smoothing>0, or tie_handling='half'."
        )

    # Transition matrix (row-stochastic), Negahban et al. (2017):
    #   P_{ij} = (1/d_max) * p̂_{j,i} for i!=j on edges, else 0
    #   P_{ii} = 1 - Σ_{j!=i} P_{ij}
    with np.errstate(divide="ignore", invalid="ignore"):
        p_ji = np.zeros((L, L), dtype=float)
        p_ji[adj] = (wins_s.T[adj] / denom[adj]).astype(float)

    P = np.zeros((L, L), dtype=float)
    P[adj] = p_ji[adj] / float(d_max)
    P[np.arange(L), np.arange(L)] = 1.0 - P.sum(axis=1)

    if teleport > 0.0:
        P = (1.0 - teleport) * P + teleport * (np.ones((L, L), dtype=float) / L)

    scores = _stationary_distribution_power(P, max_iter=max_iter, tol=tol)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


__all__ = ["rank_centrality"]
