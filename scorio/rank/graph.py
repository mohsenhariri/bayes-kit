"""
Graph-based ranking methods.

These methods convert pairwise outcomes into a graph or Markov chain and rank
models with stationary-distribution, spectral, or equilibrium concepts.

Notation
--------

Let :math:`R \\in \\{0,1\\}^{L \\times M \\times N}` and define decisive wins
:math:`W_{ij}` and ties :math:`T_{ij}`. A shared empirical pairwise relation is

.. math::
    \\widehat{P}_{i\\succ j}
    = \\frac{W_{ij} + \\tfrac12 T_{ij}}
           {W_{ij} + W_{ji} + T_{ij}}.

Graph methods construct an operator :math:`\\mathcal{G}(\\widehat{P})` and rank
from a derived score vector :math:`s`, such as a stationary distribution,
principal eigenvector, or game-theoretic equilibrium score.
"""

import numpy as np
from scipy.optimize import linprog

from scorio.utils import rank_scores

from ._base import build_pairwise_counts, validate_input
from ._types import RankMethod, RankResult


def _validate_positive_int(name: str, value: int, min_value: int = 1) -> int:
    """Validate integer hyperparameters."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    ivalue = int(value)
    if ivalue < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {ivalue}")
    return ivalue


def _validate_positive_float(name: str, value: float) -> float:
    """Validate positive finite scalar hyperparameters."""
    fvalue = float(value)
    if not np.isfinite(fvalue) or fvalue <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar, got {value}")
    return fvalue


def _pairwise_win_probabilities(R: np.ndarray) -> np.ndarray:
    """
    Build the empirical pairwise win-probability matrix from the response tensor.

    Using the pairwise win and tie counts from Section 2 (Representation) of the
    manuscript, define for i != j:

        P̂_{i≻j} = (W_{ij} + 1/2 T_{ij}) / (W_{ij} + W_{ji} + T_{ij}).

    Returns a matrix P of shape (L, L) with P[i, j] in [0, 1] and P[i, i] = 0.5.
    """
    wins, ties = build_pairwise_counts(R)
    total = wins + wins.T + ties

    L = wins.shape[0]
    P = np.full((L, L), 0.5, dtype=float)
    mask = total > 0
    P[mask] = (wins[mask] + 0.5 * ties[mask]) / total[mask]
    np.fill_diagonal(P, 0.5)
    return P


def _power_stationary_distribution_row_stochastic(
    C: np.ndarray, max_iter: int = 100_000, tol: float = 1e-12
) -> np.ndarray:
    """Stationary distribution π for a row-stochastic C via π <- π C."""
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    pi = np.ones(n, dtype=float) / n
    for _ in range(int(max_iter)):
        pi_new = pi @ C
        s = pi_new.sum()
        if s <= 0 or not np.all(np.isfinite(pi_new)):
            return np.ones(n, dtype=float) / n
        pi_new = pi_new / s
        if np.linalg.norm(pi_new - pi, 1) < tol:
            return pi_new
        pi = pi_new
    return pi


def pagerank(
    R: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    method: RankMethod = "competition",
    return_scores: bool = False,
    teleport: np.ndarray | None = None,
) -> RankResult:
    """
    Rank models with PageRank on the pairwise win-probability graph.

    Method context:
        Build a directed graph where edge ``j -> i`` has weight
        ``P_hat[i, j]`` (losers link to winners). Column-normalize to obtain a
        transition matrix and run the damped PageRank fixed point with a
        teleportation vector ``e``.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        damping: PageRank damping factor ``d`` in ``(0, 1)``.
        max_iter: Positive maximum number of power iterations.
        tol: Positive L1 convergence tolerance.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        teleport: Optional teleportation vector ``e`` (shape ``(L,)``,
            nonnegative, finite). If ``None``, uses uniform teleportation
            ``(1/L) * 1``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns PageRank scores ``r``
        (shape ``(L,)``).

    Notation:
        ``P_hat[i, j]`` is the tied-split empirical win probability of model
        ``i`` against model ``j``.

    Formula:
        .. math::
            P_{ij}
            = \\frac{\\widehat{P}_{i\\succ j}}
                   {\\sum_{k\\neq j}\\widehat{P}_{k\\succ j}}

        .. math::
            r = d P r + (1-d)\\,e

        where ``e`` is a probability vector. The default choice is
        :math:`e = \\frac{1}{L}\\mathbf{1}`.

    References:
        Page, L., et al. (1999). The PageRank Citation Ranking: Bringing
        Order to the Web. Stanford InfoLab.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> rank.pagerank(R).tolist()
        [1, 2]
    """
    damping = float(damping)
    if not np.isfinite(damping) or not (0.0 < damping < 1.0):
        raise ValueError("damping must be in (0, 1)")
    max_iter = _validate_positive_int("max_iter", max_iter)
    tol = _validate_positive_float("tol", tol)

    R = validate_input(R)
    L = R.shape[0]

    if teleport is None:
        e = np.ones(L, dtype=float) / L
    else:
        e = np.asarray(teleport, dtype=float)
        if e.ndim != 1 or e.shape[0] != L:
            raise ValueError(f"teleport must have shape (L={L},), got {e.shape}")
        if not np.all(np.isfinite(e)):
            raise ValueError("teleport must contain finite values")
        if np.any(e < 0):
            raise ValueError("teleport must be nonnegative")
        s = float(e.sum())
        if s <= 0:
            raise ValueError("teleport must sum to a positive value")
        e = e / s

    # Pairwise win probabilities P̂_{i≻j} in [0,1].
    P_hat = _pairwise_win_probabilities(R)

    # Build column-stochastic transition matrix
    # P[i, j] = probability of transitioning TO i FROM j
    W = P_hat.copy()
    np.fill_diagonal(W, 0.0)

    P = np.zeros((L, L), dtype=float)
    for j in range(L):
        col_sum = float(W[:, j].sum())
        if col_sum > 0:
            P[:, j] = W[:, j] / col_sum
        else:
            P[:, j] = 1.0 / L  # Uniform if no outgoing edges

    # PageRank iteration
    r = np.ones(L) / L

    for _ in range(max_iter):
        r_new = damping * (P @ r) + (1 - damping) * e
        if np.linalg.norm(r_new - r, 1) < tol:
            r = r_new
            break
        r = r_new

    ranking = rank_scores(r)[method]
    return (ranking, r) if return_scores else ranking


def spectral(
    R: np.ndarray,
    max_iter: int = 10_000,
    tol: float = 1e-12,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models by spectral centrality of pairwise win probabilities.

    Method context:
        Form a nonnegative matrix ``W`` with off-diagonal entries
        ``W[i, j] = P_hat[i, j]`` and diagonal self-loop mass equal to row sum.
        The normalized dominant right eigenvector is the score vector.
        This is an eigenvector-based Perron-style spectral ranking heuristic.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        max_iter: Positive max iterations for power iteration.
        tol: Positive L1 convergence tolerance.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns normalized spectral scores
        (shape ``(L,)``).

    Formula:
        .. math::
            W_{ij}=\\widehat{P}_{i\\succ j}\\;(i\\neq j),\\quad
            W_{ii}=\\sum_{j\\neq i}W_{ij}

        .. math::
            v \\propto W v,\\quad \\sum_i v_i = 1

    References:
        Vigna, S. (2016). Spectral ranking. *Network Science*.
        Keener, J. P. (1993). The Perron-Frobenius theorem and the ranking
        of football teams. *SIAM Review*.

    Examples:
        >>> import numpy as np
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> _, scores = spectral(R, return_scores=True)
        >>> scores[0] > scores[1]
        True
    """
    max_iter = _validate_positive_int("max_iter", max_iter)
    tol = _validate_positive_float("tol", tol)

    R = validate_input(R)
    L = R.shape[0]

    P_hat = _pairwise_win_probabilities(R)

    W = P_hat.copy()
    np.fill_diagonal(W, 0.0)
    np.fill_diagonal(W, W.sum(axis=1))

    v = np.ones(L, dtype=float) / L
    for _ in range(max_iter):
        v_new = W @ v
        s = float(v_new.sum())
        if s <= 0 or not np.all(np.isfinite(v_new)):
            v_uniform = np.ones(L, dtype=float) / L
            ranking = rank_scores(v_uniform)[method]
            return (ranking, v_uniform) if return_scores else ranking
        v_new = v_new / s
        if np.linalg.norm(v_new - v, 1) < tol:
            ranking = rank_scores(v_new)[method]
            return (ranking, v_new) if return_scores else ranking
        v = v_new

    ranking = rank_scores(v)[method]
    return (ranking, v) if return_scores else ranking


def alpharank(
    R: np.ndarray,
    alpha: float = 1.0,
    population_size: int = 50,
    max_iter: int = 100_000,
    tol: float = 1e-12,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> RankResult:
    """
    Rank models with single-population alpha-Rank.

    Method context:
        Treat models as strategies in a symmetric constant-sum game with payoff
        ``P_hat[i, j]``. Build fixation probabilities under Fermi selection in a
        finite population and compute the stationary distribution of the induced
        Markov chain. This corresponds to the single-population setting from
        the AlphaRank framework (the paper's general method also covers
        multi-population and asymmetric games).

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        alpha: Selection intensity ``alpha >= 0``.
        population_size: Finite population size ``m >= 2``.
        max_iter: Positive max iterations for stationary-distribution iteration.
        tol: Positive L1 convergence tolerance.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns alpha-Rank stationary
        distribution scores (shape ``(L,)``).

    Formula:
        .. math::
            u = \\alpha\\frac{m}{m-1}\\left(\\widehat{P}_{r\\succ s}-\\frac12\\right),
            \\quad
            \\rho_{r,s} =
            \\begin{cases}
            \\frac{1-e^{-u}}{1-e^{-mu}}, & u\\neq 0 \\\\
            \\frac{1}{m}, & u=0
            \\end{cases}

        .. math::
            C_{sr}=\\frac{1}{L-1}\\rho_{r,s},\\quad
            C_{ss}=1-\\sum_{r\\neq s}C_{sr}

    References:
        Omidshafiei, S., et al. (2019). α-Rank: Multi-Agent Evaluation by Evolution.
        Scientific Reports.

    Examples:
        >>> import numpy as np
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> _, scores = alpharank(R, return_scores=True)
        >>> scores[0] > scores[1]
        True
    """
    max_iter = _validate_positive_int("max_iter", max_iter)
    tol = _validate_positive_float("tol", tol)
    m = _validate_positive_int("population_size", population_size, min_value=2)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be >= 0")

    R = validate_input(R)
    L = R.shape[0]

    P_hat = _pairwise_win_probabilities(R)

    # Fixation probabilities for constant-sum symmetric games (OpenSpiel’s
    # implementation follows Omidshafiei et al., 2019; payoff sum is 1 here).
    payoff_sum = 1.0
    eta = 1.0 / float(L - 1)

    def rho(payoff_rs: float) -> float:
        # u = α * m/(m-1) * (payoff_rs - payoff_sum/2)
        u = alpha * (m / float(m - 1)) * (float(payoff_rs) - 0.5 * payoff_sum)
        if abs(u) < 1e-14:
            return 1.0 / float(m)

        # Stable computation of (1 - exp(-u)) / (1 - exp(-m u)).
        # Use expm1 to avoid catastrophic cancellation near 0.
        # Guard very large |u| to avoid overflow.
        if u > 50:
            return 1.0
        if u < -50:
            # Very unfavorable mutant; fixation probability is ~0.
            return 0.0
        num = -np.expm1(-u)
        den = -np.expm1(-float(m) * u)
        if den == 0.0:
            return 1.0 / float(m)
        out = num / den
        return float(np.clip(out, 0.0, 1.0))

    C = np.zeros((L, L), dtype=float)
    for resident in range(L):  # resident state
        for r in range(L):  # mutant candidate
            if r == resident:
                continue
            C[resident, r] = eta * rho(P_hat[r, resident])
        C[resident, resident] = 1.0 - float(np.sum(C[resident, :]))

    pi = _power_stationary_distribution_row_stochastic(C, max_iter=max_iter, tol=tol)
    pi = np.clip(pi, 0.0, None)
    total = float(pi.sum())

    scores = (pi / total) if total > 0 else (np.ones(L, dtype=float) / L)

    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def nash(
    R: np.ndarray,
    n_iter: int = 100,
    temperature: float = 0.1,
    solver: str = "lp",
    score_type: str = "vs_equilibrium",
    return_equilibrium: bool = False,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """
    Rank models via Nash equilibrium on the zero-sum meta-game.

    Method context:
        Construct antisymmetric payoff matrix
        ``A = 2 * P_hat - 1`` (with zero diagonal), solve a maximin mixed
        strategy ``x``, then score models by expected performance versus ``x``
        (Nash-averaging style) or alternative score views.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        n_iter: Unused when solver="lp" (kept for backward compatibility).
        temperature: Unused when solver="lp" (kept for backward compatibility).
        solver: Currently only "lp" is supported.
        score_type: Which score vector to rank by:
            - "vs_equilibrium": expected win probability vs equilibrium opponent.
            - "equilibrium": the equilibrium mixture itself.
            - "advantage_vs_equilibrium": expected zero-sum advantage vs equilibrium.
        return_equilibrium: If True, also return the equilibrium mixture.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns scores for the selected
        ``score_type``.
        If ``return_equilibrium=True``, also returns equilibrium mixture ``x``.

    Formula:
        .. math::
            A_{ij} = 2\\widehat{P}_{i\\succ j} - 1,\\quad A_{ii}=0

        .. math::
            x \\in \\arg\\max_{x\\in\\Delta^{L-1}}\\min_{y\\in\\Delta^{L-1}} x^\\top A y

        .. math::
            s_i = \\sum_j \\widehat{P}_{i\\succ j} x_j

    Notes:
        The cited papers establish empirical Nash equilibria and Nash-based
        population evaluation in symmetric zero-sum games. The
        ``score_type`` choices here are practical per-model summaries derived
        from that equilibrium for ranking API use.

    References:
        Balduzzi, D., Garnelo, M., Bachrach, Y., Czarnecki, W. M.,
        P{\'e}rolat, J., Jaderberg, M., & Graepel, T. (2019).
        Open-ended Learning in Symmetric Zero-sum Games. ICML.
        Balduzzi, D., Tuyls, K., P{\'e}rolat, J., & Graepel, T. (2018).
        Re-evaluating Evaluation. NeurIPS.

    Examples:
        >>> import numpy as np
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> ranks, scores, eq = nash(R, return_scores=True, return_equilibrium=True)
        >>> scores[0] > scores[1]
        True
    """
    _validate_positive_int("n_iter", n_iter)
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a positive finite scalar")

    R = validate_input(R)
    L = R.shape[0]

    solver = str(solver)
    if solver != "lp":
        raise ValueError('solver must be "lp"')

    score_type = str(score_type)
    if score_type not in {"vs_equilibrium", "equilibrium", "advantage_vs_equilibrium"}:
        raise ValueError(
            'score_type must be one of "vs_equilibrium", "equilibrium", "advantage_vs_equilibrium"'
        )

    P_hat = _pairwise_win_probabilities(R)

    # Zero-sum payoff matrix in [-1, 1], antisymmetric when P_hat is derived
    # from tied-split win rates.
    A = 2.0 * P_hat - 1.0
    np.fill_diagonal(A, 0.0)

    if np.allclose(A, 0.0, atol=1e-14):
        equilibrium = np.ones(L, dtype=float) / L
        if score_type == "equilibrium":
            scores = equilibrium
        elif score_type == "advantage_vs_equilibrium":
            scores = A @ equilibrium
        else:
            scores = P_hat @ equilibrium

        ranking = rank_scores(scores)[method]
        if return_scores and return_equilibrium:
            return ranking, scores, equilibrium
        if return_scores:
            return ranking, scores
        if return_equilibrium:
            return ranking, equilibrium
        return ranking

    # Compute a maximin mixed strategy x via linear programming:
    #   maximize v
    #   s.t. A^T x >= v 1,  sum x = 1,  x >= 0
    #
    # We solve in variables (x, v) by minimizing -v.
    c = np.zeros(L + 1, dtype=float)
    c[-1] = -1.0

    A_ub = np.hstack([-A.T, np.ones((L, 1), dtype=float)])
    b_ub = np.zeros(L, dtype=float)

    A_eq = np.zeros((1, L + 1), dtype=float)
    A_eq[0, :L] = 1.0
    b_eq = np.array([1.0], dtype=float)

    bounds = [(0.0, None)] * L + [(None, None)]
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
        equilibrium = np.ones(L, dtype=float) / L
        if score_type == "equilibrium":
            scores = equilibrium
        elif score_type == "advantage_vs_equilibrium":
            scores = A @ equilibrium
        else:
            scores = P_hat @ equilibrium

        ranking = rank_scores(scores)[method]
        if return_scores and return_equilibrium:
            return ranking, scores, equilibrium
        if return_scores:
            return ranking, scores
        if return_equilibrium:
            return ranking, equilibrium
        return ranking

    x = np.asarray(res.x[:L], dtype=float)
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    equilibrium = (x / s) if s > 0 else (np.ones(L, dtype=float) / L)

    if score_type == "equilibrium":
        scores = equilibrium
    elif score_type == "advantage_vs_equilibrium":
        scores = A @ equilibrium
    else:
        scores = P_hat @ equilibrium

    ranking = rank_scores(scores)[method]
    if return_scores and return_equilibrium:
        return ranking, scores, equilibrium
    if return_scores:
        return ranking, scores
    if return_equilibrium:
        return ranking, equilibrium
    return ranking


__all__ = ["pagerank", "spectral", "alpharank", "nash"]
