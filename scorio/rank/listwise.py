"""
Listwise and setwise probabilistic choice models (Luce family).

In the binary tensor setting :math:`R \\in \\{0,1\\}^{L \\times M \\times N}`,
each event :math:`(m,n)` induces a winner set :math:`W_{mn}` (correct models)
and loser set :math:`L_{mn}` (incorrect models).

Luce-family models assign positive strengths :math:`\\pi_i` and define
selection probabilities over comparison sets:

.. math::
    \\Pr(i \\mid S) = \\frac{\\pi_i}{\\sum_{j \\in S} \\pi_j}.

For strict orderings :math:`i_1 \\succ i_2 \\succ \\cdots \\succ i_K`, the
Plackett-Luce likelihood is

.. math::
    \\Pr(i_1 \\succ i_2 \\succ \\cdots \\succ i_K)
    = \\prod_{k=1}^{K}
    \\frac{\\pi_{i_k}}{\\sum_{j=k}^{K} \\pi_{i_j}}.

This module uses three constructions:

- **Pairwise reduction (wins only)**:
  reduce :math:`R` to decisive pairwise counts and fit the Bradley-Terry form
  of Plackett-Luce.
- **Setwise likelihood with ties**:
  treat each observed winner set as one tied choice event using
  Davidson-Luce normalization.
- **Setwise composite likelihood**:
  convert each winner into a Luce choice from ``{winner} union {losers}``
  (Bradley-Terry-Luce rank breaking).

Estimation uses MM updates for the Plackett-Luce pairwise reduction and
L-BFGS optimization for the setwise models.

References:
    Plackett, R. L. (1975). The Analysis of Permutations.
    Journal of the Royal Statistical Society: Series C.

    Luce, R. D. (1959). Individual Choice Behavior: A Theoretical Analysis.
    John Wiley & Sons.

    Hunter, D. R. (2004). MM algorithms for generalized Bradley-Terry models.
    The Annals of Statistics, 32(1), 384-406.

    Firth, D., Kosmidis, I., & Turner, H. L. (2019). Davidson-Luce model for
    multi-item choice with ties. arXiv:1909.07123.
"""

import numpy as np
from scipy.optimize import minimize

from scorio.utils import rank_scores

from ._base import (
    average_equivalent_scores,
    average_event_exchangeable_scores,
    build_pairwise_wins,
    is_strongly_connected,
    validate_input,
)
from ._types import RankMethod, RankResult
from .priors import CauchyPrior, GaussianPrior, LaplacePrior, Prior, UniformPrior


def _validate_positive_int(name: str, value: int, minimum: int = 1) -> int:
    """Validate integer hyperparameters with a lower bound."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    ivalue = int(value)
    if ivalue < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {ivalue}")
    return ivalue


def _validate_positive_float(name: str, value: float, minimum: float = 0.0) -> float:
    """Validate finite scalar hyperparameters that must be > minimum."""
    fvalue = float(value)
    if not np.isfinite(fvalue) or fvalue <= minimum:
        raise ValueError(f"{name} must be a finite scalar > {minimum}, got {value}")
    return fvalue


def _coerce_prior(prior: Prior | float) -> Prior:
    """
    Normalize prior specification to a Prior instance.

    A numeric prior value is interpreted as Gaussian variance.
    """
    if isinstance(prior, bool):
        raise TypeError("prior must be a Prior object or positive finite float")

    if isinstance(prior, (int, float, np.integer, np.floating)):
        var = _validate_positive_float("prior", float(prior), minimum=0.0)
        return GaussianPrior(mean=0.0, var=var)

    if not isinstance(prior, Prior):
        raise TypeError(
            f"prior must be a Prior object or float, got {type(prior).__name__}"
        )

    return prior


def _prior_is_exchangeable(prior: Prior) -> bool:
    """Whether permuting model labels leaves the prior penalty unchanged."""
    return type(prior) in {GaussianPrior, LaplacePrior, CauchyPrior, UniformPrior}


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return -np.inf
    max_v = np.max(values)
    return float(max_v + np.log(np.sum(np.exp(values - max_v))))


def _log_elementary_symmetric_sum(log_x: np.ndarray, k: int) -> float:
    """
    Compute log(e_k(x)) where e_k is the k-th elementary symmetric polynomial.

        e_k(x) = Σ_{|T|=k} ∏_{i∈T} x_i

    We work in log-space using the classic DP:
        e_j <- e_j + x_i * e_{j-1}
    implemented as log-add-exp.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return 0.0

    log_x = np.asarray(log_x, dtype=float)
    n = log_x.size
    if k > n:
        return -np.inf

    log_e = np.full(k + 1, -np.inf, dtype=float)
    log_e[0] = 0.0  # log(1)

    for i in range(n):
        upper = min(k, i + 1)
        for j in range(upper, 0, -1):
            log_e[j] = np.logaddexp(log_e[j], log_e[j - 1] + log_x[i])

    return float(log_e[k])


def _extract_winners_losers_events(
    R: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Convert binary responses into (winners, losers) events per (question, trial).

    Winners = models with response==1
    Losers  = models with response==0

    Events where all models are winners or all are losers are discarded.
    """
    L, M, N = R.shape
    events: list[tuple[np.ndarray, np.ndarray]] = []
    for m in range(M):
        for n in range(N):
            winners = np.flatnonzero(R[:, m, n] == 1)
            if winners.size in (0, L):
                continue
            losers = np.flatnonzero(R[:, m, n] == 0)
            events.append((winners.astype(int), losers.astype(int)))
    return events


def _davidson_equivalence_statistics(
    events: list[tuple[np.ndarray, np.ndarray]],
    n_models: int,
    max_tie_order: int,
) -> np.ndarray:
    """Count each model's winner memberships by observed tie order."""
    counts = np.zeros((n_models, max_tie_order), dtype=int)
    for winners, _ in events:
        counts[winners, winners.size - 1] += 1
    return counts


def plackett_luce(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 10_000,
    tol: float = 1e-8,
) -> RankResult:
    """
    Rank models with Plackett-Luce maximum likelihood.

    Method context:
        In Scorio's binary tensor setting, we reduce outcomes to decisive pairwise
        win counts and fit the Bradley-Terry form of a Plackett-Luce model using
        Hunter's MM updates.

    References:
        Plackett, R. L. (1975). The Analysis of Permutations.
        Luce, R. D. (1959). Individual Choice Behavior.
        Hunter, D. R. (2004). MM Algorithms for Generalized Bradley-Terry Models.

    Args:
        R: Binary tensor of shape ``(L, M, N)`` (or ``(L, M)`` treated as ``N=1``).
        method: Ranking method passed to ``scorio.utils.rank_scores``.
        return_scores: If True, return ``(ranking, scores)``.
        max_iter: Positive MM iteration budget.
        tol: Positive finite convergence tolerance on max parameter change.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, returns ``(ranking, scores)`` where scores are
        the estimated strengths ``pi``.

    Notation:
        ``W_ij``: number of decisive outcomes where model ``i`` beats ``j``.
        ``N_ij = W_ij + W_ji``: total decisive pairwise comparisons.
        ``pi_i > 0``: latent model strengths.

    Formula:
        .. math::
            \\pi_i^{(k+1)} =
            \\frac{\\sum_j W_{ij}}
                 {\\sum_{j \\ne i} N_{ij} / (\\pi_i^{(k)} + \\pi_j^{(k)})}

        followed by normalization to resolve scale non-identifiability.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [1, 0, 0, 1],
        ...     [0, 1, 0, 0],
        ...     [0, 0, 1, 0],
        ... ])
        >>> ranks = rank.plackett_luce(R)
        >>> bool(ranks[0] < ranks[1])  # Model 0 has better (lower) rank
        True

    Notes:
        This implementation intentionally ignores within-outcome ties
        (both-correct or both-incorrect), matching pairwise decisive reduction.
        A finite ML estimate requires a strongly connected directed win graph;
        separated data raises ``ValueError`` and should use
        :func:`plackett_luce_map`.
    """
    R = validate_input(R)
    max_iter = _validate_positive_int("max_iter", max_iter)
    tol = _validate_positive_float("tol", tol, minimum=0.0)

    wins = build_pairwise_wins(R)
    scores = _mm_plackett_luce(wins, max_iter=max_iter, tol=tol)
    scores = average_equivalent_scores(scores, R)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def plackett_luce_map(
    R: np.ndarray,
    prior: Prior | float = 1.0,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
) -> RankResult:
    """
    Rank models with Plackett-Luce maximum a posteriori estimation.

    Method context:
        Adds a prior penalty on centered log-strengths to the pairwise-reduced
        Plackett-Luce likelihood. Numeric priors are interpreted as Gaussian prior
        variances.

    References:
        Luce, R. D. (1959). Individual Choice Behavior.
        Hunter, D. R. (2004). MM Algorithms for Generalized Bradley-Terry Models.

    Args:
        R: Binary tensor of shape ``(L, M, N)`` (or ``(L, M)`` treated as ``N=1``).
        prior: ``Prior`` instance or positive finite float variance for
            ``GaussianPrior(mean=0, var=prior)``.
        method: Ranking method passed to ``scorio.utils.rank_scores``.
        return_scores: If True, return ``(ranking, scores)``.
        max_iter: Positive optimizer iteration budget.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, returns ``(ranking, scores)``.

    Formula:
        Let ``theta_i = log(pi_i)`` and ``P(theta)`` be the prior penalty.

        .. math::
            \\hat\\theta \\in
            \\arg\\min_{\\theta}
            \\left[
                -\\sum_{i \\ne j} W_{ij}
                \\left(\\theta_i - \\log(e^{\\theta_i}+e^{\\theta_j})\\right)
                + P(\\theta - \\bar\\theta)
            \\right]

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> ranks = rank.plackett_luce_map(R, prior=1.0)
        >>> bool(ranks[0] < ranks[1])
        True

    Notes:
        The MAP objective is solved with L-BFGS-B over centered log-strengths.
    """
    R = validate_input(R)
    max_iter = _validate_positive_int("max_iter", max_iter)
    prior = _coerce_prior(prior)

    wins = build_pairwise_wins(R)
    scores = _estimate_pl_map(wins, prior, max_iter=max_iter)
    if _prior_is_exchangeable(prior):
        scores = average_equivalent_scores(scores, R)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def davidson_luce(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
    max_tie_order: int | None = None,
) -> RankResult:
    """
    Rank models with Davidson-Luce maximum likelihood (setwise ties).

    Method context:
        Each question-trial induces a winner set ``W`` and loser set ``L``.
        Davidson-Luce models tied winners directly with tie-order parameters.
        Normalization terms are computed with elementary symmetric polynomials.

    References:
        Firth, D., Kosmidis, I., & Turner, H. L. (2019).
        Davidson-Luce model for multi-item choice with ties.
        https://arxiv.org/abs/1909.07123

    Args:
        R: Binary tensor of shape ``(L, M, N)`` (or ``(L, M)`` treated as ``N=1``).
        method: Ranking method passed to ``scorio.utils.rank_scores``.
        return_scores: If True, return ``(ranking, scores)``.
        max_iter: Positive optimizer iteration budget.
        max_tie_order: Maximum tie order ``D`` used in normalization;
            default is ``L-1``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, returns ``(ranking, scores)`` where scores are
        strengths ``alpha``.

    Notation:
        ``S = W \\cup L`` is the comparison set and ``t = |W|``.
        ``delta_1 = 1`` and ``delta_t > 0`` for ``t >= 2``.
        ``g_t(T) = (\\prod_{i\\in T} alpha_i)^{1/t}``.

    Formula:
        .. math::
            \\Pr(W \\mid S) =
            \\frac{\\delta_t g_t(W)}
            {\\sum_{t'=1}^{\\min(D,|S|)} \\delta_{t'}
             \\sum_{|T|=t'} g_{t'}(T)}

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [1, 0, 0, 1],
        ...     [0, 1, 0, 0],
        ...     [0, 0, 1, 0],
        ... ])
        >>> ranks = rank.davidson_luce(R)
        >>> ranks.tolist()
        [1, 2, 2]

    Notes:
        Events with all winners or all losers are dropped as uninformative.
    """
    R = validate_input(R)
    max_iter = _validate_positive_int("max_iter", max_iter)

    events = _extract_winners_losers_events(R)
    L = R.shape[0]

    if max_tie_order is None:
        max_tie_order = max(L - 1, 1)
    max_tie_order = _validate_positive_int("max_tie_order", max_tie_order)
    if max_tie_order > L:
        raise ValueError(f"max_tie_order must be <= number of models ({L})")
    if any(winners.size > max_tie_order for winners, _ in events):
        raise ValueError(
            "Observed winner-set size exceeds max_tie_order; increase "
            "max_tie_order so every observed event has positive model probability."
        )

    scores, _ = _estimate_davidson_luce_ml(
        events, n_models=L, max_tie_order=max_tie_order, max_iter=max_iter
    )
    scores = average_equivalent_scores(
        scores,
        _davidson_equivalence_statistics(events, L, max_tie_order),
    )
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def davidson_luce_map(
    R: np.ndarray,
    prior: Prior | float = 1.0,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
    max_tie_order: int | None = None,
) -> RankResult:
    """
    Rank models with Davidson-Luce MAP estimation.

    Method context:
        Adds a prior penalty on centered log-strengths to the Davidson-Luce
        setwise tie likelihood.

    Args:
        R: Binary tensor of shape ``(L, M, N)`` (or ``(L, M)`` treated as ``N=1``).
        prior: ``Prior`` instance or positive finite float variance for
            ``GaussianPrior(mean=0, var=prior)``.
        method: Ranking method passed to ``scorio.utils.rank_scores``.
        return_scores: If True, return ``(ranking, scores)``.
        max_iter: Positive optimizer iteration budget.
        max_tie_order: Maximum tie order ``D`` used in normalization;
            default is ``L-1``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, returns ``(ranking, scores)``.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> ranks = rank.davidson_luce_map(R, prior=1.0)
        >>> bool(ranks[0] < ranks[1])
        True
    """
    R = validate_input(R)
    max_iter = _validate_positive_int("max_iter", max_iter)

    L = R.shape[0]

    prior = _coerce_prior(prior)

    events = _extract_winners_losers_events(R)

    if max_tie_order is None:
        max_tie_order = max(L - 1, 1)
    max_tie_order = _validate_positive_int("max_tie_order", max_tie_order)
    if max_tie_order > L:
        raise ValueError(f"max_tie_order must be <= number of models ({L})")
    if any(winners.size > max_tie_order for winners, _ in events):
        raise ValueError(
            "Observed winner-set size exceeds max_tie_order; increase "
            "max_tie_order so every observed event has positive model probability."
        )

    scores, _ = _estimate_davidson_luce_map(
        events,
        n_models=L,
        prior=prior,
        max_tie_order=max_tie_order,
        max_iter=max_iter,
    )
    if _prior_is_exchangeable(prior):
        scores = average_equivalent_scores(
            scores,
            _davidson_equivalence_statistics(events, L, max_tie_order),
        )
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def bradley_terry_luce(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
) -> RankResult:
    """
    Rank models with Bradley-Terry-Luce composite-likelihood ML.

    Method context:
        For each event ``(W, L)``, each winner ``i in W`` is treated as a Luce
        choice from ``{i} union L``. This yields a rank-breaking composite
        likelihood objective (not a normalized probability for the whole winner
        set ``W`` as a single event).

    References:
        Luce, R. D. (1959). Individual Choice Behavior.

    Args:
        R: Binary tensor of shape ``(L, M, N)`` (or ``(L, M)`` treated as ``N=1``).
        method: Ranking method passed to ``scorio.utils.rank_scores``.
        return_scores: If True, return ``(ranking, scores)``.
        max_iter: Positive optimizer iteration budget.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, returns ``(ranking, scores)``.

    Formula:
        For event ``(W, L)``, define per-winner choice sets
        ``C_i = {i} union L`` and optimize the composite log-likelihood

        .. math::
            \\ell_{\\mathrm{comp}}(\\pi)
            = \\sum_{(W,L)}\\sum_{i\\in W}
            \\left[
              \\log \\pi_i
              - \\log\\left(\\pi_i + \\sum_{j\\in L}\\pi_j\\right)
            \\right]

    Notes:
        This objective is a Luce-style composite likelihood induced by
        rank-breaking, rather than a full normalized likelihood over all
        possible winner subsets. Its unregularized finite MLE requires a
        strongly connected directed choice graph; otherwise use
        :func:`bradley_terry_luce_map`.
    """
    R = validate_input(R)
    max_iter = _validate_positive_int("max_iter", max_iter)

    events = _extract_winners_losers_events(R)
    scores = _estimate_btl_ml(events, n_models=R.shape[0], max_iter=max_iter)
    scores = average_event_exchangeable_scores(scores, R)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def bradley_terry_luce_map(
    R: np.ndarray,
    prior: Prior | float = 1.0,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
) -> RankResult:
    """
    Rank models with Bradley-Terry-Luce composite-likelihood MAP estimation.

    Method context:
        Adds a prior penalty on centered log-strengths to the BTL setwise-choice
        composite likelihood.

    Args:
        R: Binary tensor of shape ``(L, M, N)`` (or ``(L, M)`` treated as ``N=1``).
        prior: ``Prior`` instance or positive finite float variance for
            ``GaussianPrior(mean=0, var=prior)``.
        method: Ranking method passed to ``scorio.utils.rank_scores``.
        return_scores: If True, return ``(ranking, scores)``.
        max_iter: Positive optimizer iteration budget.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, returns ``(ranking, scores)``.
    """
    R = validate_input(R)
    max_iter = _validate_positive_int("max_iter", max_iter)
    prior = _coerce_prior(prior)

    events = _extract_winners_losers_events(R)
    scores = _estimate_btl_map(
        events, n_models=R.shape[0], prior=prior, max_iter=max_iter
    )
    if _prior_is_exchangeable(prior):
        scores = average_event_exchangeable_scores(scores, R)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def _mm_plackett_luce(
    wins: np.ndarray, max_iter: int = 500, tol: float = 1e-8
) -> np.ndarray:
    """
    MM algorithm for Plackett-Luce and Bradley-Terry MLE.

    The MM algorithm from Hunter (2004) iteratively updates strength
    parameters using a guaranteed-to-converge update rule:

        π_i^{new} = W_i / Σⱼ≠ᵢ (n_ij + n_ji) / (π_i^{old} + π_j^{old})

    where W_i is the total number of wins for model i.

    Args:
        wins: Pairwise win matrix of shape (L, L).
              wins[i, j] = number of times model i beats model j.
        max_iter: Maximum number of MM iterations.
        tol: Convergence tolerance (max change in π).

    Returns:
        Strength parameters π of shape (L,), normalized to sum to 1.

    References:
        Hunter, D. R. (2004). MM algorithms for generalized Bradley-Terry
        models. The Annals of Statistics, 32(1), 384-406.
    """
    L = wins.shape[0]

    # Initialize with win proportions
    W = wins.sum(axis=1)  # Total wins for each model
    total_wins = W.sum()

    if total_wins == 0:
        # No comparisons - return uniform
        return np.ones(L) / L
    if not is_strongly_connected(wins > 0):
        raise ValueError(
            "Plackett-Luce has no finite maximum-likelihood estimate because "
            "the directed win graph is not strongly connected; use "
            "plackett_luce_map for a regularized estimate."
        )

    pi = W / total_wins
    pi = np.maximum(pi, 1e-10)

    # Total comparisons between each pair
    n_comparisons = wins + wins.T

    converged = False
    for _iteration in range(max_iter):
        pi_old = pi.copy()

        for i in range(L):
            denom = 0.0
            for j in range(L):
                if i == j:
                    continue
                if n_comparisons[i, j] > 0:
                    denom += n_comparisons[i, j] / (pi_old[i] + pi_old[j])

            if denom > 0:
                pi[i] = W[i] / denom
            else:
                pi[i] = pi_old[i]

        # Normalize to sum to 1
        pi_sum = pi.sum()
        if pi_sum > 0:
            pi = pi / pi_sum
        pi = np.maximum(pi, 1e-10)

        # Check convergence
        if np.max(np.abs(pi - pi_old)) < tol:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            "plackett_luce MM optimization did not converge within max_iter"
        )

    return pi


def _estimate_pl_map(wins: np.ndarray, prior: Prior, max_iter: int = 500) -> np.ndarray:
    """
    Estimate Plackett-Luce strengths via MAP with configurable prior.

    Maximizes the posterior: log-likelihood + log-prior

    The log-likelihood for pairwise comparisons is:
        L(θ) = Σᵢⱼ n_ij * [θ_i - log(exp(θ_i) + exp(θ_j))]

    where θ_i = log(π_i) are the log-strengths and n_ij is the number
    of times model i beats model j.

    Args:
        wins: Pairwise win matrix of shape (L, L).
        prior: Prior distribution on log-strengths.
        max_iter: Maximum optimization iterations.

    Returns:
        Strength parameters π of shape (L,).
    """
    if type(prior) is UniformPrior:
        return _mm_plackett_luce(wins, max_iter=max_iter)

    L = wins.shape[0]

    def negative_log_posterior(log_pi):
        # Center for identifiability (model is identified up to constant)
        log_pi = log_pi - log_pi.mean()

        # Negative log-likelihood
        nll = 0.0
        for i in range(L):
            for j in range(L):
                if i == j:
                    continue
                n_ij = wins[i, j]
                if n_ij > 0:
                    # log P(i beats j) = θ_i - log(exp(θ_i) + exp(θ_j))
                    nll -= n_ij * (log_pi[i] - np.logaddexp(log_pi[i], log_pi[j]))

        # Prior penalty (negative log-prior)
        prior_penalty = prior.penalty(log_pi)

        return nll + prior_penalty

    # Initialize with win proportions
    total_wins = wins.sum(axis=1)
    total_wins = np.maximum(total_wins, 1)
    log_pi_init = np.log(total_wins / total_wins.sum())

    result = minimize(
        negative_log_posterior,
        log_pi_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    if not result.success:
        raise RuntimeError(f"plackett_luce_map optimization failed: {result.message}")

    log_pi = result.x
    log_pi = log_pi - log_pi.mean()
    return np.exp(np.clip(log_pi, -30.0, 30.0))


def _log_denominator_davidson_luce(
    log_alpha: np.ndarray,
    log_delta_params: np.ndarray,
    comparison_set: np.ndarray,
    max_tie_order: int,
) -> float:
    """
    Compute log Z where:

        Z = Σ_{t=1..D} δ_t · Σ_{|T|=t} (∏_{i∈T} α_i)^{1/t}
          = Σ_{t=1..D} δ_t · e_t(α^{1/t})

    with δ_1 ≡ 1 and δ_t = exp(log_delta_params[t-2]) for t>=2.
    """
    items = np.asarray(comparison_set, dtype=int)
    if items.size == 0:
        return -np.inf

    D = min(int(max_tie_order), items.size)
    terms: list[float] = []
    for t in range(1, D + 1):
        if t == 1:
            log_delta_t = 0.0
        else:
            idx = t - 2
            log_delta_t = (
                float(log_delta_params[idx]) if idx < log_delta_params.size else 0.0
            )

        log_x = log_alpha[items] / float(t)  # x_i = alpha_i^{1/t}
        log_e_t = _log_elementary_symmetric_sum(log_x, t)
        if np.isneginf(log_e_t):
            continue
        terms.append(log_delta_t + log_e_t)

    return _logsumexp(np.asarray(terms, dtype=float))


def _estimate_davidson_luce_ml(
    events: list[tuple[np.ndarray, np.ndarray]],
    n_models: int,
    max_tie_order: int,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate (alpha, delta) via ML for the Davidson–Luce model.

    Events are (winners, losers); the comparison set is winners ∪ losers.
    """
    L = int(n_models)
    if not events:
        return np.ones(L) / L, np.ones(max(max_tie_order - 1, 1))

    directed_choices = np.zeros((L, L), dtype=bool)
    for winners, losers in events:
        directed_choices[np.ix_(winners, losers)] = True
        directed_choices[np.ix_(winners, winners)] = True
    np.fill_diagonal(directed_choices, False)
    if not is_strongly_connected(directed_choices):
        raise ValueError(
            "Davidson-Luce has no finite maximum-likelihood strength estimate "
            "because its win/co-winner graph is not strongly connected; use "
            "davidson_luce_map with a proper prior."
        )

    # In this binary setting, each event partitions all models into winners/losers.
    comparison_set = np.arange(L, dtype=int)

    def negative_log_likelihood(params: np.ndarray) -> float:
        log_alpha = params[:L]
        log_delta_params = params[L:]

        log_alpha = log_alpha - log_alpha.mean()

        nll = 0.0
        for winners, _ in events:
            t = winners.size
            if t < 1 or t > max_tie_order:
                raise ValueError("Observed winner-set size exceeds max_tie_order")

            log_delta_t = 0.0 if t == 1 else float(log_delta_params[t - 2])
            log_numerator = log_delta_t + float(log_alpha[winners].mean())

            log_denom = _log_denominator_davidson_luce(
                log_alpha, log_delta_params, comparison_set, max_tie_order=max_tie_order
            )

            nll -= log_numerator - log_denom

        return float(nll)

    log_alpha0 = np.zeros(L, dtype=float)
    log_delta0 = np.zeros(max(max_tie_order - 1, 0), dtype=float)
    params0 = np.concatenate([log_alpha0, log_delta0])

    result = minimize(
        negative_log_likelihood,
        params0,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    if not result.success:
        raise RuntimeError(f"davidson_luce optimization failed: {result.message}")

    log_alpha_hat = result.x[:L]
    log_alpha_hat = log_alpha_hat - log_alpha_hat.mean()
    alpha = np.exp(np.clip(log_alpha_hat, -30.0, 30.0))

    log_delta_hat = result.x[L:]
    delta = np.exp(log_delta_hat) if log_delta_hat.size > 0 else np.array([1.0])

    return alpha, delta


def _estimate_davidson_luce_map(
    events: list[tuple[np.ndarray, np.ndarray]],
    n_models: int,
    prior: Prior,
    max_tie_order: int,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    if type(prior) is UniformPrior:
        return _estimate_davidson_luce_ml(
            events,
            n_models=n_models,
            max_tie_order=max_tie_order,
            max_iter=max_iter,
        )

    L = int(n_models)
    if not events:
        return np.ones(L) / L, np.ones(max(max_tie_order - 1, 1))

    comparison_set = np.arange(L, dtype=int)

    def negative_log_posterior(params: np.ndarray) -> float:
        log_alpha = params[:L]
        log_delta_params = params[L:]

        log_alpha = log_alpha - log_alpha.mean()

        nll = 0.0
        for winners, _ in events:
            t = winners.size
            if t < 1 or t > max_tie_order:
                raise ValueError("Observed winner-set size exceeds max_tie_order")

            log_delta_t = 0.0 if t == 1 else float(log_delta_params[t - 2])
            log_numerator = log_delta_t + float(log_alpha[winners].mean())

            log_denom = _log_denominator_davidson_luce(
                log_alpha, log_delta_params, comparison_set, max_tie_order=max_tie_order
            )
            nll -= log_numerator - log_denom

        return float(nll + prior.penalty(log_alpha))

    log_alpha0 = np.zeros(L, dtype=float)
    log_delta0 = np.zeros(max(max_tie_order - 1, 0), dtype=float)
    params0 = np.concatenate([log_alpha0, log_delta0])

    result = minimize(
        negative_log_posterior,
        params0,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    if not result.success:
        raise RuntimeError(f"davidson_luce_map optimization failed: {result.message}")

    log_alpha_hat = result.x[:L]
    log_alpha_hat = log_alpha_hat - log_alpha_hat.mean()
    alpha = np.exp(np.clip(log_alpha_hat, -30.0, 30.0))

    log_delta_hat = result.x[L:]
    delta = np.exp(log_delta_hat) if log_delta_hat.size > 0 else np.array([1.0])

    return alpha, delta


def _estimate_btl_ml(
    events: list[tuple[np.ndarray, np.ndarray]],
    n_models: int,
    max_iter: int = 500,
) -> np.ndarray:
    L = int(n_models)
    if not events:
        return np.ones(L) / L

    directed_choices = np.zeros((L, L), dtype=bool)
    for winners, losers in events:
        directed_choices[np.ix_(winners, losers)] = True
    if not is_strongly_connected(directed_choices):
        raise ValueError(
            "Bradley-Terry-Luce has no finite maximum-likelihood estimate "
            "because the directed choice graph is not strongly connected; "
            "use bradley_terry_luce_map for a regularized estimate."
        )

    def negative_log_likelihood(log_pi: np.ndarray) -> float:
        log_pi = log_pi - log_pi.mean()

        nll = 0.0
        for winners, losers in events:
            log_sum_losers = _logsumexp(log_pi[losers])
            nll -= float(np.sum(log_pi[winners]))
            nll += float(np.sum(np.logaddexp(log_pi[winners], log_sum_losers)))

        return float(nll)

    log_pi0 = np.zeros(L, dtype=float)
    result = minimize(
        negative_log_likelihood,
        log_pi0,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    if not result.success:
        raise RuntimeError(f"bradley_terry_luce optimization failed: {result.message}")

    log_pi_hat = result.x - result.x.mean()
    return np.exp(np.clip(log_pi_hat, -30.0, 30.0))


def _estimate_btl_map(
    events: list[tuple[np.ndarray, np.ndarray]],
    n_models: int,
    prior: Prior,
    max_iter: int = 500,
) -> np.ndarray:
    if type(prior) is UniformPrior:
        return _estimate_btl_ml(events, n_models=n_models, max_iter=max_iter)

    L = int(n_models)
    if not events:
        return np.ones(L) / L

    def negative_log_posterior(log_pi: np.ndarray) -> float:
        log_pi = log_pi - log_pi.mean()

        nll = 0.0
        for winners, losers in events:
            log_sum_losers = _logsumexp(log_pi[losers])
            nll -= float(np.sum(log_pi[winners]))
            nll += float(np.sum(np.logaddexp(log_pi[winners], log_sum_losers)))

        return float(nll + prior.penalty(log_pi))

    log_pi0 = np.zeros(L, dtype=float)
    result = minimize(
        negative_log_posterior,
        log_pi0,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    if not result.success:
        raise RuntimeError(
            f"bradley_terry_luce_map optimization failed: {result.message}"
        )

    log_pi_hat = result.x - result.x.mean()
    return np.exp(np.clip(log_pi_hat, -30.0, 30.0))


__all__ = [
    "plackett_luce",
    "plackett_luce_map",
    "davidson_luce",
    "davidson_luce_map",
    "bradley_terry_luce",
    "bradley_terry_luce_map",
]
