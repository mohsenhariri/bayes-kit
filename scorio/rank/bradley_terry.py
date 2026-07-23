"""

Paired-comparison (pairwise) probabilistic ranking models.

In Scorio's binary tensor setting :math:`R \\in \\{0,1\\}^{L \\times M \\times N}`,
each model pair ``(i, j)`` generates decisive wins and ties:

.. math::
    W_{ij}
    = \\sum_{m,n}
    \\mathbf{1}\\{R_{imn}=1,\\ R_{jmn}=0\\}, \\qquad
    T_{ij}
    = \\sum_{m,n}
    \\mathbf{1}\\{R_{imn}=R_{jmn}\\}.

Paired-comparison models assume latent positive strengths :math:`\\pi_i` and map
them to pairwise event probabilities. A common likelihood template is

.. math::
    \\log p(W,T \\mid \\pi, \\eta)
    = \\sum_{i<j}
    \\left[
      W_{ij}\\log p_{ij}
      + W_{ji}\\log p_{ji}
      + T_{ij}\\log p^{\\text{tie}}_{ij}
    \\right],

where :math:`\\eta` denotes tie-related parameter(s) when present.

Main families implemented here:

- **Bradley-Terry (BT)** (no explicit ties):

  .. math::
      \\Pr(i \\succ j) = \\frac{\\pi_i}{\\pi_i + \\pi_j}.

- **Davidson (1970)** tie extension with :math:`\\nu > 0`:

  .. math::
      \\Pr(i \\sim j)
      = \\frac{\\nu\\sqrt{\\pi_i\\pi_j}}
      {\\pi_i + \\pi_j + \\nu\\sqrt{\\pi_i\\pi_j}}.

- **Rao-Kupper (1967)** tie extension with :math:`\\kappa \\ge 1`:

  .. math::
      \\Pr(i \\succ j) = \\frac{\\pi_i}{\\pi_i + \\kappa\\pi_j}, \\quad
      \\Pr(i \\sim j)
      = \\frac{(\\kappa^2-1)\\pi_i\\pi_j}
      {(\\pi_i+\\kappa\\pi_j)(\\kappa\\pi_i+\\pi_j)}.

Each family has ML and MAP variants (MAP adds a prior penalty on the latent
log-strengths :math:`\\theta_i=\\log\\pi_i` for regularization and numerical
stability).
"""

import numpy as np
from scipy.optimize import minimize

from scorio.utils import rank_scores

from ._base import (
    average_equivalent_scores,
    build_pairwise_counts,
    build_pairwise_wins,
    is_strongly_connected,
    validate_input,
)
from ._types import RankMethod, RankResult
from .priors import (
    CauchyPrior,
    GaussianPrior,
    LaplacePrior,
    Prior,
    UniformPrior,
)


def _validate_max_iter(max_iter: int) -> int:
    """Validate optimizer iteration budget."""
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
        raise TypeError(f"max_iter must be an integer, got {type(max_iter).__name__}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be > 0, got {max_iter}")
    return int(max_iter)


def _validate_tie_strength(tie_strength: float) -> float:
    """Validate Rao-Kupper tie parameter kappa."""
    if isinstance(tie_strength, bool):
        raise TypeError(
            f"tie_strength must be a finite scalar >= 1.0, got {type(tie_strength).__name__}"
        )
    kappa = float(tie_strength)
    if not np.isfinite(kappa):
        raise ValueError("tie_strength must be finite.")
    if kappa < 1.0:
        raise ValueError("tie_strength must be >= 1.0 for Rao-Kupper")
    return kappa


def _coerce_prior(prior: Prior | float) -> Prior:
    """
    Normalize prior specification to a Prior instance.

    A numeric prior value is interpreted as Gaussian prior variance.
    """
    if isinstance(prior, bool):
        raise TypeError("prior must be a Prior object or positive finite float")

    if isinstance(prior, (int, float, np.integer, np.floating)):
        var = float(prior)
        if not np.isfinite(var) or var <= 0.0:
            raise ValueError("prior must be a positive finite scalar variance")
        return GaussianPrior(mean=0.0, var=var)

    if not isinstance(prior, Prior):
        raise TypeError(
            f"prior must be a Prior object or float, got {type(prior).__name__}"
        )

    return prior


def _prior_is_exchangeable(prior: Prior) -> bool:
    """Whether permuting model labels leaves the prior penalty unchanged."""
    return type(prior) in {GaussianPrior, LaplacePrior, CauchyPrior, UniformPrior}


def _require_optimizer_success(result, model_name: str) -> None:
    """Reject unfinished optimizer iterates as parameter estimates."""
    if not result.success:
        raise RuntimeError(f"{model_name} optimization failed: {result.message}")


def bradley_terry(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
) -> RankResult:
    """
    Rank models with Bradley-Terry maximum-likelihood strengths.

    Method context:
        Pairwise decisive outcomes are modeled with positive strengths
        :math:`\\pi_i`. Tied outcomes are ignored for BT-ML.

    References:
        Bradley, R. A., & Terry, M. E. (1952). Rank Analysis of Incomplete
        Block Designs: I. The Method of Paired Comparisons. Biometrika.
        https://doi.org/10.1093/biomet/39.3-4.324

        Hunter, D. R. (2004). MM algorithms for generalized Bradley-Terry
        models. The Annals of Statistics.
        https://doi.org/10.1214/aos/1079120141

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive maximum number of L-BFGS iterations.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns BT strengths
        ``pi`` (shape ``(L,)``).

    Notation:
        ``W_ij`` is the number of decisive outcomes where model ``i`` beats
        model ``j``.

    Formula:
        .. math::
            \\Pr(i \\succ j) = \\frac{\\pi_i}{\\pi_i + \\pi_j}

        .. math::
            \\log p(W\\mid\\pi) =
            \\sum_{i\\ne j} W_{ij}
            \\left[\\log \\pi_i - \\log(\\pi_i+\\pi_j)\\right]

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [1, 0, 0, 1],
        ...     [0, 1, 0, 0],
        ...     [0, 0, 1, 0],
        ... ])
        >>> ranks, scores = rank.bradley_terry(R, return_scores=True)
        >>> ranks.tolist()
        [1, 2, 2]
        >>> float(scores[0] > scores[1])
        1.0

    Notes:
        If there are no decisive outcomes at all, all strengths are returned
        equal because relative ability is unidentifiable. Otherwise, a finite
        ML estimate exists only when the directed win graph is strongly
        connected; separated data raises ``ValueError`` and should be fit with
        :func:`bradley_terry_map`.
    """
    R = validate_input(R)
    max_iter = _validate_max_iter(max_iter)
    wins = build_pairwise_wins(R)
    scores = _estimate_bt_ml(wins, max_iter=max_iter)
    scores = average_equivalent_scores(scores, R)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def bradley_terry_map(
    R: np.ndarray,
    prior: Prior | float = 1.0,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
) -> RankResult:
    """
    Rank models with Bradley-Terry MAP estimation.

    Method context:
        This method adds a prior penalty on centered log-strengths
        ``theta_i = log(pi_i)`` to regularize BT estimation.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        prior: Prior on log-strengths. If ``prior`` is a float, it is
            interpreted as Gaussian prior variance using
            :class:`scorio.rank.GaussianPrior` with mean 0. If ``prior`` is a
            ``Prior`` instance, its penalty is used directly.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive maximum number of L-BFGS iterations.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns MAP strengths ``pi``
        (shape ``(L,)``).

    Notation:
        ``theta_i = log(pi_i)``, and ``P(theta)`` denotes the prior.

    Formula:
        .. math::
            \\hat{\\theta}
            = \\arg\\min_{\\theta}
            \\left[
            -\\log p(W\\mid\\theta) + \\operatorname{penalty}(\\theta)
            \\right]

        .. math::
            \\hat{\\pi}_i = \\exp(\\hat{\\theta}_i)

    References:
        Caron, F., & Doucet, A. (2012). Efficient Bayesian inference for
        generalized Bradley-Terry models. Journal of Computational and
        Graphical Statistics.
        https://doi.org/10.1080/10618600.2012.638220

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> from scorio.rank import GaussianPrior
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> ranks, scores = rank.bradley_terry_map(R, prior=1.0, return_scores=True)
        >>> ranks.tolist()
        [1, 2]

        >>> prior = GaussianPrior(mean=0.0, var=0.5)
        >>> rank.bradley_terry_map(R, prior=prior).shape
        (2,)

    Notes:
        With informative priors, MAP can remain identifiable in settings where
        BT-ML has weak or degenerate decisive information.
    """
    R = validate_input(R)
    max_iter = _validate_max_iter(max_iter)
    prior = _coerce_prior(prior)

    wins = build_pairwise_wins(R)
    scores = _estimate_bt_map(wins, prior, max_iter=max_iter)
    if _prior_is_exchangeable(prior):
        scores = average_equivalent_scores(scores, R)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def bradley_terry_davidson(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
) -> RankResult:
    """
    Rank models with the Bradley-Terry-Davidson tie model (ML).

    Method context:
        Davidson extends BT by introducing a tie parameter ``nu > 0``.
        In this benchmark setting, ties correspond to ``(1,1)`` or ``(0,0)``
        outcomes for a model pair on the same question-trial event.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive maximum number of L-BFGS iterations.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns Davidson strengths ``pi``
        (shape ``(L,)``).

    Notation:
        ``W_ij`` are decisive win counts and ``T_ij`` tie counts.

    Formula:
        .. math::
            \\Pr(i\\succ j) =
            \\frac{\\pi_i}{\\pi_i+\\pi_j+\\nu\\sqrt{\\pi_i\\pi_j}},
            \\quad
            \\Pr(i\\sim j) =
            \\frac{\\nu\\sqrt{\\pi_i\\pi_j}}{\\pi_i+\\pi_j+\\nu\\sqrt{\\pi_i\\pi_j}}

    References:
        Davidson, R. R. (1970). On extending the Bradley-Terry model to
        accommodate ties in paired comparison experiments. Journal of the
        American Statistical Association.
        https://doi.org/10.1080/01621459.1970.10481082

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 0]],
        ...     [[1, 0], [0, 0]],
        ... ])
        >>> ranks, scores = rank.bradley_terry_davidson(R, return_scores=True)
        >>> ranks.tolist()
        [1, 2]

    Notes:
        If there are ties but no decisive outcomes, strengths are set equal
        because relative ability is not identified from ties alone.
    """
    R = validate_input(R)
    max_iter = _validate_max_iter(max_iter)
    wins, ties = build_pairwise_counts(R)
    scores = _estimate_bt_davidson(wins, ties, max_iter=max_iter)
    scores = average_equivalent_scores(scores, R)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def bradley_terry_davidson_map(
    R: np.ndarray,
    prior: Prior | float = 1.0,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
) -> RankResult:
    """
    Rank models with the Bradley-Terry-Davidson tie model (MAP).

    Method context:
        Adds a prior penalty on centered log-strengths on top of Davidson's tie
        likelihood.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        prior: Prior on log-strengths.
            - ``float``: interpreted as Gaussian prior variance with mean 0.
            - ``Prior`` instance: custom prior penalty.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive maximum number of L-BFGS iterations.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns Davidson-MAP strengths ``pi``
        (shape ``(L,)``).

    Notation:
        ``theta_i = log(pi_i)`` and ``nu`` is the Davidson tie parameter.

    Formula:
        .. math::
            \\hat{\\theta},\\hat{\\nu} =
            \\arg\\min_{\\theta,\\nu>0}
            \\left[
            -\\log p(W,T\\mid\\theta,\\nu)
            + \\operatorname{penalty}(\\theta)
            \\right]

    References:
        Davidson, R. R. (1970). On extending the Bradley-Terry model to
        accommodate ties in paired comparison experiments. Journal of the
        American Statistical Association.
        https://doi.org/10.1080/01621459.1970.10481082

        Caron, F., & Doucet, A. (2012). Efficient Bayesian inference for
        generalized Bradley-Terry models.
        https://doi.org/10.1080/10618600.2012.638220

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 0]],
        ...     [[1, 0], [0, 0]],
        ... ])
        >>> ranks, scores = rank.bradley_terry_davidson_map(
        ...     R, prior=1.0, return_scores=True
        ... )
        >>> ranks.tolist()
        [1, 2]
    """
    R = validate_input(R)
    max_iter = _validate_max_iter(max_iter)
    prior = _coerce_prior(prior)

    wins, ties = build_pairwise_counts(R)
    scores = _estimate_bt_davidson_map(wins, ties, prior, max_iter=max_iter)
    if _prior_is_exchangeable(prior):
        scores = average_equivalent_scores(scores, R)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def rao_kupper(
    R: np.ndarray,
    tie_strength: float = 1.1,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
) -> RankResult:
    """
    Rank models with the Rao-Kupper tie model (ML).

    Method context:
        Rao-Kupper augments paired comparison with a threshold parameter
        ``kappa >= 1`` controlling tie prevalence. In this API, ``kappa`` is a
        fixed hyperparameter (``tie_strength``), not estimated.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        tie_strength: Rao-Kupper parameter ``kappa >= 1``.
            ``kappa=1`` reduces to no-tie BT.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive maximum number of L-BFGS iterations.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns Rao-Kupper strengths ``pi``
        (shape ``(L,)``).

    Notation:
        ``W_ij`` are decisive win counts and ``T_ij`` tie counts.

    Formula:
        .. math::
            \\Pr(i\\succ j) = \\frac{\\pi_i}{\\pi_i + \\kappa\\pi_j}, \\quad
            \\Pr(j\\succ i) = \\frac{\\pi_j}{\\kappa\\pi_i + \\pi_j}

        .. math::
            \\Pr(i\\sim j)=
            \\frac{(\\kappa^2-1)\\pi_i\\pi_j}
            {(\\pi_i+\\kappa\\pi_j)(\\kappa\\pi_i+\\pi_j)}

    References:
        Rao, P. V., & Kupper, L. L. (1967). Ties in paired-comparison
        experiments: A generalization of the Bradley-Terry model.
        Journal of the American Statistical Association.
        https://doi.org/10.1080/01621459.1967.10482901

        Hunter, D. R. (2004). MM algorithms for generalized Bradley-Terry
        models. The Annals of Statistics.
        https://doi.org/10.1214/aos/1079120141

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 0]],
        ...     [[1, 0], [0, 0]],
        ... ])
        >>> ranks, scores = rank.rao_kupper(R, tie_strength=1.1, return_scores=True)
        >>> ranks.tolist()
        [1, 2]

    Notes:
        If ``tie_strength=1`` and tie counts are present, the model is
        inconsistent and raises ``ValueError``.
    """
    R = validate_input(R)
    max_iter = _validate_max_iter(max_iter)
    tie_strength = _validate_tie_strength(tie_strength)
    wins, ties = build_pairwise_counts(R)
    scores = _estimate_rao_kupper_ml(wins, ties, tie_strength, max_iter=max_iter)
    scores = average_equivalent_scores(scores, R)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def rao_kupper_map(
    R: np.ndarray,
    tie_strength: float = 1.1,
    prior: Prior | float = 1.0,
    method: RankMethod = "competition",
    return_scores: bool = False,
    max_iter: int = 500,
) -> RankResult:
    """
    Rank models with the Rao-Kupper tie model (MAP).

    Method context:
        Adds a prior penalty on centered log-strengths to Rao-Kupper
        likelihood while keeping ``kappa`` fixed.

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        tie_strength: Rao-Kupper parameter ``kappa >= 1``.
        prior: Prior on log-strengths.
            - ``float``: interpreted as Gaussian prior variance with mean 0.
            - ``Prior`` instance: custom prior penalty.
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, scores)``.
        max_iter: Positive maximum number of L-BFGS iterations.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns Rao-Kupper MAP strengths
        ``pi`` (shape ``(L,)``).

    Notation:
        ``theta_i = log(pi_i)`` and ``kappa`` is fixed.

    Formula:
        .. math::
            \\hat{\\theta}
            = \\arg\\min_{\\theta}
            \\left[
            -\\log p(W,T\\mid\\theta,\\kappa)
            + \\operatorname{penalty}(\\theta)
            \\right]

    References:
        Rao, P. V., & Kupper, L. L. (1967). Ties in paired-comparison
        experiments: A generalization of the Bradley-Terry model.
        Journal of the American Statistical Association.
        https://doi.org/10.1080/01621459.1967.10482901

        Caron, F., & Doucet, A. (2012). Efficient Bayesian inference for
        generalized Bradley-Terry models.
        https://doi.org/10.1080/10618600.2012.638220
    """
    R = validate_input(R)
    max_iter = _validate_max_iter(max_iter)
    tie_strength = _validate_tie_strength(tie_strength)
    prior = _coerce_prior(prior)

    wins, ties = build_pairwise_counts(R)
    scores = _estimate_rao_kupper_map(
        wins, ties, tie_strength, prior, max_iter=max_iter
    )
    if _prior_is_exchangeable(prior):
        scores = average_equivalent_scores(scores, R)
    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


def _estimate_bt_ml(wins: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """Estimate Bradley-Terry strengths via maximum likelihood."""
    n = wins.shape[0]

    if float(np.sum(wins)) <= 0.0:
        return np.ones(n, dtype=float)
    if not is_strongly_connected(wins > 0):
        raise ValueError(
            "Bradley-Terry has no finite maximum-likelihood estimate because "
            "the directed win graph is not strongly connected; use "
            "bradley_terry_map for a regularized estimate."
        )

    def negative_log_likelihood(log_pi):
        log_pi = log_pi - log_pi.mean()
        log_pi_capped = np.clip(log_pi, -30.0, 30.0)
        pi = np.exp(log_pi_capped)

        nll = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                n_ij = wins[i, j]
                if n_ij > 0:
                    nll -= n_ij * (log_pi_capped[i] - np.log(pi[i] + pi[j]))
        return float(nll)

    # Initialize
    total_wins = wins.sum(axis=1)
    total_wins = np.maximum(total_wins, 1)
    log_pi_init = np.log(total_wins / total_wins.sum())

    result = minimize(
        negative_log_likelihood,
        log_pi_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    _require_optimizer_success(result, "bradley_terry")

    log_pi = result.x - result.x.mean()
    return np.exp(np.clip(log_pi, -30.0, 30.0))


def _estimate_bt_map(wins: np.ndarray, prior: Prior, max_iter: int = 500) -> np.ndarray:
    """Estimate Bradley-Terry strengths via MAP with configurable prior."""
    if type(prior) is UniformPrior:
        return _estimate_bt_ml(wins, max_iter=max_iter)

    n = wins.shape[0]
    no_decisive_outcomes = float(np.sum(wins)) <= 0.0

    if (
        no_decisive_outcomes
        and isinstance(prior, GaussianPrior)
        and float(prior.mean) == 0.0
    ):
        return np.ones(n, dtype=float)

    def negative_log_posterior(log_pi):
        log_pi = log_pi - log_pi.mean()
        log_pi_capped = np.clip(log_pi, -30.0, 30.0)
        pi = np.exp(log_pi_capped)

        nll = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                n_ij = wins[i, j]
                if n_ij > 0:
                    nll -= n_ij * (log_pi_capped[i] - np.log(pi[i] + pi[j]))

        # Prior penalty (negative log-prior)
        prior_penalty = prior.penalty(log_pi)

        return float(nll + prior_penalty)

    # Initialize
    total_wins = wins.sum(axis=1)
    total_wins = np.maximum(total_wins, 1)
    log_pi_init = np.log(total_wins / total_wins.sum())

    result = minimize(
        negative_log_posterior,
        log_pi_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    _require_optimizer_success(result, "bradley_terry_map")

    log_pi = result.x - result.x.mean()
    scores = np.exp(np.clip(log_pi, -30.0, 30.0))
    if no_decisive_outcomes and float(np.max(scores) - np.min(scores)) <= 1e-5:
        return np.ones(n, dtype=float)
    return scores


def _estimate_bt_davidson(
    wins: np.ndarray, ties: np.ndarray, max_iter: int = 500
) -> np.ndarray:
    """Estimate Bradley-Terry-Davidson strengths with tie parameter."""
    n = wins.shape[0]
    eps = 1e-10

    if float(np.sum(wins)) <= 0.0:
        return np.ones(n, dtype=float)
    existence_graph = (wins > 0) | (ties > 0)
    if not is_strongly_connected(existence_graph):
        raise ValueError(
            "Bradley-Terry-Davidson has no finite maximum-likelihood strength "
            "estimate because its win/tie graph is not strongly connected; use "
            "bradley_terry_davidson_map with a proper prior."
        )

    def negative_log_likelihood(params):
        log_pi = params[:n]
        log_theta = params[n]

        log_pi = log_pi - log_pi.mean()
        log_pi_capped = np.clip(log_pi, -30.0, 30.0)
        pi = np.exp(log_pi_capped)
        theta = np.exp(np.clip(log_theta, -10, 10))

        nll = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                n_ij = wins[i, j]
                n_ji = wins[j, i]
                n_tie = ties[i, j]

                denom = pi[i] + pi[j] + theta * np.sqrt(pi[i] * pi[j])
                denom = max(denom, eps)

                if n_ij > 0:
                    nll -= n_ij * np.log(max(pi[i] / denom, eps))
                if n_ji > 0:
                    nll -= n_ji * np.log(max(pi[j] / denom, eps))
                if n_tie > 0:
                    tie_prob = theta * np.sqrt(pi[i] * pi[j]) / denom
                    nll -= n_tie * np.log(max(tie_prob, eps))

        return float(nll)

    # Initialize
    total_wins = wins.sum(axis=1)
    total_wins = np.maximum(total_wins, 1)
    log_pi_init = np.log(total_wins / total_wins.sum())
    log_theta_init = 0.0

    params_init = np.concatenate([log_pi_init, [log_theta_init]])

    result = minimize(
        negative_log_likelihood,
        params_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    _require_optimizer_success(result, "bradley_terry_davidson")

    log_pi = result.x[:n] - result.x[:n].mean()
    return np.exp(np.clip(log_pi, -30.0, 30.0))


def _estimate_bt_davidson_map(
    wins: np.ndarray, ties: np.ndarray, prior: Prior, max_iter: int = 500
) -> np.ndarray:
    """Estimate Bradley-Terry-Davidson strengths with tie parameter via MAP."""
    if type(prior) is UniformPrior:
        return _estimate_bt_davidson(wins, ties, max_iter=max_iter)

    n = wins.shape[0]
    eps = 1e-10
    no_decisive_outcomes = float(np.sum(wins)) <= 0.0

    if (
        no_decisive_outcomes
        and isinstance(prior, GaussianPrior)
        and float(prior.mean) == 0.0
    ):
        return np.ones(n, dtype=float)

    def negative_log_posterior(params):
        log_pi = params[:n]
        log_theta = params[n]

        log_pi = log_pi - log_pi.mean()
        log_pi_capped = np.clip(log_pi, -30.0, 30.0)
        pi = np.exp(log_pi_capped)
        theta = np.exp(np.clip(log_theta, -10, 10))

        nll = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                n_ij = wins[i, j]
                n_ji = wins[j, i]
                n_tie = ties[i, j]

                denom = pi[i] + pi[j] + theta * np.sqrt(pi[i] * pi[j])
                denom = max(denom, eps)

                if n_ij > 0:
                    nll -= n_ij * np.log(max(pi[i] / denom, eps))
                if n_ji > 0:
                    nll -= n_ji * np.log(max(pi[j] / denom, eps))
                if n_tie > 0:
                    tie_prob = theta * np.sqrt(pi[i] * pi[j]) / denom
                    nll -= n_tie * np.log(max(tie_prob, eps))

        # Prior penalty on log-strengths (negative log-prior)
        prior_penalty = prior.penalty(log_pi)

        return float(nll + prior_penalty)

    # Initialize
    total_wins = wins.sum(axis=1)
    total_wins = np.maximum(total_wins, 1)
    log_pi_init = np.log(total_wins / total_wins.sum())
    log_theta_init = 0.0

    params_init = np.concatenate([log_pi_init, [log_theta_init]])

    result = minimize(
        negative_log_posterior,
        params_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    _require_optimizer_success(result, "bradley_terry_davidson_map")

    log_pi = result.x[:n] - result.x[:n].mean()
    scores = np.exp(np.clip(log_pi, -30.0, 30.0))
    if no_decisive_outcomes and float(np.max(scores) - np.min(scores)) <= 1e-5:
        return np.ones(n, dtype=float)
    return scores


def _estimate_rao_kupper_ml(
    wins: np.ndarray,
    ties: np.ndarray,
    tie_strength: float,
    max_iter: int = 500,
) -> np.ndarray:
    n = wins.shape[0]
    eps = 1e-12

    kappa = _validate_tie_strength(tie_strength)

    total_ties = float(np.triu(ties, 1).sum())
    if kappa == 1.0 and total_ties > 0:
        raise ValueError("tie_strength=1.0 implies no ties, but tie counts exist")
    if float(np.sum(wins)) <= 0.0:
        return np.ones(n, dtype=float)
    existence_graph = (wins > 0) | (ties > 0)
    if not is_strongly_connected(existence_graph):
        raise ValueError(
            "Rao-Kupper has no finite maximum-likelihood strength estimate "
            "because its win/tie graph is not strongly connected; use "
            "rao_kupper_map with a proper prior."
        )

    def negative_log_likelihood(log_pi: np.ndarray) -> float:
        log_pi = log_pi - log_pi.mean()
        pi = np.exp(np.clip(log_pi, -30, 30))

        nll = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                n_ij = wins[i, j]
                n_ji = wins[j, i]
                n_tie = ties[i, j]

                denom_ij = pi[i] + kappa * pi[j]
                denom_ji = kappa * pi[i] + pi[j]

                p_ij = max(pi[i] / denom_ij, eps)
                p_ji = max(pi[j] / denom_ji, eps)

                if kappa > 1.0:
                    p_tie = (
                        (kappa * kappa - 1.0) * pi[i] * pi[j] / (denom_ij * denom_ji)
                    )
                    p_tie = max(p_tie, eps)
                else:
                    p_tie = 0.0

                if n_ij > 0:
                    nll -= n_ij * np.log(p_ij)
                if n_ji > 0:
                    nll -= n_ji * np.log(p_ji)
                if n_tie > 0:
                    if kappa == 1.0:
                        return float("inf")
                    nll -= n_tie * np.log(p_tie)

        return float(nll)

    log_pi0 = np.zeros(n, dtype=float)
    result = minimize(
        negative_log_likelihood,
        log_pi0,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    _require_optimizer_success(result, "rao_kupper")

    log_pi = result.x - result.x.mean()
    return np.exp(np.clip(log_pi, -30.0, 30.0))


def _estimate_rao_kupper_map(
    wins: np.ndarray,
    ties: np.ndarray,
    tie_strength: float,
    prior: Prior,
    max_iter: int = 500,
) -> np.ndarray:
    n = wins.shape[0]
    eps = 1e-12

    if type(prior) is UniformPrior:
        return _estimate_rao_kupper_ml(wins, ties, tie_strength, max_iter=max_iter)

    kappa = _validate_tie_strength(tie_strength)

    total_ties = float(np.triu(ties, 1).sum())
    if kappa == 1.0 and total_ties > 0:
        raise ValueError("tie_strength=1.0 implies no ties, but tie counts exist")
    no_decisive_outcomes = float(np.sum(wins)) <= 0.0

    if (
        no_decisive_outcomes
        and isinstance(prior, GaussianPrior)
        and float(prior.mean) == 0.0
    ):
        return np.ones(n, dtype=float)

    def negative_log_posterior(log_pi: np.ndarray) -> float:
        log_pi = log_pi - log_pi.mean()
        pi = np.exp(np.clip(log_pi, -30, 30))

        nll = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                n_ij = wins[i, j]
                n_ji = wins[j, i]
                n_tie = ties[i, j]

                denom_ij = pi[i] + kappa * pi[j]
                denom_ji = kappa * pi[i] + pi[j]

                p_ij = max(pi[i] / denom_ij, eps)
                p_ji = max(pi[j] / denom_ji, eps)

                if kappa > 1.0:
                    p_tie = (
                        (kappa * kappa - 1.0) * pi[i] * pi[j] / (denom_ij * denom_ji)
                    )
                    p_tie = max(p_tie, eps)
                else:
                    p_tie = 0.0

                if n_ij > 0:
                    nll -= n_ij * np.log(p_ij)
                if n_ji > 0:
                    nll -= n_ji * np.log(p_ji)
                if n_tie > 0:
                    if kappa == 1.0:
                        return float("inf")
                    nll -= n_tie * np.log(p_tie)

        return float(nll + prior.penalty(log_pi))

    log_pi0 = np.zeros(n, dtype=float)
    result = minimize(
        negative_log_posterior,
        log_pi0,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    _require_optimizer_success(result, "rao_kupper_map")

    log_pi = result.x - result.x.mean()
    scores = np.exp(np.clip(log_pi, -30.0, 30.0))
    if no_decisive_outcomes and float(np.max(scores) - np.min(scores)) <= 1e-5:
        return np.ones(n, dtype=float)
    return scores


__all__ = [
    "bradley_terry",
    "bradley_terry_map",
    "bradley_terry_davidson",
    "bradley_terry_davidson_map",
    "rao_kupper",
    "rao_kupper_map",
]
