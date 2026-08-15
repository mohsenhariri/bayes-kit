r"""Max-reward metrics and uncertainty estimators for categorical outcomes.

``Max@k`` is the continuous-reward generalization of Pass@k: instead of asking
whether at least one sampled response is correct, it scores the best response
among ``k`` sampled traces according to a user-specified reward scale.

The point estimator implemented here matches the appendix evaluation formula in
Bagirov et al. (2025), "The Best of N Worlds: Aligning Reinforcement Learning
with Best-of-N Sampling via max@k Optimization" (Appendix C.1 / Listing 1,
arXiv:2510.23393, https://arxiv.org/abs/2510.23393). This module adapts that
estimator to ``scorio``'s categorical outcome representation via ``R`` plus a
reward map ``w``.

The companion ``*_ci`` functions are a ``scorio`` Bayesian extension. They use
the same grouped-Dirichlet posterior model as :func:`~scorio.eval.bayes` and
are not part of the paper above.

Methods
-------
- ``max_at_k``: expected best reward among ``k`` sampled traces.

Each metric has a companion ``*_ci`` function that returns
``(mu, sigma, lo, hi)`` under the Bayesian uncertainty model used here.
"""

import math

import numpy as np
from scipy.stats import hypergeom

from ._categorical import (
    CategoricalBank,
    _scaled_rewards,
    observed_mean,
    prepare_categorical_bank,
)
from ._inputs import validate_finite_k, validate_latent_k
from .bayes import bayes_ci
from .utils import normal_credible_interval


def _hypergeom_any_above(
    cumulative_counts: np.ndarray,
    trial_count: int,
    k: int,
) -> np.ndarray:
    """Probability that at least one draw exceeds a reward threshold."""
    above_counts = trial_count - cumulative_counts
    probabilities = hypergeom.sf(0, trial_count, above_counts, k)
    return np.asarray(probabilities, dtype=float)


def _log_beta_power_moment(alpha: float, beta: float, power: int) -> float:
    """Return ``log(E[p**power])`` without subtracting large log-gammas."""
    if power == 0:
        return 0.0

    offsets = np.arange(power, dtype=float)
    denominators = alpha + beta + offsets
    complement = beta / denominators
    terms = np.empty(power, dtype=float)
    near_one = complement <= 0.5
    terms[near_one] = np.log1p(-complement[near_one])
    terms[~near_one] = np.log(alpha + offsets[~near_one]) - np.log(
        denominators[~near_one]
    )
    return math.fsum(float(term) for term in terms)


def _log_dirichlet_nested_cumulative_moment(
    total: float, a: float, b: float, k: int
) -> float:
    """Return ``log(E[X^k (X+Y)^k])`` for a Dirichlet partition.

    Writing ``U = X + Y`` and ``V = X / (X + Y)``, Dirichlet neutrality gives
    independent ``U ~ Beta(a+b, total-a-b)`` and ``V ~ Beta(a, b)``. Hence
    ``E[X^k U^k] = E[U^(2k)] E[V^k]``.
    """
    if b <= 0.0:
        raise ValueError("b must be > 0 for nested cumulative moments")
    return _log_beta_power_moment(
        a + b,
        total - a - b,
        2 * k,
    ) + _log_beta_power_moment(a, b, k)


def _covariance_from_log_moments(
    log_cross_moment: float,
    log_left_mean: float,
    log_right_mean: float,
) -> float:
    """Subtract two positive moments without cancellation."""
    log_product = log_left_mean + log_right_mean
    difference = log_product - log_cross_moment
    tolerance = (
        64.0
        * np.finfo(float).eps
        * max(
            1.0,
            abs(log_product),
            abs(log_cross_moment),
        )
    )
    if difference > 0.0:
        if difference <= tolerance:
            difference = 0.0
        else:
            raise FloatingPointError(
                "nested cumulative covariance is materially negative"
            )
    return math.exp(log_cross_moment) * -math.expm1(difference)


def max_at_k(R: np.ndarray, k: int, w: np.ndarray | None = None) -> float:
    r"""
    Max@k: expected best reward among ``k`` sampled traces.

    When ``w = [0, 1]``, Max@k reduces exactly to Pass@k. More generally, the
    reward vector ``w`` maps categorical outcomes to arbitrary real-valued
    scores, and Max@k averages the best score obtainable from a subset of size
    ``k``.

    References:
        - Bagirov, F., et al. (2025). The Best of N Worlds: Aligning
          Reinforcement Learning with Best-of-N Sampling via max@k
          Optimization. *arXiv:2510.23393*.
          https://arxiv.org/abs/2510.23393
          The finite-sample estimator below matches Appendix C.1 / Listing 1.
        - Walder, C., & Karkhanis, D. (2025). Pass@K Policy Optimization:
          Solving Harder Reinforcement Learning Problems. *arXiv:2505.15201*.

    Args:
        R: :math:`M \times N` categorical outcome matrix with integer entries
           in :math:`\{0, \ldots, C\}`.
        k: Number of selected samples, with ``1 <= k <= N``.
        w: Optional reward vector of shape ``(C+1,)``. If omitted, ``R`` must
           be binary and ``w = [0, 1]`` is used.

    Returns:
        float: Average Max@k score across prompts.

    Formula:
        Let :math:`g_{\alpha 1} \le \cdots \le g_{\alpha N}` denote the
        sorted rewards for prompt :math:`\alpha`. Then the unbiased finite-
        sample estimator is

        .. math::

            \mathrm{Max@k}_\alpha = \frac{1}{\binom{N}{k}}
            \sum_{i=k}^{N} \binom{i-1}{k-1} g_{\alpha i}.

        The dataset-level metric is the average across prompts:

        .. math::

            \mathrm{Max@k} = \frac{1}{M}
            \sum_{\alpha=1}^{M} \mathrm{Max@k}_\alpha

    Examples:
        Binary (reduces to Pass@k):

        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(max_at_k(R, 2), 6)
        0.95

        Weighted categorical rewards:

        >>> R = np.array([[0, 1, 2, 2, 1],
        ...               [1, 1, 0, 2, 2]])
        >>> w = np.array([0.0, 0.5, 1.0])
        >>> round(max_at_k(R, 2, w=w), 6)
        0.85
    """
    bank = prepare_categorical_bank(R, w=w)
    k = validate_finite_k(bank.trial_count, k)
    if k == 1:
        return observed_mean(bank)

    grouped_counts, levels = bank.grouped_observed_counts()
    if levels.size == 1:
        return float(levels[0])

    cumulative_counts = np.cumsum(grouped_counts, axis=1)[:, :-1]
    any_above = _hypergeom_any_above(
        cumulative_counts,
        bank.trial_count,
        k,
    )
    offset, reward_scale, normalized_levels = _scaled_rewards(levels)
    normalized_scores = normalized_levels[0] + any_above @ np.diff(normalized_levels)
    return float(offset + reward_scale * np.mean(normalized_scores))


def _max_at_k_bayes(
    bank: CategoricalBank,
    k: int,
) -> tuple[float, float]:
    """Posterior mean/std for Max@k under a grouped Dirichlet posterior."""
    gamma, levels = bank.grouped_posterior_counts()
    M = bank.question_count
    L = int(levels.size)
    total = float(bank.category_count + bank.prior_trial_count + bank.trial_count)

    # The posterior moments describe the latent distribution, so k is not
    # restricted by the observed sample size once the posterior is defined.

    if L == 1:
        mu = float(levels[0])
        return mu, 0.0

    offset, reward_scale, normalized_levels = _scaled_rewards(levels)
    gaps = np.diff(normalized_levels)

    means = np.empty(M, dtype=float)
    row_sigmas = np.empty(M, dtype=float)

    for row in range(M):
        gamma_row = gamma[row]
        cum = np.cumsum(gamma_row)[:-1]  # A_l parameters for l = 1..L-1

        log_e_ak = np.empty(L - 1, dtype=float)
        log_e_a2k = np.empty(L - 1, dtype=float)
        for idx in range(L - 1):
            a = float(cum[idx])
            b = total - a
            log_e_ak[idx] = _log_beta_power_moment(a, b, k)
            log_e_a2k[idx] = _log_beta_power_moment(a, b, 2 * k)

        exceedance_probabilities = -np.expm1(log_e_ak)
        mean_increment = float(np.dot(gaps, exceedance_probabilities))

        covariance = np.empty((L - 1, L - 1), dtype=float)
        for i in range(L - 1):
            covariance[i, i] = _covariance_from_log_moments(
                log_e_a2k[i],
                log_e_ak[i],
                log_e_ak[i],
            )
            for j in range(i + 1, L - 1):
                a = float(cum[i])
                b = float(cum[j] - cum[i])
                log_cross = _log_dirichlet_nested_cumulative_moment(
                    total,
                    a,
                    b,
                    k,
                )
                value = _covariance_from_log_moments(
                    log_cross,
                    log_e_ak[i],
                    log_e_ak[j],
                )
                covariance[i, j] = value
                covariance[j, i] = value

        normalized_variance = float(gaps @ covariance @ gaps)
        if normalized_variance < 0.0:
            tolerance = 64.0 * np.finfo(float).eps
            if normalized_variance >= -tolerance:
                normalized_variance = 0.0
            else:
                raise FloatingPointError(
                    "Max@k posterior variance is materially negative"
                )

        normalized_mean = float(
            np.clip(
                normalized_levels[0] + mean_increment,
                normalized_levels[0],
                normalized_levels[-1],
            )
        )
        means[row] = offset + reward_scale * normalized_mean
        row_sigmas[row] = reward_scale * math.sqrt(normalized_variance)

    mu = float(np.mean(means))
    sigma_scale = float(np.max(row_sigmas))
    if sigma_scale == 0.0:
        sigma = 0.0
    else:
        sigma = float(
            sigma_scale * math.sqrt(float(np.sum((row_sigmas / sigma_scale) ** 2))) / M
        )
    return mu, sigma


def max_at_k_ci(
    R: np.ndarray,
    k: int,
    w: np.ndarray | None = None,
    R0: np.ndarray | None = None,
    confidence: float = 0.95,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float, float, float]:
    r"""
    Bayesian posterior summary for :func:`max_at_k`.

    The posterior uses the same Dirichlet-plus-one construction as
    :func:`~scorio.eval.bayes`. When ``k = 1``, ``Max@1`` reduces to the usual
    single-draw expected score, so this function agrees with
    :func:`~scorio.eval.bayes_ci`.

    This uncertainty model is a ``scorio`` extension. Bagirov et al. (2025)
    define the finite-sample max@k point estimator, but do not derive these
    Bayesian credible intervals.

    Args:
        R: :math:`M \times N` categorical outcome matrix with integer entries
           in :math:`\{0, \ldots, C\}`.
        k: Selection count. The posterior target is defined for any integer
           ``k >= 1``; ``k = 1`` matches :func:`~scorio.eval.bayes_ci`.
        w: Optional reward vector of shape ``(C+1,)``. If omitted, ``R`` must
           be binary and ``w = [0, 1]`` is used.
        R0: Optional :math:`M \times D` matrix of prior outcomes.
        confidence: Credibility level for the normal-approximation interval.
        bounds: Optional ``(lo, hi)`` clipping bounds. If omitted, the interval
            is clipped to the minimum and maximum reward levels in ``w``.

    Returns:
        tuple[float, float, float, float]:
            :math:`(\mu,\; \sigma,\; \text{lo},\; \text{hi})`

    Formula:
        Let :math:`r_1 < \cdots < r_L` be the unique reward levels and
        :math:`A_{\alpha \ell}` the posterior cumulative probability of
        obtaining reward at most :math:`r_\ell` for prompt :math:`\alpha`.
        Then the per-prompt latent target is

        .. math::

            g_\alpha = r_L - \sum_{\ell=1}^{L-1}
            (r_{\ell+1} - r_\ell) A_{\alpha \ell}^k

        and posterior moments are computed in closed form under the grouped
        Dirichlet posterior.

    Examples:
        Binary:

        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> mu, sigma, lo, hi = max_at_k_ci(R, 2)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.839286, 0.097263, 0.6487, 1.0)

        Weighted categorical rewards:

        >>> R = np.array([[0, 1, 2, 2, 1],
        ...               [1, 1, 0, 2, 2]])
        >>> w = np.array([0.0, 0.5, 1.0])
        >>> mu, sigma, lo, hi = max_at_k_ci(R, 2, w=w)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.75, 0.08812, 0.5773, 0.9227)
    """
    k = validate_latent_k(k)
    if k == 1:
        return bayes_ci(R, w=w, R0=R0, confidence=confidence, bounds=bounds)

    bank = prepare_categorical_bank(R, w=w, R0=R0)
    mu, sigma = _max_at_k_bayes(bank, k)
    if bounds is None:
        bounds = (float(np.min(bank.weights)), float(np.max(bank.weights)))
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


__all__ = [
    "max_at_k",
    "max_at_k_ci",
]
