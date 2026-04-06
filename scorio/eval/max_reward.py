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
the same grouped-Dirichlet posterior model as :func:`scorio.eval.bayes` and are
not part of the paper above.

Methods
-------
- ``max_at_k``: expected best reward among ``k`` sampled traces.

Each metric has a companion ``*_ci`` function that returns
``(mu, sigma, lo, hi)`` under the Bayesian uncertainty model used here.
"""

import math

import numpy as np
from scipy.special import comb, gammaln, logsumexp

from .bayes import bayes_ci
from .pass_at_k import _beta_ratio
from .utils import _as_2d_int_matrix, _validate_matrix_range, normal_credible_interval


def _prepare_categorical_input(
    R: np.ndarray,
    w: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize R, w, and R0 for weighted categorical metrics."""
    Rm = _as_2d_int_matrix(R)

    if w is None:
        unique_vals = np.unique(Rm)
        is_binary = len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1]))
        if not is_binary:
            unique_str = ", ".join(map(str, sorted(unique_vals)))
            raise ValueError(
                f"R contains more than 2 unique values ({unique_str}), so weight vector 'w' must be provided. "
                f"Please specify a weight vector of length {len(unique_vals)} to map each category to a score."
            )
        wv = np.array([0.0, 1.0], dtype=float)
    else:
        wv = np.asarray(w, dtype=float)

    M, _ = Rm.shape
    C = int(wv.size - 1)
    _validate_matrix_range(Rm, 0, C, "R")

    if R0 is None:
        R0m = np.zeros((M, 0), dtype=int)
    else:
        R0m = np.asarray(R0, dtype=int)
        if R0m.ndim == 1:
            R0m = R0m.reshape(M, -1)
        if R0m.shape[0] != M:
            raise ValueError("R0 must have the same number of rows (M) as R.")
        _validate_matrix_range(R0m, 0, C, "R0")

    return Rm, wv, R0m


def _validate_k(N: int, k: int) -> None:
    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")


def _row_bincount(A: np.ndarray, length: int) -> np.ndarray:
    """Count occurrences of 0..length-1 in each row of A."""
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], length), dtype=int)
    out = np.zeros((A.shape[0], length), dtype=int)
    rows = np.repeat(np.arange(A.shape[0]), A.shape[1])
    np.add.at(out, (rows, A.ravel()), 1)
    return out


def _grouped_posterior_params(
    R: np.ndarray,
    w: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grouped Dirichlet posterior parameters and unique reward levels."""
    Rm, wv, R0m = _prepare_categorical_input(R, w=w, R0=R0)
    C = int(wv.size - 1)

    unique_levels, inverse = np.unique(wv, return_inverse=True)
    L = int(unique_levels.size)

    n_counts = _row_bincount(Rm, C + 1)
    n0_counts = _row_bincount(R0m, C + 1) + 1
    alpha_cat = n_counts + n0_counts

    gamma = np.zeros((Rm.shape[0], L), dtype=float)
    for cat in range(C + 1):
        gamma[:, inverse[cat]] += alpha_cat[:, cat]

    return gamma, unique_levels


def _dirichlet_nested_cumulative_moment(
    total: float, a: float, b: float, k: int
) -> float:
    """E[X^k (X+Y)^k] for X,Y from a 3-part Dirichlet partition.

    Here ``X`` has parameter ``a``, ``Y`` has parameter ``b``, and the omitted
    remainder has parameter ``total - a - b``. The formula follows from the
    multinomial expansion of ``(X+Y)^k`` and Dirichlet raw moments.
    """
    if b <= 0.0:
        raise ValueError("b must be > 0 for nested cumulative moments")
    r = np.arange(k + 1, dtype=float)
    log_terms = (
        gammaln(k + 1.0)
        - gammaln(r + 1.0)
        - gammaln(k - r + 1.0)
        + gammaln(a + k + r)
        - gammaln(a)
        + gammaln(b + k - r)
        - gammaln(b)
        - (gammaln(total + 2.0 * k) - gammaln(total))
    )
    return float(np.exp(logsumexp(log_terms)))


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
    Rm, wv, _ = _prepare_categorical_input(R, w=w, R0=None)
    _, N = Rm.shape
    _validate_k(N, k)

    rewards = wv[Rm]
    coeff = comb(np.arange(k - 1, N, dtype=float), k - 1) / comb(N, k)

    vals = np.empty(Rm.shape[0], dtype=float)
    for i in range(Rm.shape[0]):
        sorted_rewards = np.sort(rewards[i])
        vals[i] = float(np.dot(coeff, sorted_rewards[k - 1 :]))
    return float(np.mean(vals))


def _max_at_k_bayes(
    R: np.ndarray,
    k: int,
    w: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Posterior mean/std for Max@k under a grouped Dirichlet posterior."""
    gamma, levels = _grouped_posterior_params(R, w=w, R0=R0)
    M = gamma.shape[0]
    L = gamma.shape[1]
    total = float(np.sum(gamma[0]))

    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    # The posterior moments describe the latent distribution, so k is not
    # restricted by the observed sample size once the posterior is defined.

    if L == 1:
        mu = float(levels[0])
        return mu, 0.0, levels

    gaps = np.diff(levels)
    top = float(levels[-1])

    means = np.empty(M, dtype=float)
    vars_ = np.empty(M, dtype=float)

    for row in range(M):
        gamma_row = gamma[row]
        cum = np.cumsum(gamma_row)[:-1]  # A_l parameters for l = 1..L-1

        e_ak = np.empty(L - 1, dtype=float)
        e_a2k = np.empty(L - 1, dtype=float)
        for idx in range(L - 1):
            a = float(cum[idx])
            b = total - a
            e_ak[idx] = _beta_ratio(a, b, k, 0)
            e_a2k[idx] = _beta_ratio(a, b, 2 * k, 0)

        m = top - float(np.dot(gaps, e_ak))

        cross = np.empty((L - 1, L - 1), dtype=float)
        for i in range(L - 1):
            cross[i, i] = e_a2k[i]
            for j in range(i + 1, L - 1):
                a = float(cum[i])
                b = float(cum[j] - cum[i])
                moment = _dirichlet_nested_cumulative_moment(total, a, b, k)
                cross[i, j] = moment
                cross[j, i] = moment

        e2 = top * top - 2.0 * top * float(np.dot(gaps, e_ak))
        e2 += float(gaps @ cross @ gaps)

        v = max(0.0, e2 - m * m)
        means[row] = m
        vars_[row] = v

    mu = float(np.mean(means))
    sigma = float(math.sqrt(float(np.sum(vars_))) / M)
    return mu, sigma, levels


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
    :func:`scorio.eval.bayes`. When ``k = 1``, ``Max@1`` reduces to the usual
    single-draw expected score, so this function agrees with
    :func:`scorio.eval.bayes_ci`.

    This uncertainty model is a ``scorio`` extension. Bagirov et al. (2025)
    define the finite-sample max@k point estimator, but do not derive these
    Bayesian credible intervals.

    Args:
        R: :math:`M \times N` categorical outcome matrix with integer entries
           in :math:`\{0, \ldots, C\}`.
        k: Selection count. The posterior target is defined for any integer
           ``k >= 1``; ``k = 1`` matches :func:`scorio.eval.bayes_ci`.
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
    if k == 1:
        return bayes_ci(R, w=w, R0=R0, confidence=confidence, bounds=bounds)

    mu, sigma, levels = _max_at_k_bayes(R, k, w=w, R0=R0)
    if bounds is None:
        bounds = (float(np.min(levels)), float(np.max(levels)))
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


__all__ = [
    "max_at_k",
    "max_at_k_ci",
]
