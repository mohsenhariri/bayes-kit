r"""AUC@K evaluation metrics for binary outcomes.

Estimate normalized area under the Pass@j curve for budgets
:math:`j = 1, \ldots, k`. For a binary outcome matrix
:math:`R \in \{0,1\}^{M \times N}`, :math:`AUC@K` averages per-question
Pass@j values using trapezoidal weights matching Eq. (7) of Hu et al. (2026).
The Bayesian summary computes posterior ``mu`` and ``sigma`` under a Beta
model for each question's latent success rate.

Available API
-------------------
- ``auc_at_k`` returns the point estimate.
- ``auc_at_k_ci`` returns ``(mu, sigma, lo, hi)`` using a normal-approximation
  credible interval around ``mu``.
"""

import math

import numpy as np
from scipy.special import comb

from .pass_at_k import (
    _beta_ratio,
    _binary_beta_posterior_params,
    pass_at_k,
    pass_at_k_ci,
)
from .utils import _as_2d_int_matrix, _validate_binary, normal_credible_interval


def _validate_k(N: int, k: int) -> None:
    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")


def _pass_at_k_values_from_counts(nu: np.ndarray, N: int, k: int) -> np.ndarray:
    """Vectorized Pass@k values from per-row success counts."""
    denom = comb(N, k)
    return 1.0 - comb(N - nu, k) / denom


def _auc_at_k_coefficients(k: int) -> np.ndarray:
    """Eq. (7) trapezoidal-rule coefficients for AUC@K over Pass@1..Pass@K."""
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if k == 1:
        return np.array([1.0], dtype=float)
    coeff = np.full(k, 1.0 / (k - 1), dtype=float)
    coeff[0] = 0.5 / (k - 1)
    coeff[-1] = 0.5 / (k - 1)
    return coeff


def auc_at_k(R: np.ndarray, k: int) -> float:
    r"""
    Performance evaluation using AUC@K.

    References:
        Hu, Z., et al. (2026).
        Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in
        LLMs. *arXiv:2601.08763*.
        https://arxiv.org/abs/2601.08763

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0, 1\}`.
           :math:`R_{\alpha i} = 1` if trial :math:`i` for question
           :math:`\alpha` passed, 0 otherwise.
        k: Maximum sampling budget (:math:`1 \le k \le N`).

    Returns:
        float: The average AUC@K score across all :math:`M` questions.

    Notation:
        For each row :math:`\alpha`:

        .. math::

            \nu_\alpha = \sum_{i=1}^{N} R_{\alpha i} \quad \text{(number of correct samples)}

            \mathrm{Pass@}j_\alpha = 1 - \frac{\binom{N - \nu_\alpha}{j}}{\binom{N}{j}}

        For :math:`k > 1`, define trapezoidal coefficients
        :math:`c_1 = c_k = \frac{1}{2(k-1)}` and
        :math:`c_j = \frac{1}{k-1}` for :math:`2 \le j \le k-1`.
        For :math:`k = 1`, :math:`\mathrm{AUC@1} = \mathrm{Pass@1}`.

    Formula:
        .. math::

            \mathrm{AUC@}k_\alpha = \sum_{j=1}^{k} c_j \, \mathrm{Pass@}j_\alpha

        .. math::

            \mathrm{AUC@}k = \frac{1}{M} \sum_{\alpha=1}^{M} \mathrm{AUC@}k_\alpha

        Equivalently, for :math:`k > 1`,

        .. math::

            \mathrm{AUC@}k =
            \frac{1}{k - 1} \sum_{j=1}^{k-1}
            \frac{\mathrm{Pass@}j + \mathrm{Pass@}(j + 1)}{2}

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(auc_at_k(R, 1), 6)
        0.7
        >>> round(auc_at_k(R, 2), 6)
        0.825
        >>> round(auc_at_k(R, 3), 6)
        0.9
    """
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    _, N = Rm.shape
    _validate_k(N, k)

    if k == 1:
        return pass_at_k(Rm, 1)

    nu = np.sum(Rm, axis=1)
    coeff = _auc_at_k_coefficients(k)

    vals = np.zeros(Rm.shape[0], dtype=float)
    for j, c_j in enumerate(coeff, start=1):
        vals += c_j * _pass_at_k_values_from_counts(nu, N, j)
    return float(np.mean(vals))


def _auc_at_k_bayes(
    R: np.ndarray, k: int, alpha0: float = 1.0, beta0: float = 1.0
) -> tuple[float, float]:
    """Posterior mean/std for :func:`auc_at_k`."""
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    M, N = Rm.shape
    _validate_k(N, k)

    alpha, beta = _binary_beta_posterior_params(Rm, alpha0=alpha0, beta0=beta0)
    coeff = _auc_at_k_coefficients(k)
    js = np.arange(1, k + 1, dtype=int)

    means = np.empty(M, dtype=float)
    vars_ = np.empty(M, dtype=float)

    # Eq. (7) becomes a weighted sum of Pass@j terms, and for Bernoulli
    # success rate p we use Pass@j(p) = 1 - (1 - p)^j.
    for i in range(M):
        a_i = float(alpha[i])
        b_i = float(beta[i])

        eq = np.array([_beta_ratio(a_i, b_i, 0, int(j)) for j in js], dtype=float)
        m = 1.0 - float(np.dot(coeff, eq))

        e2 = 1.0
        e2 -= 2.0 * float(np.dot(coeff, eq))
        for idx_j, j in enumerate(js):
            c_j = float(coeff[idx_j])
            for idx_l, l in enumerate(js):
                c_l = float(coeff[idx_l])
                e2 += c_j * c_l * _beta_ratio(a_i, b_i, 0, int(j + l))

        v = max(0.0, e2 - m * m)
        means[i] = m
        vars_[i] = v

    mu = float(np.mean(means))
    sigma = float(math.sqrt(float(np.sum(vars_))) / M)
    return mu, sigma


def auc_at_k_ci(
    R: np.ndarray,
    k: int,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    r""":math:`AUC@K`'s :math:`\mu` and :math:`\sigma` plus a normal-approx credible interval (CrI)."""
    if k == 1:
        return pass_at_k_ci(
            R,
            1,
            confidence=confidence,
            bounds=bounds,
            alpha0=alpha0,
            beta0=beta0,
        )

    mu, sigma = _auc_at_k_bayes(R, k, alpha0=alpha0, beta0=beta0)
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


__all__ = [
    "auc_at_k",
    "auc_at_k_ci",
]
