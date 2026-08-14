r"""Generalized pass-family evaluation metrics for binary outcomes.

Estimate stability-style pass metrics from a binary outcome matrix
:math:`R \in \{0,1\}^{M \times N}`. ``G-Pass@k`` is the all-success
threshold, ``G-Pass@k``\ :sub:`\tau` requires at least
:math:`\max(1, \lceil \tau k \rceil)` successes, and ``mG-Pass@k`` uses
the cited discrete approximation over thresholds
:math:`\tau \in [0.5, 1.0]`. Bayesian summaries compute posterior
``mu`` and ``sigma`` under a Beta model for each question's latent success
rate.

Available API
-------------------
- ``g_pass_at_k`` returns the :math:`\tau = 1` alias of ``pass_hat_k``.
- ``g_pass_at_k_tau`` returns the thresholded generalized pass metric.
- ``mg_pass_at_k`` returns the mean generalized pass metric.
- companion ``*_ci`` functions return ``(mu, sigma, lo, hi)`` using a
  normal-approximation credible interval around ``mu``.
"""

import math

import numpy as np

from ._count_score import CountScore
from ._inputs import prepare_binary_bank, validate_finite_k
from ._posterior import posterior_moments
from .pass_at_k import pass_hat_k, pass_hat_k_ci
from .utils import normal_credible_interval


def g_pass_at_k(R: np.ndarray, k: int) -> float:
    r"""
    Performance evaluation using G-Pass@k.

    Equivalent to :func:`~scorio.eval.pass_hat_k`, included for literature
    that uses the G-Pass@k naming convention for the
    :math:`\tau = 1` threshold.

    References:
        Liu, J., Liu, H., Xiao, L., et al. (2025).
        Are Your LLMs Capable of Stable Reasoning?
        *Findings of ACL 2025*, 17594--17632.
        https://doi.org/10.18653/v1/2025.findings-acl.905

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0, 1\}`.
           :math:`R_{\alpha i} = 1` if trial :math:`i` for question
           :math:`\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \le k \le N`).

    Returns:
        float: The average G-Pass@k score across all :math:`M` questions.

    Notation:
        For each row :math:`\alpha`:

        .. math::

            \nu_\alpha = \sum_{i=1}^{N} R_{\alpha i} \quad \text{(number of correct samples)}

        :math:`C(a, b)` denotes the binomial coefficient :math:`\binom{a}{b}`.

    Formula:
        .. math::

            \mathrm{G\text{-}Pass@}k_\alpha = \frac{C(\nu_\alpha, k)}{C(N, k)}

        .. math::

            \mathrm{G\text{-}Pass@}k = \frac{1}{M} \sum_{\alpha=1}^{M}
                \mathrm{G\text{-}Pass@}k_\alpha

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(g_pass_at_k(R, 1), 6)
        0.7
        >>> round(g_pass_at_k(R, 2), 6)
        0.45
    """
    return pass_hat_k(R, k)


def g_pass_at_k_tau(R: np.ndarray, k: int, tau: float) -> float:
    r"""
    Performance evaluation using G-Pass@k\ :sub:`τ`.

    References:
        Liu, J., Liu, H., Xiao, L., et al. (2025).
        Are Your LLMs Capable of Stable Reasoning?
        *Findings of ACL 2025*, 17594--17632.
        https://doi.org/10.18653/v1/2025.findings-acl.905

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0, 1\}`.
           :math:`R_{\alpha i} = 1` if trial :math:`i` for question
           :math:`\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \le k \le N`).
        tau: Threshold parameter :math:`\tau \in [0, 1]`. Requires at
             least :math:`\max(1, \lceil \tau \cdot k \rceil)` successes.
             When :math:`\tau = 0`, equivalent to Pass@k.
             When :math:`\tau = 1`, equivalent to Pass^k.

    Returns:
        float: The average G-Pass@k\ :sub:`τ` score across all :math:`M` questions.

    Notation:
        For each row :math:`\alpha`:

        .. math::

            \nu_\alpha = \sum_{i=1}^{N} R_{\alpha i} \quad \text{(number of correct samples)}

        :math:`C(a, b)` denotes the binomial coefficient :math:`\binom{a}{b}`.

        :math:`j_0 = \max(1, \lceil \tau \cdot k \rceil)` is the minimum
        number of successes required. The lower bound of one makes
        :math:`\tau=0` equivalent to Pass@k rather than a vacuous event.

    Formula:
        .. math::

            \mathrm{G\text{-}Pass@}k_{\tau, \alpha} = \sum_{j=j_0}^{k}
                \frac{C(\nu_\alpha, j) \cdot C(N - \nu_\alpha, k - j)}{C(N, k)}

        .. math::

            \mathrm{G\text{-}Pass@}k_\tau = \frac{1}{M} \sum_{\alpha=1}^{M}
                \mathrm{G\text{-}Pass@}k_{\tau, \alpha}

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(g_pass_at_k_tau(R, 2, 0.5), 6)
        0.95
        >>> round(g_pass_at_k_tau(R, 2, 1.0), 6)
        0.45
    """
    bank = prepare_binary_bank(R)

    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be in [0, 1]; got {tau}")
    k = validate_finite_k(bank.trial_count, k)

    if tau <= 0.0:
        return CountScore.pass_at_k(k).mean(bank)

    threshold = max(1, int(math.ceil(tau * k)))
    return CountScore.threshold_at_k(k, threshold).mean(bank)


def mg_pass_at_k(R: np.ndarray, k: int) -> float:
    r"""
    Performance evaluation using mG-Pass@k.

    References:
        Liu, J., Liu, H., Xiao, L., et al. (2025).
        Are Your LLMs Capable of Stable Reasoning?
        *Findings of ACL 2025*, 17594--17632.
        https://doi.org/10.18653/v1/2025.findings-acl.905

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0, 1\}`.
           :math:`R_{\alpha i} = 1` if trial :math:`i` for question
           :math:`\alpha` passed, 0 otherwise.
        k: Number of samples to select (:math:`1 \le k \le N`).

    Returns:
        float: The average mG-Pass@k score across all :math:`M` questions.

    Notation:
        For each row :math:`\alpha`:

        .. math::

            \nu_\alpha = \sum_{i=1}^{N} R_{\alpha i} \quad \text{(number of correct samples)}

        :math:`m = \lceil k/2 \rceil` is the majority threshold.
        Let :math:`X_\alpha \sim \mathrm{Hypergeometric}(N, \nu_\alpha, k)`
        be the number of correct samples among :math:`k` selections for row
        :math:`\alpha`.

    Formula:
        The cited G-Pass metric defines mG-Pass by the following discrete
        summation. It is a right-endpoint approximation to the normalized
        continuous area over :math:`\tau \in [0.5,1]`, rather than an exact
        integral identity for arbitrary (in particular, odd) :math:`k`.

        .. math::

            \mathrm{mG\text{-}Pass@}k_\alpha \approx 2 \int_{0.5}^{1.0}
                \mathrm{G\text{-}Pass@}k_{\tau, \alpha} \, d\tau

        .. math::

            \mathrm{mG\text{-}Pass@}k_\alpha = \frac{2}{k} \sum_{j=m+1}^{k}
                (j - m) \cdot P(X_\alpha = j)

        .. math::

            P(X_\alpha = j) = \frac{C(\nu_\alpha, j) \cdot C(N - \nu_\alpha, k - j)}{C(N, k)}

        .. math::

            \mathrm{mG\text{-}Pass@}k = \frac{1}{M} \sum_{\alpha=1}^{M}
                \mathrm{mG\text{-}Pass@}k_\alpha

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(mg_pass_at_k(R, 2), 6)
        0.45
        >>> round(mg_pass_at_k(R, 3), 6)
        0.166667
    """
    bank = prepare_binary_bank(R)
    k = validate_finite_k(bank.trial_count, k)
    return CountScore.mg_at_k(k).mean(bank)


def _g_pass_at_k_tau_bayes(
    R: np.ndarray, k: int, tau: float, alpha0: float = 1.0, beta0: float = 1.0
) -> tuple[float, float]:
    """Posterior mean/std for the i.i.d. G-Pass@k_τ quantity."""
    bank = prepare_binary_bank(R)
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be in [0, 1]; got {tau}")
    k = validate_finite_k(bank.trial_count, k)
    threshold = max(1, int(math.ceil(tau * k)))
    moments = posterior_moments(
        bank,
        CountScore.threshold_at_k(k, threshold),
        alpha0=alpha0,
        beta0=beta0,
    )
    return moments.mean, float(math.sqrt(moments.variance))


def _mg_pass_at_k_bayes(
    R: np.ndarray, k: int, alpha0: float = 1.0, beta0: float = 1.0
) -> tuple[float, float]:
    """Posterior mean/std for the i.i.d. mG-Pass@k quantity."""
    bank = prepare_binary_bank(R)
    k = validate_finite_k(bank.trial_count, k)
    moments = posterior_moments(
        bank,
        CountScore.mg_at_k(k),
        alpha0=alpha0,
        beta0=beta0,
    )
    return moments.mean, float(math.sqrt(moments.variance))


def g_pass_at_k_ci(
    R: np.ndarray,
    k: int,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    r"""
    Bayesian posterior summary for G-Pass@k.

    G-Pass@k is the all-success threshold, so this is the same posterior
    target as :func:`~scorio.eval.pass_hat_k_ci`.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
        k: Number of selected samples with ``1 <= k <= N``.
        confidence: credibility level of the interval.
        bounds: ``(lo, hi)`` clipping bounds for the interval.
        alpha0: Beta prior parameter :math:`\alpha_0`.
        beta0: Beta prior parameter :math:`\beta_0`.

    Returns:
        tuple[float, float, float, float]:
            :math:`(\mu,\; \sigma,\; \text{lo},\; \text{hi})`.
    """
    return pass_hat_k_ci(
        R, k, confidence=confidence, bounds=bounds, alpha0=alpha0, beta0=beta0
    )


def g_pass_at_k_tau_ci(
    R: np.ndarray,
    k: int,
    tau: float,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    r"""
    Bayesian posterior summary for thresholded G-Pass@k.

    The latent target is the probability that at least
    :math:`\lceil \tau k \rceil` of ``k`` i.i.d. samples are correct.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
        k: Number of selected samples with ``1 <= k <= N``.
        tau: Threshold parameter in ``[0, 1]``.
        confidence: credibility level of the interval.
        bounds: ``(lo, hi)`` clipping bounds for the interval.
        alpha0: Beta prior parameter :math:`\alpha_0`.
        beta0: Beta prior parameter :math:`\beta_0`.

    Returns:
        tuple[float, float, float, float]:
            :math:`(\mu,\; \sigma,\; \text{lo},\; \text{hi})`.
    """
    mu, sigma = _g_pass_at_k_tau_bayes(R, k, tau, alpha0=alpha0, beta0=beta0)
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


def mg_pass_at_k_ci(
    R: np.ndarray,
    k: int,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    r"""
    Bayesian posterior summary for mG-Pass@k.

    The latent target uses the published discrete threshold weighting from
    ``0.5`` to ``1.0`` implemented by :func:`~scorio.eval.mg_pass_at_k`.
    This weighting approximates, but for odd ``k`` does not exactly equal,
    the corresponding continuous area.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
        k: Number of selected samples with ``1 <= k <= N``.
        confidence: credibility level of the interval.
        bounds: ``(lo, hi)`` clipping bounds for the interval.
        alpha0: Beta prior parameter :math:`\alpha_0`.
        beta0: Beta prior parameter :math:`\beta_0`.

    Returns:
        tuple[float, float, float, float]:
            :math:`(\mu,\; \sigma,\; \text{lo},\; \text{hi})`.
    """
    mu, sigma = _mg_pass_at_k_bayes(R, k, alpha0=alpha0, beta0=beta0)
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


__all__ = [
    "g_pass_at_k",
    "g_pass_at_k_tau",
    "mg_pass_at_k",
    "g_pass_at_k_ci",
    "g_pass_at_k_tau_ci",
    "mg_pass_at_k_ci",
]
