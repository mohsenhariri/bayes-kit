r"""Majority-vote metrics and uncertainty estimators for binary outcomes.

This module evaluates whether a sampled subset contains a strict majority of
correct traces. It is a binary-outcome wrapper around the generalized
threshold-pass family.

Methods
-------
- ``maj_at_k``: probability that more than half of the ``k`` selected traces
  are correct.

Each metric has a companion ``*_ci`` function that returns
``(mu, sigma, lo, hi)`` under the Bayesian uncertainty model used here.
"""

import numpy as np

from ._count_score import CountScore
from ._inputs import prepare_binary_bank, validate_finite_k
from ._posterior import posterior_moments
from .utils import normal_credible_interval


def maj_at_k(R: np.ndarray, k: int) -> float:
    r"""
    Maj@k: strict-majority correctness over ``k`` samples.

    This metric measures the probability that a uniformly sampled subset of
    ``k`` observed traces contains strictly more than half correct solutions.
    It is a binary-outcome proxy for the majority-vote metrics often reported
    in reasoning papers when only correctness labels are available.

    Args:
        R: :math:`M \times N` binary outcome matrix.
        k: Number of sampled traces, with ``1 <= k <= N``.

    Returns:
        float: Average strict-majority success probability across prompts.

    Formula:
        Let :math:`\nu_\alpha = \sum_i R_{\alpha i}` and
        :math:`j_0 = \lfloor k/2 \rfloor + 1`. Then

        .. math::

            \mathrm{Maj@k}_\alpha = \sum_{j=j_0}^{k}
            \frac{\binom{\nu_\alpha}{j}\binom{N-\nu_\alpha}{k-j}}
                 {\binom{N}{k}}.

        Equivalently, this is :math:`\mathrm{G\text{-}Pass@k}_{\tau}` with
        :math:`\tau = j_0 / k`.

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(maj_at_k(R, 1), 6)
        0.7
        >>> round(maj_at_k(R, 2), 6)
        0.45
        >>> round(maj_at_k(R, 3), 6)
        0.85
    """
    bank = prepare_binary_bank(R)
    k = validate_finite_k(bank.trial_count, k)
    threshold = (k // 2) + 1
    return CountScore.threshold_at_k(k, threshold).mean(bank)


def maj_at_k_ci(
    R: np.ndarray,
    k: int,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    r"""
    Bayesian posterior summary for :func:`maj_at_k`.

    This reuses the generalized threshold-pass posterior with the strict
    majority threshold :math:`j_0 = \lfloor k/2 \rfloor + 1`.

    Args:
        R: :math:`M \times N` binary outcome matrix.
        k: Number of sampled traces, with ``1 <= k <= N``.
        confidence: Credibility level for the normal-approximation interval.
        bounds: ``(lo, hi)`` clipping bounds for the interval.
        alpha0: Beta prior parameter :math:`\alpha_0`.
        beta0: Beta prior parameter :math:`\beta_0`.

    Returns:
        tuple[float, float, float, float]:
            :math:`(\mu,\; \sigma,\; \text{lo},\; \text{hi})`

    Formula:
        This is exactly :func:`~scorio.eval.g_pass_at_k_tau_ci` evaluated at

        .. math::

            \tau = \frac{\lfloor k/2 \rfloor + 1}{k}.

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> mu, sigma, lo, hi = maj_at_k_ci(R, 2)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.446429, 0.146167, 0.1599, 0.7329)
        >>> mu, sigma, lo, hi = maj_at_k_ci(R, 3)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.684524, 0.151958, 0.3867, 0.9824)
    """
    bank = prepare_binary_bank(R)
    k = validate_finite_k(bank.trial_count, k)
    threshold = (k // 2) + 1
    moments = posterior_moments(
        bank,
        CountScore.threshold_at_k(k, threshold),
        alpha0=alpha0,
        beta0=beta0,
    )
    mu = moments.mean
    sigma = float(np.sqrt(moments.variance))
    lo, hi = normal_credible_interval(
        mu,
        sigma,
        credibility=confidence,
        two_sided=True,
        bounds=bounds,
    )
    return float(mu), sigma, float(lo), float(hi)


__all__ = [
    "maj_at_k",
    "maj_at_k_ci",
]
