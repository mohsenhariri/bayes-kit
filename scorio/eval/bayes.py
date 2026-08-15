r"""Bayes evaluation metrics for categorical outcomes.

Estimate :math:`\mu` and uncertainty (:math:`\sigma`) from repeated outcomes with optional
prior observations. :math:`Bayes@N` supports binary and multi-category outcomes
through a category-weight vector.

Let :math:`R \in \{0,\ldots,C\}^{M \times N}` be observed outcomes,
:math:`w \in \mathbb{R}^{C+1}` be category weights, and optional
:math:`R^0 \in \{0,\ldots,C\}^{M \times D}` be prior outcomes.
For each question :math:`\alpha` and class :math:`k`, Bayes@N forms counts
from :math:`R` and :math:`R^0`, adds Dirichlet plus one pseudo-counts, and
computes closed-form posterior moments ``mu`` and ``sigma``.

Available API
-------------------
- ``bayes`` returns ``(mu, sigma)``.
- ``bayes_ci`` returns ``(mu, sigma, lo, hi)`` using a normal-approximation
  credible interval around ``mu``.
"""

import numpy as np

from ._categorical import bayes_moments, prepare_categorical_bank
from .utils import normal_credible_interval


def bayes(
    R: np.ndarray,
    w: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> tuple[float, float]:
    r"""
    Performance evaluation using the Bayes@N framework.

    References:
        Hariri, M., Samandar, A., Hinczewski, M., & Chaudhary, V. (2026).
        Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation.
        *ICLR 2026*, *arXiv:2510.04265*.
        https://arxiv.org/abs/2510.04265

    Args:
        R: :math:`M \times N` int matrix with entries in :math:`\{0,\ldots,C\}`.
           Row :math:`\alpha` are the N outcomes for question :math:`\alpha`.
        w: length :math:`(C+1)` weight vector :math:`(w_0,\ldots,w_C)` that maps
           category k to score :math:`w_k`.
        R0: optional :math:`M \times D` int matrix supplying D prior outcomes per row.
             If omitted, :math:`D=0`.

    Returns:
        tuple[float, float]: :math:`(\mu, \sigma)` performance metric estimate and its uncertainty.

    Notation:
        :math:`\delta_{a,b}` is the Kronecker delta. For each row :math:`\alpha` and class :math:`k \in \{0,\ldots,C\}`:

        .. math::

            n_{\alpha k} &= \sum_{i=1}^N \delta_{k, R_{\alpha i}} \quad \text{(counts in R)}

            n^0_{\alpha k} &= 1 + \sum_{i=1}^D \delta_{k, R^0_{\alpha i}} \quad \text{(Dirichlet(+1) prior)}

            \nu_{\alpha k} &= n_{\alpha k} + n^0_{\alpha k}

        Effective sample size: :math:`T = 1 + C + D + N` (scalar)

    Formula:
        .. math::

            \mu = w_0 + \frac{1}{M \cdot T} \sum_{\alpha=1}^M \sum_{j=0}^C \nu_{\alpha j} (w_j - w_0)

        .. math::

            \sigma = \sqrt{ \frac{1}{M^2(T+1)} \sum_{\alpha=1}^M \left[
                \sum_j \frac{\nu_{\alpha j}}{T} (w_j - w_0)^2
                - \left( \sum_j \frac{\nu_{\alpha j}}{T} (w_j - w_0) \right)^2 \right] }

    Examples:
        >>> import numpy as np
        >>> R  = np.array([[0, 1, 2, 2, 1],
        ...                [1, 1, 0, 2, 2]])
        >>> w  = np.array([0.0, 0.5, 1.0])
        >>> R0 = np.array([[0, 2],
        ...                [1, 2]])

        With prior (D=2 → T=10):

        >>> mu, sigma = bayes(R, w, R0)
        >>> round(mu, 6), round(sigma, 6)
        (0.575, 0.084275)

        Without prior (D=0 → T=8):

        >>> mu2, sigma2 = bayes(R, w)
        >>> round(mu2, 6), round(sigma2, 6)
        (0.5625, 0.091998)

    """
    return bayes_moments(prepare_categorical_bank(R, w=w, R0=R0))


def bayes_ci(
    R: np.ndarray,
    w: np.ndarray | None = None,
    R0: np.ndarray | None = None,
    confidence: float = 0.95,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float, float, float]:
    r"""
    Bayes@N posterior mean, uncertainty, and credible interval.

    This is the interval-valued companion to :func:`bayes`. It computes
    :math:`\mu` and :math:`\sigma` with the Bayes@N posterior moments, then
    forms a central normal-approximation credible interval.

    Args:
        R: :math:`M \times N` int matrix with entries in
           :math:`\{0,\ldots,C\}`.
        w: optional length :math:`(C+1)` weight vector
           :math:`(w_0,\ldots,w_C)`. If omitted, ``R`` must be binary and
           ``w = [0, 1]`` is used.
        R0: optional :math:`M \times D` int matrix supplying prior outcomes
            for each row.
        confidence: credibility level of the interval.
        bounds: optional ``(lo, hi)`` clipping bounds for the interval.

    Returns:
        tuple[float, float, float, float]:
            :math:`(\mu,\; \sigma,\; \text{lo},\; \text{hi})`.

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> mu, sigma, lo, hi = bayes_ci(R, bounds=(0.0, 1.0))
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.642857, 0.118451, 0.4107, 0.875)
    """
    mu, sigma = bayes(R, w, R0)
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


__all__ = [
    "bayes",
    "bayes_ci",
]
